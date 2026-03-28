"""
Rebuild the Silero TTS v5 model as plain nn.Module, load weights, replace ISTFT,
and export to ONNX.

This avoids both blockers:
- No ScriptModule → no param count assertion
- Real-valued ISTFT → no complex numbers
"""
import sys
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ===========================================================================
# Model architecture (replicated from JIT model structure)
# ===========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, pe_data, scale_data):
        super().__init__()
        self.register_buffer("pe", torch.from_numpy(pe_data))
        self.register_buffer("scale", torch.from_numpy(scale_data))

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x * self.scale
        x = x + self.pe[:x.size(0)]
        return x


class FFTBlock(nn.Module):
    """Single FFT transformer block (self-attention + conv feedforward)."""
    def __init__(self, d_model, nhead, conv1_w, conv1_b, conv2_w, conv2_b,
                 attn_in_proj_w, attn_in_proj_b, attn_out_proj_w, attn_out_proj_b,
                 norm1_w, norm1_b, norm2_w, norm2_b):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=False)
        self.self_attn.in_proj_weight = nn.Parameter(torch.from_numpy(attn_in_proj_w))
        self.self_attn.in_proj_bias = nn.Parameter(torch.from_numpy(attn_in_proj_b))
        self.self_attn.out_proj.weight = nn.Parameter(torch.from_numpy(attn_out_proj_w))
        self.self_attn.out_proj.bias = nn.Parameter(torch.from_numpy(attn_out_proj_b))

        # Conv feedforward
        out_ch = conv1_w.shape[0]
        in_ch = conv1_w.shape[1]
        kernel = conv1_w.shape[2]
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.conv1.weight = nn.Parameter(torch.from_numpy(conv1_w))
        self.conv1.bias = nn.Parameter(torch.from_numpy(conv1_b))

        out_ch2 = conv2_w.shape[0]
        in_ch2 = conv2_w.shape[1]
        self.conv2 = nn.Conv1d(in_ch2, out_ch2, 1)
        self.conv2.weight = nn.Parameter(torch.from_numpy(conv2_w))
        self.conv2.bias = nn.Parameter(torch.from_numpy(conv2_b))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm1.weight = nn.Parameter(torch.from_numpy(norm1_w))
        self.norm1.bias = nn.Parameter(torch.from_numpy(norm1_b))

        self.norm2 = nn.LayerNorm(d_model)
        self.norm2.weight = nn.Parameter(torch.from_numpy(norm2_w))
        self.norm2.bias = nn.Parameter(torch.from_numpy(norm2_b))

    def forward(self, x, mask=None):
        # Manual self-attention (avoids nn.MultiheadAttention tracing issues)
        residual = x  # [L, B, C]

        # QKV projection
        L, B, C = x.shape
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head: [L, B, C] -> [B, nhead, L, head_dim]
        nhead = self.self_attn.num_heads
        head_dim = C // nhead
        q = q.reshape(L, B, nhead, head_dim).permute(1, 2, 0, 3)  # [B, nhead, L, head_dim]
        k = k.reshape(L, B, nhead, head_dim).permute(1, 2, 0, 3)
        v = v.reshape(L, B, nhead, head_dim).permute(1, 2, 0, 3)

        # Scaled dot-product attention
        scale = math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, nhead, L, L]

        if mask is not None:
            # mask: [B, L] bool, True = ignore
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # [B, nhead, L, head_dim]

        # Reshape back: [B, nhead, L, head_dim] -> [L, B, C]
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(L, B, C)

        # Output projection
        attn_out = F.linear(attn_out, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)

        x = self.norm1(residual + attn_out)

        # Conv feedforward
        residual = x
        x_conv = x.permute(1, 2, 0)  # [B, C, L]
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = self.conv2(x_conv)
        x_conv = x_conv.permute(2, 0, 1)  # [L, B, C]
        x = self.norm2(residual + x_conv)
        return x


class ForwardTransformer(nn.Module):
    def __init__(self, params, prefix, d_model, nhead, num_layers):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model,
            params[f"{prefix}.pos_encoder.pe"],
            params[f"{prefix}.pos_encoder.scale"],
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            p = f"{prefix}.layers.{i}"
            block = FFTBlock(
                d_model, nhead,
                params[f"{p}.conv1.weight"], params[f"{p}.conv1.bias"],
                params[f"{p}.conv2.weight"], params[f"{p}.conv2.bias"],
                params[f"{p}.self_attn.in_proj_weight"], params[f"{p}.self_attn.in_proj_bias"],
                params[f"{p}.self_attn.out_proj.weight"], params[f"{p}.self_attn.out_proj.bias"],
                params[f"{p}.norm1.weight"], params[f"{p}.norm1.bias"],
                params[f"{p}.norm2.weight"], params[f"{p}.norm2.bias"],
            )
            self.layers.append(block)

        self.norm = nn.LayerNorm(d_model)
        self.norm.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.norm.weight"]))
        self.norm.bias = nn.Parameter(torch.from_numpy(params[f"{prefix}.norm.bias"]))

    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class TransformerSeriesPredictor(nn.Module):
    """Duration or pitch predictor: embedding + transformer + linear."""
    def __init__(self, params, prefix, d_model=64, nhead=2, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(47, d_model)
        self.embedding.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.embedding.weight"]))

        self.speaker_embedding = nn.Embedding(4, d_model)
        self.speaker_embedding.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.speaker_embedding.weight"]))

        self.type_embedding = nn.Embedding(
            params[f"{prefix}.type_embedding.weight"].shape[0], d_model)
        self.type_embedding.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.type_embedding.weight"]))

        self.transformer = ForwardTransformer(params, f"{prefix}.transformer", d_model, nhead, num_layers)

        self.lin = nn.Linear(d_model, 1)
        self.lin.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.lin.weight"]))
        self.lin.bias = nn.Parameter(torch.from_numpy(params[f"{prefix}.lin.bias"]))

    def forward(self, sequence, speaker_ids, mask, rate=1.0, type_ids=None):
        # sequence: [B, L] int64
        x = self.embedding(sequence)  # [B, L, D]
        spk = self.speaker_embedding(speaker_ids)  # [B, D]
        x = x + spk.unsqueeze(1)

        # Add type embedding
        if type_ids is not None:
            x = x + self.type_embedding(type_ids)

        # Transpose for transformer: [L, B, D]
        x = x.transpose(0, 1)
        x = self.transformer(x, mask)
        x = x.transpose(0, 1)  # back to [B, L, D]

        out = self.lin(x).squeeze(-1)  # [B, L]
        return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, params, prefix, dim=512):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.dwconv.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.dwconv.weight"]))
        self.dwconv.bias = nn.Parameter(torch.from_numpy(params[f"{prefix}.dwconv.bias"]))

        self.norm = nn.LayerNorm(dim)
        self.norm.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.norm.weight"]))
        self.norm.bias = nn.Parameter(torch.from_numpy(params[f"{prefix}.norm.bias"]))

        self.pwconv1 = nn.Linear(dim, dim * 3)
        self.pwconv1.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.pwconv1.weight"]))
        self.pwconv1.bias = nn.Parameter(torch.from_numpy(params[f"{prefix}.pwconv1.bias"]))

        self.pwconv2 = nn.Linear(dim * 3, dim)
        self.pwconv2.weight = nn.Parameter(torch.from_numpy(params[f"{prefix}.pwconv2.weight"]))
        self.pwconv2.bias = nn.Parameter(torch.from_numpy(params[f"{prefix}.pwconv2.bias"]))

        self.register_buffer("gamma", torch.from_numpy(params[f"{prefix}.gamma"]))

    def forward(self, x):
        # x: [B, C, T]
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # [B, C, T]
        x = residual + x
        return x


class RealISTFT(nn.Module):
    """ISTFT using only real-valued operations."""
    def __init__(self, n_fft, hop_length, window_np):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.from_numpy(window_np))

        n_freq = n_fft // 2 + 1
        n = torch.arange(n_fft, dtype=torch.float32)
        k = torch.arange(n_freq, dtype=torch.float32)
        angles = 2.0 * math.pi * n[:, None] * k[None, :] / n_fft

        cos_b = torch.cos(angles)
        sin_b = torch.sin(angles)

        scale = torch.ones(n_freq) * 2.0 / n_fft
        scale[0] = 1.0 / n_fft
        if n_fft % 2 == 0:
            scale[-1] = 1.0 / n_fft

        cos_b = cos_b * scale[None, :]  # [n_fft, n_freq]
        sin_b = sin_b * scale[None, :]

        self.register_buffer("cos_basis", cos_b)
        self.register_buffer("sin_basis", sin_b)

    def forward(self, mag, cos_phase, sin_phase):
        """
        mag, cos_phase, sin_phase: [B, n_freq, T]
        Returns: audio [B, audio_len]
        """
        S_real = mag * cos_phase  # [B, n_freq, T]
        S_imag = mag * sin_phase

        # IDFT: frame[n] = sum_k (S_real[k]*cos[n,k] - S_imag[k]*sin[n,k])
        # [B, T, n_freq] @ [n_freq, n_fft] -> [B, T, n_fft]
        S_real_t = S_real.transpose(1, 2)
        S_imag_t = S_imag.transpose(1, 2)
        frames = S_real_t @ self.cos_basis.T - S_imag_t @ self.sin_basis.T

        # Apply window: [B, T, n_fft]
        frames = frames * self.window.unsqueeze(0).unsqueeze(0)

        # Overlap-add (manual, avoids F.fold which has ONNX issues)
        B, T_frames, N = frames.shape
        output_length = (T_frames - 1) * self.hop_length + self.n_fft

        # Build index tensor for scatter_add: each frame contributes to positions [t*hop : t*hop+n_fft]
        # Create output by iterating (unrolled in trace since T_frames is fixed per input)
        audio = torch.zeros(B, output_length, device=frames.device)
        win_env = torch.zeros(B, output_length, device=frames.device)
        win_sq = self.window ** 2

        for t in range(T_frames):
            start = t * self.hop_length
            audio[:, start:start + self.n_fft] += frames[:, t, :]
            win_env[:, start:start + self.n_fft] += win_sq

        audio = audio / (win_env + 1e-11)

        # Trim center padding
        c = self.n_fft // 2
        audio = audio[:, c:-c]
        return audio


class VocoderHead(nn.Module):
    """Vocoder head: Linear → split mag/phase → real ISTFT."""
    def __init__(self, params, n_fft, hop_length, window_np):
        super().__init__()
        self.out = nn.Linear(512, 2 * (n_fft // 2 + 1))
        self.out.weight = nn.Parameter(torch.from_numpy(params["vocoder.head.out.weight"]))
        self.out.bias = nn.Parameter(torch.from_numpy(params["vocoder.head.out.bias"]))
        self.istft = RealISTFT(n_fft, hop_length, window_np)

    def forward(self, x):
        # x: [B, C, T] from backbone
        x = self.out(x.transpose(1, 2))  # [B, T, 2*n_freq]
        x = x.transpose(1, 2)  # [B, 2*n_freq, T]

        # Split into magnitude and phase
        mag_input, phase_input = x.chunk(2, dim=1)

        mag = torch.exp(mag_input).clamp(max=100.0)
        cos_p = torch.cos(phase_input)
        sin_p = torch.sin(phase_input)

        audio = self.istft(mag, cos_p, sin_p)
        return audio


class Vocoder(nn.Module):
    def __init__(self, params, n_fft, hop_length, window_np):
        super().__init__()
        # Backbone
        self.embed = nn.Conv1d(192, 512, 7, padding=3)
        self.embed.weight = nn.Parameter(torch.from_numpy(params["vocoder.backbone.embed.weight"]))
        self.embed.bias = nn.Parameter(torch.from_numpy(params["vocoder.backbone.embed.bias"]))

        self.norm = nn.LayerNorm(512)
        self.norm.weight = nn.Parameter(torch.from_numpy(params["vocoder.backbone.norm.weight"]))
        self.norm.bias = nn.Parameter(torch.from_numpy(params["vocoder.backbone.norm.bias"]))

        self.convnext = nn.ModuleList([
            ConvNeXtBlock(params, f"vocoder.backbone.convnext.{i}") for i in range(8)
        ])

        self.final_norm = nn.LayerNorm(512)
        self.final_norm.weight = nn.Parameter(torch.from_numpy(params["vocoder.backbone.final_layer_norm.weight"]))
        self.final_norm.bias = nn.Parameter(torch.from_numpy(params["vocoder.backbone.final_layer_norm.bias"]))

        self.head = VocoderHead(params, n_fft, hop_length, window_np)

    def forward(self, mel):
        # mel: [B, C_mel, T]
        x = self.embed(mel)  # [B, 512, T]

        # Initial LayerNorm (needs transpose)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        for block in self.convnext:
            x = block(x)

        x = x.transpose(1, 2)
        x = self.final_norm(x)
        x = x.transpose(1, 2)

        audio = self.head(x)
        return audio


class SileroTTSv5(nn.Module):
    """Simplified Silero TTS v5 — tokens + speaker → audio.

    For ONNX export, we implement the default inference path only
    (no SSML, no ground-truth durations, no focus masks).
    """
    def __init__(self, params, n_fft, hop_length, window_np):
        super().__init__()

        # Token + speaker embeddings
        self.embedding = nn.Embedding(47, 128)
        self.embedding.weight = nn.Parameter(torch.from_numpy(params["tacotron.embedding.weight"]))

        self.speaker_embedding = nn.Embedding(4, 128)
        self.speaker_embedding.weight = nn.Parameter(torch.from_numpy(params["tacotron.speaker_embedding.weight"]))

        # Encoder
        self.encoder = ForwardTransformer(params, "tacotron.encoder", 128, 2, 4)

        # Duration predictor
        self.dur_predictor = TransformerSeriesPredictor(params, "dur_predictor.dur_pred", 64, 2, 4)

        # Pitch predictor
        self.pitch_predictor = TransformerSeriesPredictor(params, "pitch_predictor.pitch_pred", 64, 2, 4)
        self.pitch_proj = nn.Conv1d(1, 128, 3, padding=1)
        self.pitch_proj.weight = nn.Parameter(torch.from_numpy(params["pitch_predictor.pitch_proj.weight"]))
        self.pitch_proj.bias = nn.Parameter(torch.from_numpy(params["pitch_predictor.pitch_proj.bias"]))

        # Decoder components (HourGlass Transformer) (HourGlass Transformer)
        self.decoder_pos = PositionalEncoding(
            128,
            params["tacotron.decoder.pos_encoder.pe"],
            params["tacotron.decoder.pos_encoder.scale"],
        )
        self.decoder_pre_layers = nn.ModuleList([
            self._make_fft_block(params, "tacotron.decoder.pre_vanilla_layers.0", 128, 2),
        ])
        self.decoder_shorten_layers = nn.ModuleList([
            self._make_fft_block(params, f"tacotron.decoder.shorten_layers.{i}", 128, 2)
            for i in range(2)
        ])
        self.decoder_post_layers = nn.ModuleList([
            self._make_fft_block(params, "tacotron.decoder.post_vanilla_layers.0", 128, 2),
        ])
        self.decoder_norm = nn.LayerNorm(128)
        self.decoder_norm.weight = nn.Parameter(torch.from_numpy(params["tacotron.decoder.norm.weight"]))
        self.decoder_norm.bias = nn.Parameter(torch.from_numpy(params["tacotron.decoder.norm.bias"]))

        # Downsample/upsample for hourglass
        self.downsample_pool = nn.AvgPool1d(2, ceil_mode=True)
        upsample_w = params["tacotron.decoder.upsample.proj.weight"]  # [384, 128]
        upsample_b = params["tacotron.decoder.upsample.proj.bias"]    # [384]
        self.upsample_proj = nn.Linear(128, upsample_w.shape[0])
        self.upsample_proj.weight = nn.Parameter(torch.from_numpy(upsample_w))
        self.upsample_proj.bias = nn.Parameter(torch.from_numpy(upsample_b))

        # Pitch/energy projections in tacotron
        self.tac_pitch_proj = nn.Conv1d(1, 128, 3, padding=1)
        self.tac_pitch_proj.weight = nn.Parameter(torch.from_numpy(params["tacotron.pitch_proj.weight"]))
        self.tac_pitch_proj.bias = nn.Parameter(torch.from_numpy(params["tacotron.pitch_proj.bias"]))

        # Output linear
        self.lin = nn.Linear(128, 192)
        self.lin.weight = nn.Parameter(torch.from_numpy(params["tacotron.lin.weight"]))
        self.lin.bias = nn.Parameter(torch.from_numpy(params["tacotron.lin.bias"]))

        # Vocoder
        self.vocoder = Vocoder(params, n_fft, hop_length, window_np)

    def _make_fft_block(self, params, prefix, d_model, nhead):
        return FFTBlock(
            d_model, nhead,
            params[f"{prefix}.conv1.weight"], params[f"{prefix}.conv1.bias"],
            params[f"{prefix}.conv2.weight"], params[f"{prefix}.conv2.bias"],
            params[f"{prefix}.self_attn.in_proj_weight"], params[f"{prefix}.self_attn.in_proj_bias"],
            params[f"{prefix}.self_attn.out_proj.weight"], params[f"{prefix}.self_attn.out_proj.bias"],
            params[f"{prefix}.norm1.weight"], params[f"{prefix}.norm1.bias"],
            params[f"{prefix}.norm2.weight"], params[f"{prefix}.norm2.bias"],
        )

    def forward(self, sequence: torch.Tensor, speaker_ids: torch.Tensor) -> torch.Tensor:
        B, L = sequence.shape

        # Create mask
        mask = sequence == 0  # padding mask

        # Embeddings
        x = self.embedding(sequence)  # [B, L, 128]
        spk = self.speaker_embedding(speaker_ids)  # [B, 128]
        x = x + spk.unsqueeze(1)

        # Encode: [L, B, 128]
        x_enc = x.transpose(0, 1)
        x_enc = self.encoder(x_enc, mask)
        x_enc = x_enc.transpose(0, 1)  # [B, L, 128]

        # Duration prediction
        pred_log_dur = self.dur_predictor(sequence, speaker_ids, mask)
        durations = torch.clamp(torch.round(torch.exp(pred_log_dur) - 1), min=0).long()  # [B, L]

        # Length regulation: expand by durations
        x_expanded = self._length_regulate(x_enc, durations)  # [B, L', 128]

        # Pitch prediction
        pred_pitch = self.pitch_predictor(sequence, speaker_ids, mask)
        # Expand pitch to match regulated length
        pitch_expanded = self._length_regulate(pred_pitch.unsqueeze(-1), durations).squeeze(-1)
        pitch_proj = self.tac_pitch_proj(pitch_expanded.unsqueeze(1))  # [B, 128, L']
        x_expanded = x_expanded + pitch_proj.transpose(1, 2)

        # Decoder (simplified hourglass)
        L2 = x_expanded.shape[1]
        x_dec = x_expanded.transpose(0, 1)  # [L', B, 128]
        x_dec = self.decoder_pos(x_dec)

        # Pre-vanilla layers
        for layer in self.decoder_pre_layers:
            x_dec = layer(x_dec)

        # Downsample
        x_down = x_dec.permute(1, 2, 0)  # [B, 128, L']
        x_down = self.downsample_pool(x_down)  # [B, 128, L'/2]
        x_short = x_down.permute(2, 0, 1)  # [L'/2, B, 128]

        for layer in self.decoder_shorten_layers:
            x_short = layer(x_short)

        # Upsample
        x_up = x_short.transpose(0, 1)  # [B, L'/2, 128]
        x_up = self.upsample_proj(x_up)  # [B, L'/2, 384]
        x_up = x_up.reshape(B, -1, 128)  # [B, L'/2*3, 128]
        x_up = x_up[:, :L2, :]  # trim to original length
        x_up = x_up.transpose(0, 1)  # [L', B, 128]

        x_dec = x_dec + x_up

        # Post-vanilla layers
        for layer in self.decoder_post_layers:
            x_dec = layer(x_dec)

        x_dec = self.decoder_norm(x_dec)
        x_dec = x_dec.transpose(0, 1)  # [B, L', 128]

        # Linear to mel dimension
        mel = self.lin(x_dec)  # [B, L', 192]
        mel = mel.transpose(1, 2)  # [B, 192, L']

        # Vocoder
        audio = self.vocoder(mel)  # [B, audio_len]

        return audio

    def _length_regulate(self, x, durations):
        """Expand sequence by predicted durations using repeat_interleave (ONNX-compatible)."""
        # x: [B, L, D], durations: [B, L]
        # For batch size 1 (TTS inference), use repeat_interleave directly
        durs = durations[0].clamp(min=1).long()  # [L]
        expanded = torch.repeat_interleave(x[0], durs, dim=0)  # [L', D]
        return expanded.unsqueeze(0)  # [1, L', D]


def main():
    print("Loading extracted weights...")
    params = dict(np.load("assets/model_weights.npz", allow_pickle=True))

    with open("assets/istft_params.json") as f:
        istft = json.load(f)

    n_fft = istft["n_fft"]
    hop_length = istft["hop_length"]
    window = np.array(istft["window"], dtype=np.float32)

    print(f"Building model (n_fft={n_fft}, hop={hop_length})...")
    model = SileroTTSv5(params, n_fft, hop_length, window)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Model built: {total:,} parameters")

    # Test forward pass
    with open("assets/symbols.json", encoding="utf-8") as f:
        symbol_to_id = json.load(f)

    tokens = [symbol_to_id.get(c, 0) for c in "тест."]
    seq = torch.LongTensor([tokens])
    spk = torch.LongTensor([3])

    print("Running forward pass...")
    with torch.no_grad():
        audio = model(seq, spk)
    print(f"Output: {audio.shape}, range=[{audio.min():.4f}, {audio.max():.4f}]")

    # Save test audio
    try:
        import soundfile as sf
        sf.write("test_rebuilt.wav", audio.numpy().flatten(), 48000)
        print("Saved test_rebuilt.wav")
    except Exception as e:
        print(f"WAV: {e}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    output_file = "silero_v5.onnx"

    try:
        torch.onnx.export(
            model, (seq, spk), output_file,
            input_names=["input", "speaker"],
            output_names=["audio"],
            dynamic_axes={"input": {1: "seq_len"}, "audio": {1: "audio_len"}},
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"ONNX EXPORT SUCCESS: {output_file} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"ONNX export FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validate
    print("\nValidating...")
    import onnxruntime as ort
    sess = ort.InferenceSession(output_file)
    print(f"Inputs: {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

    ort_out = sess.run(None, {"input": seq.numpy(), "speaker": spk.numpy()})
    print(f"ORT: {ort_out[0].shape}, range=[{ort_out[0].min():.4f}, {ort_out[0].max():.4f}]")

    # Test different lengths
    for txt in ["привет.", "съешьте ещё этих мягких французских булочек."]:
        t = np.array([[symbol_to_id.get(c, 0) for c in txt]], dtype=np.int64)
        try:
            r = sess.run(None, {"input": t, "speaker": np.array([3], dtype=np.int64)})
            print(f"  '{txt[:40]}' -> {r[0].shape} ({r[0].shape[1]/48000:.2f}s)")
        except Exception as e:
            print(f"  '{txt[:40]}' -> {str(e)[:120]}")

    print(f"\nDONE: {output_file}")


if __name__ == "__main__":
    main()
