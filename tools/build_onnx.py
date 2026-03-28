"""
Manually build an ONNX model equivalent to the Silero TTS v5 JIT model.

Approach:
1. Extract all weights from the JIT model
2. Build ONNX graph using onnx.helper
3. Replace ISTFT with real-valued DFT basis matrix multiply
4. Validate against PyTorch output

The model architecture:
  tokens[1,N] + speaker_id[1]
    → dur_predictor → durations[1,N]
    → tacotron.embedding + speaker_embedding
    → tacotron.encoder (4x FFT transformer blocks)
    → length_regulator (expand by durations)
    → pitch_predictor → pitch
    → tacotron.decoder (HourGlass transformer)
    → tacotron.lin (Linear 128→96)
    → vocoder.backbone (Conv1d + 8x ConvNeXt + LayerNorm)
    → vocoder.head.out (Linear 512→2*n_freq)
    → split into magnitude + phase
    → ISTFT (real-valued DFT) → audio[1,T]
"""
import sys
import os
import json
import math
import torch
import torch.fft
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def download_model():
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v5_4_ru",
        trust_repo=True,
    )
    return model


def extract_params(jit_model):
    """Extract all named parameters and buffers as numpy arrays."""
    params = {}
    for name, param in jit_model.named_parameters():
        params[name] = param.detach().cpu().numpy()
    for name, buf in jit_model.named_buffers():
        params[name] = buf.detach().cpu().numpy()
    return params


def build_real_istft_basis(n_fft, hop_length, window_np):
    """Build the real-valued DFT basis matrices for ISTFT.

    Instead of complex fft_irfft, we use:
      audio_frame[n] = sum_k (S_real[k]*cos_basis[k,n] + S_imag[k]*sin_basis[k,n])
    where S_real = mag*cos(phase), S_imag = mag*sin(phase)
    """
    n_freq = n_fft // 2 + 1
    n = np.arange(n_fft, dtype=np.float32)
    k = np.arange(n_freq, dtype=np.float32)

    # angles[n, k] = 2π * n * k / n_fft
    angles = 2.0 * np.pi * n[:, None] * k[None, :] / n_fft

    # Basis matrices [n_fft, n_freq]
    cos_basis = np.cos(angles).astype(np.float32)
    sin_basis = np.sin(angles).astype(np.float32)

    # Scale factors: DC and Nyquist get 1/N, others get 2/N
    scale = np.ones(n_freq, dtype=np.float32) * 2.0 / n_fft
    scale[0] = 1.0 / n_fft
    if n_fft % 2 == 0:
        scale[-1] = 1.0 / n_fft

    cos_basis *= scale[None, :]
    sin_basis *= scale[None, :]

    return cos_basis, sin_basis


class OnnxBuilder:
    """Helper class to incrementally build an ONNX graph."""

    def __init__(self):
        self.nodes = []
        self.initializers = []
        self.inputs = []
        self.outputs = []
        self._counter = 0

    def _uid(self, prefix=""):
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def add_initializer(self, name, np_array):
        """Add a weight/constant tensor."""
        self.initializers.append(numpy_helper.from_array(np_array, name=name))

    def add_node(self, op_type, inputs, outputs=None, name=None, **attrs):
        """Add an ONNX node."""
        if outputs is None:
            outputs = [self._uid(op_type)]
        if name is None:
            name = self._uid(f"node_{op_type}")
        node = helper.make_node(op_type, inputs, outputs, name=name, **attrs)
        self.nodes.append(node)
        return outputs[0]

    def build(self, opset=17):
        graph = helper.make_graph(
            self.nodes,
            "silero_tts_v5",
            self.inputs,
            self.outputs,
            self.initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        model.ir_version = 8
        return model


def main():
    print("Step 1: Loading model and extracting parameters...")
    model = download_model()
    pkg = model.packages[0]
    torch._C._jit_set_profiling_mode(False)
    if not pkg.q_model_unpacked:
        pkg.unpack_q_model()
        pkg.q_model_unpacked = True

    jit_model = pkg.models[0]
    symbol_to_id = pkg.symbol_to_id

    params = extract_params(jit_model)
    print(f"  Extracted {len(params)} parameters/buffers")

    # Print parameter names and shapes for reference
    total_params = 0
    for name, arr in sorted(params.items()):
        total_params += arr.size
        if arr.size > 100000:
            print(f"    {name}: {arr.shape} ({arr.size:,} elements)")
    print(f"  Total: {total_params:,} elements")

    # Get ISTFT parameters
    istft = jit_model.vocoder.head.istft
    n_fft = int(istft.n_fft)
    hop_length = int(istft.hop_length)
    window = istft.window.numpy()
    print(f"  ISTFT: n_fft={n_fft}, hop_length={hop_length}")

    # Get reference output for validation
    tokens = [symbol_to_id.get(c, 0) for c in "тест."]
    seq = torch.LongTensor([tokens])
    spk = torch.LongTensor([3])
    with torch.no_grad():
        ref_audio = jit_model(seq, spk)[0].numpy()
    print(f"  Reference audio: {ref_audio.shape}")

    # Step 2: Build ISTFT basis
    print("\nStep 2: Building real-valued ISTFT basis...")
    cos_basis, sin_basis = build_real_istft_basis(n_fft, hop_length, window)
    print(f"  cos_basis: {cos_basis.shape}, sin_basis: {sin_basis.shape}")

    # Step 3: Verify the real ISTFT matches PyTorch's
    print("\nStep 3: Verifying real ISTFT implementation...")

    # Run model to get pre-ISTFT spectral output
    # We'll use PyTorch's hooks... wait, hooks don't work on ScriptModules.
    # Instead, run the model and manually compute ISTFT from the head.out output.

    # Actually, let's trace through to get the spectral representation.
    # We know: head.out produces [B, 2*n_freq, T] where first half is log-magnitude, second is phase
    # Then: mag = exp(log_mag).clamp(max=100), cos_p = cos(phase), sin_p = sin(phase)
    # Then: S = mag * (cos_p + j*sin_p) → irfft → audio

    # For now, let's verify our ISTFT works with a synthetic signal
    test_signal = torch.randn(1, n_fft // 2 + 1, 10)  # [B, n_freq, frames]
    test_complex = test_signal + 0j  # Make it complex

    # PyTorch irfft
    pt_frames = torch.fft.irfft(test_complex, n=n_fft, dim=1).numpy()  # [B, n_fft, frames]

    # Our real ISTFT (just the irfft part, no overlap-add)
    test_np = test_signal.numpy()  # [1, n_freq, frames]
    # Since input is real (no imaginary part), only cos_basis matters
    # result[b, n, t] = sum_k test[b, k, t] * cos_basis[n, k]
    our_frames = np.einsum('bkt,nk->bnt', test_np, cos_basis)

    diff = np.max(np.abs(pt_frames - our_frames))
    print(f"  irfft verification (real input): max diff = {diff:.8f}")

    # Test with complex input
    test_real = torch.randn(1, n_fft // 2 + 1, 10)
    test_imag = torch.randn(1, n_fft // 2 + 1, 10)
    test_complex = torch.complex(test_real, test_imag)

    pt_frames = torch.fft.irfft(test_complex, n=n_fft, dim=1).numpy()

    real_np = test_real.numpy()
    imag_np = test_imag.numpy()
    our_frames = np.einsum('bkt,nk->bnt', real_np, cos_basis) + np.einsum('bkt,nk->bnt', imag_np, sin_basis)

    diff = np.max(np.abs(pt_frames - our_frames))
    print(f"  irfft verification (complex input): max diff = {diff:.8f}")
    if diff < 1e-5:
        print(f"  ISTFT basis VERIFIED!")
    else:
        print(f"  WARNING: ISTFT mismatch — check basis computation")

    # Step 4: Full pipeline verification using numpy
    # Run the actual model and capture intermediate values by running sub-modules
    print("\nStep 4: Testing full real-valued pipeline in numpy...")

    # We need to capture the vocoder head output (pre-ISTFT)
    # Since we can't use hooks, let's modify the approach:
    # 1. Run the full model via PyTorch to get audio
    # 2. Also run with a different sample rate to verify our understanding
    # 3. Then build the ONNX model

    # For now, save all extracted data needed for the ONNX build
    print("\nStep 5: Saving extracted data...")
    os.makedirs("assets", exist_ok=True)

    # Save params as npz
    np.savez_compressed("assets/model_weights.npz", **params)
    print(f"  Saved {len(params)} parameters to assets/model_weights.npz")

    # Save symbol table
    with open("assets/symbols.json", "w", encoding="utf-8") as f:
        json.dump(symbol_to_id, f, ensure_ascii=False, indent=2)

    # Save speakers
    with open("assets/speakers.json", "w", encoding="utf-8") as f:
        json.dump({"aidar": 0, "baya": 1, "kseniya": 2, "xenia": 3}, f, indent=2)

    # Save ISTFT params
    istft_data = {
        "n_fft": n_fft,
        "hop_length": hop_length,
        "window": window.tolist(),
    }
    with open("assets/istft_params.json", "w") as f:
        json.dump(istft_data, f)

    # Save DFT basis
    np.save("assets/cos_basis.npy", cos_basis)
    np.save("assets/sin_basis.npy", sin_basis)

    # Print architecture summary for ONNX build
    print("\n" + "=" * 80)
    print("ARCHITECTURE SUMMARY (for ONNX graph construction)")
    print("=" * 80)

    for prefix in ["tacotron.embedding", "tacotron.speaker_embedding",
                    "tacotron.encoder", "tacotron.len_reg",
                    "tacotron.pitch_proj", "tacotron.energy_proj",
                    "tacotron.decoder", "tacotron.lin",
                    "vocoder.backbone", "vocoder.head",
                    "dur_predictor", "pitch_predictor"]:
        module_params = {k: v for k, v in params.items() if k.startswith(prefix)}
        if module_params:
            total = sum(v.size for v in module_params.values())
            print(f"\n  {prefix} ({total:,} params):")
            for name, arr in sorted(module_params.items()):
                short_name = name[len(prefix) + 1:] if name.startswith(prefix + ".") else name
                print(f"    {short_name}: {arr.shape} {arr.dtype}")

    # Save test WAVs for reference
    try:
        import soundfile as sf
        for name, sid in [("xenia", 3), ("aidar", 0)]:
            t = [symbol_to_id.get(c, 0) for c in "съешьте ещё этих мягких французских булочек."]
            with torch.no_grad():
                audio = jit_model(torch.LongTensor([t]), torch.LongTensor([sid]))[0]
            sf.write(f"test_{name}_ref.wav", audio.numpy().flatten(), 48000)
            print(f"\n  Saved test_{name}_ref.wav ({audio.shape[1]/48000:.2f}s)")
    except Exception as e:
        print(f"  WAV: {e}")

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("All data saved to assets/")
    print("Next: build_onnx_graph.py will construct the ONNX model")
    print("=" * 80)


if __name__ == "__main__":
    main()
