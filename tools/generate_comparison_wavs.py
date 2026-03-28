"""
Generate reference WAV files from the original Silero model for comparison.
Also generate the correct token sequences so we can verify the C# TorchSharp
output produces identical results.
"""
import sys, os, json, torch, numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def download_model():
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models", model="silero_tts",
        language="ru", speaker="v5_4_ru", trust_repo=True)
    return model

def main():
    import soundfile as sf

    model = download_model()
    pkg = model.packages[0]
    torch._C._jit_set_profiling_mode(False)
    if not pkg.q_model_unpacked:
        pkg.unpack_q_model()
        pkg.q_model_unpacked = True
    jit_model = pkg.models[0]
    symbol_to_id = pkg.symbol_to_id

    # Also save the JIT model for C# to use
    jit_path = "silero_v5_jit.pt"
    if not os.path.exists(jit_path):
        torch.jit.save(jit_model, jit_path)
        print(f"Saved JIT model: {jit_path} ({os.path.getsize(jit_path)/1024/1024:.1f} MB)")

    os.makedirs("comparison", exist_ok=True)

    test_cases = [
        ("Привет, мир!", "xenia", 3),
        ("Привет, мир!", "aidar", 0),
        ("Съешьте ещё этих мягких французских булочек, да выпейте чаю.", "xenia", 3),
        ("Добрый день. Как ваши дела?", "xenia", 3),
        ("Тест.", "xenia", 3),
    ]

    results = []

    for text, spk_name, spk_id in test_cases:
        safe = text[:20].replace(" ", "_").replace(",", "").replace(".", "").replace("!", "").replace("?", "")

        # Get correct tokens via original preprocessing
        sentences, clean, breaks, rates, pitches, sp_ids = \
            pkg.prepare_tts_model_input(text, ssml=False, speaker_ids=[spk_id], lang=None)
        with torch.no_grad():
            sequence, _, _, _ = pkg.merge_batch_model(
                sentences, breaks, rates, pitches,
                put_accent=True, put_stress_homo=True,
                put_yo=True, put_yo_homo=True, stress_single_vowel=True)

        tokens = sequence.flatten().tolist()

        # Generate audio via original apply_tts (full pipeline)
        with torch.no_grad():
            audio_full = pkg.apply_tts(text=text, speaker=spk_name, sample_rate=48000,
                                       put_accent=True, put_yo=True).numpy()

        # Generate audio via raw JIT model with correct tokens
        with torch.no_grad():
            audio_jit = jit_model(sequence, torch.LongTensor([spk_id]))[0].numpy().flatten()

        # Verify they match
        min_len = min(len(audio_full), len(audio_jit))
        diff = np.max(np.abs(audio_full[:min_len] - audio_jit[:min_len]))

        wav_file = f"comparison/{safe}_{spk_name}_pytorch.wav"
        sf.write(wav_file, audio_full, 48000)

        result = {
            "text": text,
            "speaker": spk_name,
            "speaker_id": spk_id,
            "tokens": tokens,
            "audio_length": len(audio_full),
            "wav_file": wav_file,
            "full_vs_jit_diff": float(diff),
        }
        results.append(result)

        print(f"'{text}' ({spk_name}): {len(tokens)} tokens, {len(audio_full)} samples "
              f"({len(audio_full)/48000:.2f}s), full_vs_jit_diff={diff:.8f}")

    # Save test manifest for C# validation
    with open("comparison/test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} test cases to comparison/")
    print("The C# sample app should produce identical WAVs when using silero_v5_jit.pt")
    print("with the same token sequences listed in test_manifest.json")

if __name__ == "__main__":
    main()
