"""
Investigate the Silero TTS v5 model internals.

This script loads the v5_4_ru.pt model and inspects its architecture,
symbol table, speaker map, and the neural network's input/output signatures.

Usage:
    pip install -r requirements.txt
    python investigate_model.py [path_to_v5_4_ru.pt]

If no path is given, the model is downloaded automatically via torch.hub.
"""

import sys
import os
import json
import torch
import numpy as np

# Fix Windows console encoding for Cyrillic output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def download_model():
    """Download the Silero TTS v5 Russian model via torch.hub."""
    print("Downloading Silero TTS v5 model via torch.hub...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v5_4_ru",
        trust_repo=True,
    )
    return model


def load_from_file(path: str):
    """Load model from a local .pt file."""
    print(f"Loading model from {path}...")
    imp = torch.package.PackageImporter(path)
    model = imp.load_pickle("tts_models", "model")
    return model


def inspect_model(model):
    """Inspect the model object: attributes, methods, and structure."""
    print("\n" + "=" * 80)
    print("MODEL INSPECTION")
    print("=" * 80)

    # List all public attributes
    print("\n--- Public attributes ---")
    for attr in sorted(dir(model)):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(model, attr)
            if callable(val):
                print(f"  {attr}() — callable")
            else:
                val_repr = repr(val)
                if len(val_repr) > 200:
                    val_repr = val_repr[:200] + "..."
                print(f"  {attr} = {val_repr}")
        except Exception as e:
            print(f"  {attr} — error accessing: {e}")

    # Speakers
    print("\n--- Speakers ---")
    if hasattr(model, "speakers"):
        print(f"  {model.speakers}")
    else:
        print("  (no 'speakers' attribute)")

    # Symbols / character set
    print("\n--- Symbol table ---")
    for attr_name in ["symbols", "vocab", "char_to_id", "symbol_to_id", "characters"]:
        if hasattr(model, attr_name):
            val = getattr(model, attr_name)
            print(f"  {attr_name} = {repr(val)}")

    # Try to find the inner nn.Module
    print("\n--- Inner nn.Module search ---")
    for attr_name in ["model", "_model", "tts_model", "net", "synthesizer", "decoder"]:
        if hasattr(model, attr_name):
            inner = getattr(model, attr_name)
            print(f"  Found: model.{attr_name} — type: {type(inner).__name__}")
            if isinstance(inner, torch.nn.Module):
                print(f"  It IS an nn.Module!")
                print(f"  Children: {[name for name, _ in inner.named_children()]}")

    # Full module tree (if model itself is nn.Module)
    if isinstance(model, torch.nn.Module):
        print("\n--- Model is nn.Module, printing architecture ---")
        print(model)
    else:
        print(f"\n--- Model type: {type(model).__name__} (not nn.Module directly) ---")
        # Try to find the nn.Module inside
        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                val = getattr(model, attr_name)
                if isinstance(val, torch.nn.Module):
                    print(f"\n  Found nn.Module at model.{attr_name}:")
                    # Print just top-level children to avoid huge output
                    children = list(val.named_children())
                    print(f"  Top-level children ({len(children)}):")
                    for name, child in children:
                        print(f"    {name}: {type(child).__name__}")
            except Exception:
                pass


def trace_apply_tts(model):
    """Trace through apply_tts to understand the pipeline stages."""
    print("\n" + "=" * 80)
    print("TRACING apply_tts()")
    print("=" * 80)

    test_texts = [
        "Привет, мир!",
        "Съешьте ещё этих мягких французских булочек.",
    ]

    speakers = model.speakers if hasattr(model, "speakers") else ["xenia"]

    for text in test_texts:
        print(f"\n--- Text: '{text}' ---")
        for speaker in speakers[:1]:  # Just test first speaker
            for sr in [8000, 24000, 48000]:
                try:
                    audio = model.apply_tts(
                        text=text,
                        speaker=speaker,
                        sample_rate=sr,
                    )
                    if isinstance(audio, torch.Tensor):
                        print(
                            f"  speaker={speaker}, sr={sr}: "
                            f"tensor shape={audio.shape}, dtype={audio.dtype}, "
                            f"min={audio.min().item():.4f}, max={audio.max().item():.4f}, "
                            f"duration={audio.shape[-1] / sr:.3f}s"
                        )
                    else:
                        print(f"  speaker={speaker}, sr={sr}: type={type(audio)}")
                except Exception as e:
                    print(f"  speaker={speaker}, sr={sr}: ERROR — {e}")

    # Test all speakers at one sample rate
    print(f"\n--- All speakers at 48000 Hz ---")
    for speaker in speakers:
        try:
            audio = model.apply_tts(
                text="Тест.", speaker=speaker, sample_rate=48000
            )
            print(
                f"  {speaker}: shape={audio.shape}, duration={audio.shape[-1] / 48000:.3f}s"
            )
        except Exception as e:
            print(f"  {speaker}: ERROR — {e}")


def inspect_forward_pass(model):
    """Try to understand the forward pass inputs/outputs for ONNX export."""
    print("\n" + "=" * 80)
    print("FORWARD PASS INSPECTION (for ONNX export)")
    print("=" * 80)

    # Try to find how text is converted to tokens
    print("\n--- Looking for tokenization internals ---")

    # Hook into the model to intercept the forward call
    inner_model = None
    for attr_name in ["model", "_model", "tts_model", "net"]:
        if hasattr(model, attr_name):
            candidate = getattr(model, attr_name)
            if isinstance(candidate, torch.nn.Module):
                inner_model = candidate
                print(f"  Found inner model at model.{attr_name}")
                break

    if inner_model is None:
        # Model itself might be nn.Module
        if isinstance(model, torch.nn.Module):
            inner_model = model
            print("  Using model itself as nn.Module")

    if inner_model is not None:
        # List all parameters to understand dimensions
        print("\n--- Parameter shapes ---")
        total_params = 0
        for name, param in inner_model.named_parameters():
            total_params += param.numel()
            if any(
                kw in name.lower()
                for kw in ["embed", "speaker", "spk", "style", "enc.weight"]
            ):
                print(f"  {name}: {param.shape} ({param.dtype})")
        print(f"\n  Total parameters: {total_params:,}")

        # Try to hook the forward method
        print("\n--- Attempting to hook forward pass ---")
        original_forward = inner_model.forward
        captured_inputs = []

        def hook_forward(*args, **kwargs):
            captured_inputs.append({"args": args, "kwargs": kwargs})
            return original_forward(*args, **kwargs)

        inner_model.forward = hook_forward

        try:
            audio = model.apply_tts(
                text="Тест.", speaker="xenia", sample_rate=48000
            )
            if captured_inputs:
                print(f"  Forward was called {len(captured_inputs)} time(s)")
                for i, call in enumerate(captured_inputs):
                    print(f"\n  Call {i}:")
                    for j, arg in enumerate(call["args"]):
                        if isinstance(arg, torch.Tensor):
                            print(
                                f"    arg[{j}]: Tensor shape={arg.shape}, "
                                f"dtype={arg.dtype}, "
                                f"values={arg.flatten()[:20].tolist()}"
                            )
                        else:
                            print(f"    arg[{j}]: {type(arg).__name__} = {repr(arg)[:200]}")
                    for k, v in call["kwargs"].items():
                        if isinstance(v, torch.Tensor):
                            print(
                                f"    {k}: Tensor shape={v.shape}, "
                                f"dtype={v.dtype}"
                            )
                        else:
                            print(f"    {k}: {type(v).__name__} = {repr(v)[:200]}")
            else:
                print("  Forward was NOT called (model may use a different method)")
        except Exception as e:
            print(f"  Error during hooked forward: {e}")
        finally:
            inner_model.forward = original_forward

    # Try to access the text-to-sequence function
    print("\n--- Looking for text_to_sequence / text_to_tokens ---")
    for attr_name in dir(model):
        if any(
            kw in attr_name.lower()
            for kw in ["text_to", "encode", "tokeniz", "phonem", "sequence"]
        ):
            print(f"  model.{attr_name}")

    # Try to find source code of apply_tts
    print("\n--- apply_tts source (if available) ---")
    try:
        import inspect
        source = inspect.getsource(model.apply_tts)
        print(source[:3000])
        if len(source) > 3000:
            print(f"  ... ({len(source)} chars total)")
    except Exception as e:
        print(f"  Cannot get source: {e}")


def export_findings(model, output_path="model_findings.json"):
    """Export discovered model metadata to JSON for use by export scripts."""
    findings = {
        "model_type": type(model).__name__,
        "is_nn_module": isinstance(model, torch.nn.Module),
        "speakers": model.speakers if hasattr(model, "speakers") else [],
    }

    # Symbol table
    for attr_name in ["symbols", "vocab", "char_to_id", "symbol_to_id"]:
        if hasattr(model, attr_name):
            val = getattr(model, attr_name)
            if isinstance(val, str):
                findings["symbols"] = val
                findings["symbol_to_id"] = {ch: i for i, ch in enumerate(val)}
            elif isinstance(val, dict):
                findings["symbol_to_id"] = {str(k): int(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                findings["symbols"] = list(val)
                findings["symbol_to_id"] = {ch: i for i, ch in enumerate(val)}

    # Inner model info
    inner = None
    for attr_name in ["model", "_model", "tts_model", "net"]:
        if hasattr(model, attr_name):
            candidate = getattr(model, attr_name)
            if isinstance(candidate, torch.nn.Module):
                inner = candidate
                findings["inner_model_attr"] = attr_name
                findings["inner_model_type"] = type(candidate).__name__
                findings["inner_model_children"] = [
                    name for name, _ in candidate.named_children()
                ]
                findings["total_parameters"] = sum(
                    p.numel() for p in candidate.parameters()
                )
                break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)
    print(f"\nFindings exported to {output_path}")


def main():
    if len(sys.argv) > 1:
        model = load_from_file(sys.argv[1])
    else:
        model = download_model()

    if hasattr(model, 'eval'):
        model.eval()
    torch.set_num_threads(4)

    inspect_model(model)
    trace_apply_tts(model)
    inspect_forward_pass(model)
    export_findings(model)

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)
    print("Next steps:")
    print("  1. Review model_findings.json")
    print("  2. Run export_onnx.py to attempt ONNX conversion")
    print("  3. Run export_assets.py to extract symbol table and stress dict")


if __name__ == "__main__":
    main()
