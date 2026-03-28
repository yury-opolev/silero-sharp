"""
Deep investigation into the Silero TTS v5 model internals.
Focused on finding the nn.Module inside the package for ONNX export.
"""
import sys
import os
import json
import torch
import numpy as np

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


def main():
    model = download_model()

    print("=" * 80)
    print("DEEP INVESTIGATION: packages[0] (PartTTSModelMultiAcc_v3)")
    print("=" * 80)

    pkg = model.packages[0]
    print(f"\nType: {type(pkg).__name__}")

    # All attributes
    print("\n--- All attributes of packages[0] ---")
    for attr in sorted(dir(pkg)):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(pkg, attr)
            if isinstance(val, torch.nn.Module):
                params = sum(p.numel() for p in val.parameters())
                print(f"  {attr}: nn.Module ({type(val).__name__}, {params:,} params)")
            elif callable(val):
                print(f"  {attr}(): callable")
            elif isinstance(val, torch.Tensor):
                print(f"  {attr}: Tensor {val.shape} {val.dtype}")
            elif isinstance(val, str) and len(val) > 100:
                print(f"  {attr}: str ({len(val)} chars)")
            elif isinstance(val, (dict, list)) and len(val) > 20:
                print(f"  {attr}: {type(val).__name__} ({len(val)} items)")
            else:
                val_str = repr(val)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"  {attr} = {val_str}")
        except Exception as e:
            print(f"  {attr}: ERROR - {e}")

    # Find all nn.Module instances recursively
    print("\n--- nn.Module search (recursive) ---")
    found_modules = []
    for attr in dir(pkg):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(pkg, attr)
            if isinstance(val, torch.nn.Module):
                found_modules.append((attr, val))
                total = sum(p.numel() for p in val.parameters())
                children = list(val.named_children())
                print(f"\n  model.packages[0].{attr}: {type(val).__name__}")
                print(f"    Total params: {total:,}")
                print(f"    Children ({len(children)}):")
                for name, child in children:
                    child_params = sum(p.numel() for p in child.parameters())
                    print(f"      {name}: {type(child).__name__} ({child_params:,} params)")
        except Exception:
            pass

    # Look for the main TTS model
    print("\n--- Looking for main TTS model ---")
    for attr_name in ["model", "tts_model", "net", "acoustic_model", "synthesizer",
                       "decoder", "vocoder", "generator", "encoder"]:
        if hasattr(pkg, attr_name):
            val = getattr(pkg, attr_name)
            print(f"  Found: packages[0].{attr_name} = {type(val).__name__}")
            if isinstance(val, torch.nn.Module):
                print(f"    It IS nn.Module with {sum(p.numel() for p in val.parameters()):,} params")

    # Try to trace the apply_tts call on the package
    print("\n--- Tracing packages[0].apply_tts ---")
    try:
        import inspect
        source = inspect.getsource(pkg.apply_tts)
        print(source[:5000])
    except Exception as e:
        print(f"  Cannot get source: {e}")

    # Hook into any nn.Module forward calls
    print("\n--- Hooking all nn.Module forward calls ---")
    hook_log = []

    def make_hook(name):
        def hook(module, input, output):
            input_info = []
            for inp in input:
                if isinstance(inp, torch.Tensor):
                    input_info.append(f"Tensor{list(inp.shape)} {inp.dtype}")
                else:
                    input_info.append(f"{type(inp).__name__}")
            output_info = ""
            if isinstance(output, torch.Tensor):
                output_info = f"Tensor{list(output.shape)} {output.dtype}"
            elif isinstance(output, tuple):
                parts = []
                for o in output:
                    if isinstance(o, torch.Tensor):
                        parts.append(f"Tensor{list(o.shape)}")
                    elif o is not None:
                        parts.append(type(o).__name__)
                output_info = f"({', '.join(parts)})"
            hook_log.append(f"  {name}: ({', '.join(input_info)}) -> {output_info}")
        return hook

    handles = []
    for attr in dir(pkg):
        try:
            val = getattr(pkg, attr)
            if isinstance(val, torch.nn.Module):
                # Register hooks on all children
                for child_name, child in val.named_modules():
                    full_name = f"{attr}.{child_name}" if child_name else attr
                    h = child.register_forward_hook(make_hook(full_name))
                    handles.append(h)
        except Exception:
            pass

    try:
        with torch.no_grad():
            audio = pkg.apply_tts(text="Тест.", speaker="xenia", sample_rate=48000)
        print(f"  Output: {audio.shape}, {audio.dtype}")
        print(f"\n  Forward calls ({len(hook_log)}):")
        for entry in hook_log:
            print(entry)
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        for h in handles:
            h.remove()

    # Export the findings
    print("\n--- Exporting deep findings ---")
    findings = {
        "package_type": type(pkg).__name__,
        "symbols": getattr(model, "symbols", ""),
        "speakers": model.speakers,
        "speaker_to_package": model.speaker_to_package,
        "nn_modules_found": [],
    }
    for attr, mod in found_modules:
        findings["nn_modules_found"].append({
            "attr": attr,
            "type": type(mod).__name__,
            "params": sum(p.numel() for p in mod.parameters()),
            "children": [name for name, _ in mod.named_children()],
        })

    with open("deep_findings.json", "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    print("Saved to deep_findings.json")


if __name__ == "__main__":
    main()
