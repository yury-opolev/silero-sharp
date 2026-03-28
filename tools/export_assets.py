"""
Extract assets from the Silero TTS v5 model for use in the C# library.

Extracts:
  - symbols.json: Character-to-token-ID mapping
  - speakers.json: Speaker name to ID mapping
  - stress_dict.json: Stress dictionary (word → stressed form)

Usage:
    python export_assets.py [path_to_v5_4_ru.pt] [--output-dir ../src/SileroSharp/Resources]
"""

import sys
import json
import struct
import argparse
from pathlib import Path

import torch


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


def extract_symbols(model, output_dir: Path):
    """Extract the symbol table (character → token ID mapping)."""
    print("\n--- Extracting symbol table ---")

    symbols = None
    for attr_name in ["symbols", "vocab", "char_to_id", "symbol_to_id", "characters"]:
        if hasattr(model, attr_name):
            val = getattr(model, attr_name)
            if isinstance(val, str):
                # String of characters where index = token ID
                symbols = {ch: i for i, ch in enumerate(val)}
                print(f"  Found symbols as string ({len(val)} chars) at model.{attr_name}")
                break
            elif isinstance(val, dict):
                symbols = {str(k): int(v) for k, v in val.items()}
                print(f"  Found symbols as dict ({len(val)} entries) at model.{attr_name}")
                break
            elif isinstance(val, (list, tuple)):
                symbols = {ch: i for i, ch in enumerate(val)}
                print(f"  Found symbols as list ({len(val)} items) at model.{attr_name}")
                break

    if symbols is None:
        # Try to discover symbols by probing the model
        print("  No direct symbol table found, attempting discovery...")
        # Look deeper in the model attributes
        for attr in dir(model):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(model, attr)
                if isinstance(val, str) and len(val) > 20 and any(
                    c in val for c in "абвгдеж"
                ):
                    symbols = {ch: i for i, ch in enumerate(val)}
                    print(f"  Discovered symbols at model.{attr} ({len(val)} chars)")
                    break
            except Exception:
                pass

    if symbols is None:
        print("  WARNING: Could not find symbol table!")
        print("  You may need to manually inspect the model with investigate_model.py")
        return

    output_path = output_dir / "symbols.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(symbols, f, ensure_ascii=False, indent=2, sort_keys=False)
    print(f"  Saved {len(symbols)} symbols to {output_path}")

    # Also save the raw symbol string for reference
    if any(isinstance(getattr(model, a, None), str) for a in ["symbols"]):
        raw_str = getattr(model, "symbols")
        print(f"  Raw symbol string: {repr(raw_str[:100])}...")


def extract_speakers(model, output_dir: Path):
    """Extract the speaker name → ID mapping."""
    print("\n--- Extracting speaker map ---")

    speakers = {}
    if hasattr(model, "speakers"):
        speaker_list = model.speakers
        if isinstance(speaker_list, (list, tuple)):
            speakers = {name: i for i, name in enumerate(speaker_list)}
            print(f"  Found {len(speakers)} speakers: {list(speakers.keys())}")
        elif isinstance(speaker_list, dict):
            speakers = {str(k): int(v) for k, v in speaker_list.items()}
            print(f"  Found {len(speakers)} speakers: {list(speakers.keys())}")
    else:
        print("  No 'speakers' attribute found")
        # Try known speakers
        known_speakers = ["aidar", "baya", "kseniya", "xenia"]
        for i, name in enumerate(known_speakers):
            try:
                model.apply_tts(text="Тест.", speaker=name, sample_rate=8000)
                speakers[name] = i
                print(f"  Verified speaker: {name}")
            except Exception:
                pass

    if speakers:
        output_path = output_dir / "speakers.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(speakers, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {output_path}")
    else:
        print("  WARNING: No speakers found!")


def extract_stress_dict(model, output_dir: Path):
    """Extract the stress dictionary if accessible."""
    print("\n--- Extracting stress dictionary ---")

    stress_dict = None

    # The stress/accentor model may store a dictionary internally
    for attr_name in dir(model):
        if any(kw in attr_name.lower() for kw in ["stress", "accent", "dict", "vocab"]):
            try:
                val = getattr(model, attr_name)
                if isinstance(val, dict) and len(val) > 100:
                    # Check if it looks like a stress dictionary
                    sample_key = next(iter(val))
                    if isinstance(sample_key, str) and any(
                        c in sample_key for c in "абвгдежзиклмнопрстуфхцчшщэюя"
                    ):
                        stress_dict = val
                        print(
                            f"  Found stress dict at model.{attr_name} "
                            f"({len(val)} entries)"
                        )
                        break
            except Exception:
                pass

    # Also check nested objects
    if stress_dict is None:
        for attr_name in dir(model):
            if attr_name.startswith("_"):
                continue
            try:
                obj = getattr(model, attr_name)
                if hasattr(obj, "__dict__"):
                    for inner_attr in dir(obj):
                        if any(
                            kw in inner_attr.lower()
                            for kw in ["stress", "accent", "dict"]
                        ):
                            inner_val = getattr(obj, inner_attr)
                            if isinstance(inner_val, dict) and len(inner_val) > 100:
                                stress_dict = inner_val
                                print(
                                    f"  Found stress dict at "
                                    f"model.{attr_name}.{inner_attr} "
                                    f"({len(inner_val)} entries)"
                                )
                                break
                if stress_dict:
                    break
            except Exception:
                pass

    if stress_dict:
        # Save as JSON (may be large)
        output_path = output_dir / "stress_dict.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stress_dict, f, ensure_ascii=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {len(stress_dict)} entries to {output_path} ({size_mb:.1f} MB)")

        # Show sample entries
        print("  Sample entries:")
        for i, (k, v) in enumerate(stress_dict.items()):
            if i >= 10:
                break
            print(f"    {k} → {v}")
    else:
        print("  WARNING: Could not find stress dictionary")
        print("  The stress model may be purely neural (no dictionary)")
        print("  Consider using the nostress model variant or building")
        print("  a dictionary by running the model on a word list")


def extract_yo_dict(model, output_dir: Path):
    """Extract the yo (ё) restoration dictionary if accessible."""
    print("\n--- Extracting yo (ё) dictionary ---")

    yo_dict = None
    for attr_name in dir(model):
        if any(kw in attr_name.lower() for kw in ["yo", "ё", "jo"]):
            try:
                val = getattr(model, attr_name)
                if isinstance(val, dict) and len(val) > 10:
                    yo_dict = val
                    print(
                        f"  Found yo dict at model.{attr_name} ({len(val)} entries)"
                    )
                    break
            except Exception:
                pass

    if yo_dict:
        output_path = output_dir / "yo_dict.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(yo_dict, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {output_path}")
    else:
        print("  No yo dictionary found (may be embedded in the stress model)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract assets from Silero TTS v5 model"
    )
    parser.add_argument("model_path", nargs="?", help="Path to v5_4_ru.pt")
    parser.add_argument(
        "--output-dir",
        default="assets",
        help="Output directory for extracted assets",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        model = load_from_file(args.model_path)
    else:
        model = download_model()

    model.eval()

    extract_symbols(model, output_dir)
    extract_speakers(model, output_dir)
    extract_stress_dict(model, output_dir)
    extract_yo_dict(model, output_dir)

    print("\n" + "=" * 80)
    print("ASSET EXTRACTION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
