# SileroSharp

C# streaming TTS library for the [Silero TTS v5](https://github.com/snakers4/silero-models) Russian speech synthesis model. No Python dependency at runtime — runs entirely in .NET via [TorchSharp](https://github.com/dotnet/TorchSharp).

## Features

- **Pure C#** — no Python, no gRPC, no subprocess. The Silero TorchScript model runs directly via TorchSharp/libtorch.
- **Bit-exact output** — produces identical audio to the original Python `apply_tts()` pipeline.
- **Streaming** — sentence-by-sentence synthesis with `IAsyncEnumerable<AudioChunk>`. Audio playback starts before the full text is synthesized.
- **Full text processing pipeline** — ported from Silero's Python code:
  - **HomoSolver** — BERT-based homograph disambiguation (1,924 context-dependent words like "дела", "замок", "село")
  - **StressAccentor** — Neural FastText n-gram stress placement for correct Russian prosody
  - **Sentence splitting** — Russian abbreviation-aware chunking for long texts
- **4 speakers** — aidar (male), baya, kseniya, xenia (female)
- **48 kHz** output, float32 PCM

## Quick Start

```csharp
using SileroSharp;
using SileroSharp.NAudio;

// Load model with auto-discovery of accentor assets
await using var tts = SileroLoader.LoadWithAutoDiscovery("silero_v5_jit.pt");

// Single-shot synthesis
var chunk = await tts.SynthesizeAsync("Привет, мир!", SileroVoice.Xenia);

// Streaming to audio device
await using var sink = new NAudioWaveOutSink();
await tts.SpeakToSinkAsync("Длинный текст для синтеза речи.", SileroVoice.Xenia, sink);

// Raw streaming (IAsyncEnumerable)
await foreach (var audioChunk in tts.SynthesizeStreamingAsync("Текст.", SileroVoice.Aidar))
{
    // Process each sentence chunk as it's synthesized
}
```

## Project Structure

```
src/
  SileroSharp/             Core library (TorchSharp, text processing)
  SileroSharp.NAudio/      Windows audio playback sink
tests/
  SileroSharp.Tests/       Unit tests
samples/
  SileroSharp.Sample/      Console demo app
accentor/                  Neural accentor model weights and dictionaries
tools/                     Python scripts for model investigation and asset extraction
```

## Prerequisites

### Model file

The TorchScript model file (`silero_v5_jit.pt`, ~89 MB) is not included in the repo. Generate it with:

```bash
cd tools
uv venv .venv --python 3.12
uv pip install -r requirements.txt
python build_onnx.py   # Downloads Silero v5 and saves silero_v5_jit.pt
```

Or extract it manually:

```python
import torch
model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts',
                           language='ru', speaker='v5_4_ru', trust_repo=True)
pkg = model.packages[0]
torch._C._jit_set_profiling_mode(False)
pkg.unpack_q_model()
torch.jit.save(pkg.models[0], 'silero_v5_jit.pt')
```

### Homosolver BERT model

For correct homograph resolution, also extract the BERT model (`homosolver_bert.pt`, ~37 MB) into the `accentor/` directory:

```python
torch.jit.save(pkg.accentor.homosolver.model, 'accentor/homosolver_bert.pt')
```

### NuGet packages

The core library depends on `TorchSharp`. The sample app also needs a libtorch runtime:

```xml
<!-- Core library reference -->
<PackageReference Include="TorchSharp" Version="0.106.0" />

<!-- CPU runtime (pick one per platform) -->
<PackageReference Include="TorchSharp-cpu" Version="0.106.0" />
```

## File Layout at Runtime

The loader expects this layout next to the model file:

```
silero_v5_jit.pt              TorchScript TTS model
accentor/
  homosolver_bert.pt          BERT homograph resolver model
  homodict.json               Homograph dictionary (1,924 entries)
  homo_vocab.json             BERT tokenizer vocabulary
  homo_special.json           BERT special token IDs
  exceptions.json             Stress exceptions dictionary (16,969 entries)
  ngram_dict.json             FastText n-gram vocabulary (126,523 entries)
  stress_embedding.npy        N-gram embedding weights
  stress_clf_*.npy            Stress classifier MLP weights
  yo_clf_*.npy                Yo classifier MLP weights
```

## Architecture

```
Input text
  │
  ├─ HomoSolver ──── BERT model (TorchSharp JIT) resolves homographs
  │                   "как ваши дела?" → "как ваши дел+а?"
  │
  ├─ StressAccentor ─ Neural FastText MLP (pure C# math) adds stress marks
  │                   "д+обрый д+ень. к+ак в+аши дел+а?"
  │
  ├─ SymbolTable ──── Character tokenization with SOS/EOS framing
  │                   [|, д, +, о, б, р, ы, й, ...]
  │
  ├─ SileroModel ──── TorchSharp JIT inference → float32 PCM audio
  │
  └─ IAudioSink ───── NAudio WaveOut playback (or custom consumer)
```

## Streaming Pipeline

For long texts, sentences are synthesized one at a time with a bounded channel (capacity 3) that overlaps inference with playback:

```
Inference:  [sentence 1]  [sentence 2]  [sentence 3]
Playback:         [play 1]      [play 2]      [play 3]
```

First audio starts playing after the first sentence is synthesized.

## License

This library is MIT licensed. The Silero TTS v5 Russian model (`v5_4_ru`) is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only).
