# SileroSharp

C# streaming TTS library for the [Silero TTS v5](https://github.com/snakers4/silero-models) Russian speech synthesis model. No Python dependency at runtime — runs entirely in .NET via [TorchSharp](https://github.com/dotnet/TorchSharp).

## Model Variants

| | v5_4_ru | v5_cis_base |
|---|---|---|
| **License** | CC BY-NC 4.0 (non-commercial) | **MIT (commercial OK)** |
| **Russian voices** | 4 (aidar, baya, kseniya, xenia) | **29** (aigul, dmitriy, eduard, etc.) |
| **Other languages** | — | Tatar, Bashkir, Kazakh, Ukrainian, + more |
| **Quality** | Better prosody and intonation | Slightly lower, but decent |
| **Built-in accentor** | Yes | No (this library provides one) |

For commercial use, choose `v5_cis_base`. For best quality in non-commercial projects, choose `v5_4_ru`.

## Features

- **Pure C#** — no Python, no gRPC, no subprocess. The Silero TorchScript model runs directly via TorchSharp/libtorch.
- **Bit-exact output** — produces identical audio to the original Python `apply_tts()` pipeline.
- **Streaming** — sentence-by-sentence synthesis with `IAsyncEnumerable<AudioChunk>`. Audio playback starts before the full text is synthesized.
- **Full text processing pipeline** — ported from Silero's Python code:
  - **HomoSolver** — BERT-based homograph disambiguation (1,924 context-dependent words like "дела", "замок", "село")
  - **StressAccentor** — Neural FastText n-gram stress placement for correct Russian prosody
  - **Sentence splitting** — Russian abbreviation-aware chunking for long texts
- **33 speakers** — 4 from v5_4_ru + 29 Russian voices from v5_cis_base
- **48 kHz** output, float32 PCM

## Quick Start

```csharp
using SileroSharp;
using SileroSharp.NAudio;

// --- Option A: v5_4_ru (CC BY-NC, higher quality) ---
await using var tts = SileroLoader.LoadWithAutoDiscovery("silero_v5_jit.pt");
var chunk = await tts.SynthesizeAsync("Привет, мир!", SileroVoice.Xenia);

// --- Option B: v5_cis_base (MIT, commercial OK) ---
var options = new SileroOptions { Variant = SileroModelVariant.V5CisBase };
await using var ttsCis = SileroLoader.LoadWithAutoDiscovery("silero_v5_cis_jit.pt", options);
var chunk2 = await ttsCis.SynthesizeAsync("Привет, мир!", SileroVoice.RuDmitriy);

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

### Model files

The TorchScript model files are not included in the repo. Extract them with Python:

```python
import torch

# v5_4_ru (CC BY-NC 4.0)
model, _ = torch.hub.load('snakers4/silero-models', 'silero_tts',
                           language='ru', speaker='v5_4_ru', trust_repo=True)
pkg = model.packages[0]
torch._C._jit_set_profiling_mode(False)
pkg.unpack_q_model()
torch.jit.save(pkg.models[0], 'silero_v5_jit.pt')

# Homosolver BERT model (for correct homograph stress)
torch.jit.save(pkg.accentor.homosolver.model, 'accentor/homosolver_bert.pt')

# v5_cis_base (MIT)
model_cis, _ = torch.hub.load('snakers4/silero-models', 'silero_tts',
                               language='ru', speaker='v5_cis_base', trust_repo=True)
pkg_cis = model_cis.packages[0]
pkg_cis.unpack_q_model()
torch.jit.save(pkg_cis.models[0], 'silero_v5_cis_jit.pt')
```

### NuGet packages

```xml
<!-- Core library reference -->
<PackageReference Include="TorchSharp" Version="0.106.0" />

<!-- CPU runtime (pick one per platform) -->
<PackageReference Include="TorchSharp-cpu" Version="0.106.0" />
```

## File Layout at Runtime

```
silero_v5_jit.pt              TorchScript TTS model (v5_4_ru or v5_cis_base)
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
  |
  +- HomoSolver ---- BERT model (TorchSharp JIT) resolves homographs
  |                   "как ваши дела?" -> "как ваши дел+а?"
  |
  +- StressAccentor - Neural FastText MLP (pure C# math) adds stress marks
  |                   "д+обрый д+ень. к+ак в+аши дел+а?"
  |
  +- SymbolTable ---- Character tokenization with SOS/EOS framing
  |                   [|, д, +, о, б, р, ы, й, ...]
  |
  +- SileroModel ---- TorchSharp JIT inference -> float32 PCM audio
  |
  +- IAudioSink ----- NAudio WaveOut playback (or custom consumer)
```

## Streaming Pipeline

For long texts, sentences are synthesized one at a time with a bounded channel (capacity 3) that overlaps inference with playback:

```
Inference:  [sentence 1]  [sentence 2]  [sentence 3]
Playback:         [play 1]      [play 2]      [play 3]
```

First audio starts playing after the first sentence is synthesized.

## License

This library (SileroSharp) is MIT licensed.

The Silero TTS models have separate licenses:
- **v5_cis_base** — MIT (commercial use OK)
- **v5_4_ru** — CC BY-NC 4.0 (non-commercial only)
