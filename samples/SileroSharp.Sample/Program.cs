using SileroSharp;

const string modelPath = "silero_v5_jit.pt";
if (!File.Exists(modelPath)) { Console.WriteLine("Model not found"); return; }

await using var tts = SileroLoader.LoadWithAutoDiscovery(modelPath);

// Generate WAVs and dump debug info via SynthesizeAsync
var text = "Добрый день. Как ваши дела?";
Console.WriteLine($"Input: {text}");

var chunk = await tts.SynthesizeAsync(text, SileroVoice.Xenia);
Console.WriteLine($"Output: {chunk.Samples.Length} samples ({chunk.Duration.TotalSeconds:F2}s)");

// Also generate all comparison WAVs
Directory.CreateDirectory("output");
var testCases = new (string Text, string Speaker, SileroVoice Voice)[]
{
    ("Привет, мир!", "xenia", SileroVoice.Xenia),
    ("Привет, мир!", "aidar", SileroVoice.Aidar),
    ("Съешьте ещё этих мягких французских булочек, да выпейте чаю.", "xenia", SileroVoice.Xenia),
    ("Добрый день. Как ваши дела?", "xenia", SileroVoice.Xenia),
    ("Тест.", "xenia", SileroVoice.Xenia),
};

foreach (var (t, speaker, voice) in testCases)
{
    var c = await tts.SynthesizeAsync(t, voice);
    var safe = new string(t.Take(20).Where(ch => !",!?.".Contains(ch)).Select(ch => ch == ' ' ? '_' : ch).ToArray());
    var wavPath = $"output/{safe}_{speaker}_csharp.wav";
    WriteWav(wavPath, c.Samples, c.SampleRate);
    Console.WriteLine($"  {safe}_{speaker}: {c.Samples.Length} samples -> {wavPath}");
}

static void WriteWav(string path, float[] samples, int sampleRate)
{
    using var stream = File.Create(path);
    using var writer = new BinaryWriter(stream);
    var dataSize = samples.Length * 2;
    writer.Write("RIFF"u8); writer.Write(36 + dataSize); writer.Write("WAVE"u8);
    writer.Write("fmt "u8); writer.Write(16); writer.Write((short)1); writer.Write((short)1);
    writer.Write(sampleRate); writer.Write(sampleRate * 2); writer.Write((short)2); writer.Write((short)16);
    writer.Write("data"u8); writer.Write(dataSize);
    foreach (var s in samples) { writer.Write((short)(Math.Clamp(s, -1f, 1f) * 32767)); }
}
