namespace SileroSharp;

/// <summary>
/// A chunk of synthesized PCM audio from the Silero TTS engine.
/// Samples are float32 in the range [-1.0, 1.0].
/// </summary>
public sealed class AudioChunk
{
    /// <summary>PCM float32 audio samples.</summary>
    public required float[] Samples { get; init; }

    /// <summary>Sample rate in Hz.</summary>
    public required int SampleRate { get; init; }

    /// <summary>Zero-based index of the sentence this chunk belongs to.</summary>
    public int SentenceIndex { get; init; }

    /// <summary>The source text that produced this chunk.</summary>
    public string SourceText { get; init; } = string.Empty;

    /// <summary>Whether this is the final chunk in the stream.</summary>
    public bool IsLast { get; init; }

    /// <summary>Estimated duration of this audio chunk.</summary>
    public TimeSpan Duration =>
        SampleRate > 0
            ? TimeSpan.FromSeconds((double)Samples.Length / SampleRate)
            : TimeSpan.Zero;

    /// <summary>
    /// Convert float32 samples to 16-bit PCM bytes (little-endian).
    /// </summary>
    public byte[] ToPcm16Bytes()
    {
        var bytes = new byte[Samples.Length * 2];
        for (var i = 0; i < Samples.Length; i++)
        {
            var clamped = Math.Clamp(Samples[i], -1.0f, 1.0f);
            var sample16 = (short)(clamped * 32767);
            bytes[i * 2] = (byte)(sample16 & 0xFF);
            bytes[i * 2 + 1] = (byte)((sample16 >> 8) & 0xFF);
        }

        return bytes;
    }

    /// <summary>
    /// Convert float32 samples to a raw IEEE float32 byte array (for NAudio BufferedWaveProvider).
    /// </summary>
    public byte[] ToFloat32Bytes()
    {
        var bytes = new byte[Samples.Length * sizeof(float)];
        Buffer.BlockCopy(Samples, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
