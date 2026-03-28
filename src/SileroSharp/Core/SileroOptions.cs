namespace SileroSharp;

/// <summary>
/// Configuration options for Silero TTS synthesis.
/// </summary>
public sealed record SileroOptions
{
    /// <summary>Which Silero model variant to use (affects symbol table and speaker IDs).</summary>
    public SileroModelVariant Variant { get; init; } = SileroModelVariant.V5Russian;

    /// <summary>Audio sample rate in Hz. Silero v5 supports 8000, 24000, 48000.</summary>
    public int SampleRate { get; init; } = 48000;

    /// <summary>Enable automatic stress placement on Russian text.</summary>
    public bool PutAccent { get; init; } = true;

    /// <summary>Restore the Russian letter ё where appropriate.</summary>
    public bool PutYo { get; init; } = true;

    /// <summary>Maximum characters per synthesis chunk. Silero has a 140-char soft limit; 130 leaves a safety margin.</summary>
    public int MaxChunkLength { get; init; } = 130;
}
