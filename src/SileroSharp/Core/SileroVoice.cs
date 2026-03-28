namespace SileroSharp;

/// <summary>
/// Represents a Silero TTS speaker voice.
/// Speaker IDs will be confirmed after model investigation (Phase 0).
/// </summary>
public sealed class SileroVoice
{
    public string Name { get; }
    public int SpeakerId { get; }

    private SileroVoice(string name, int speakerId)
    {
        Name = name;
        SpeakerId = speakerId;
    }

    /// <summary>Male voice — Aidar.</summary>
    public static readonly SileroVoice Aidar = new("aidar", 0);

    /// <summary>Female voice — Baya.</summary>
    public static readonly SileroVoice Baya = new("baya", 1);

    /// <summary>Female voice — Kseniya.</summary>
    public static readonly SileroVoice Kseniya = new("kseniya", 2);

    /// <summary>Female voice — Xenia.</summary>
    public static readonly SileroVoice Xenia = new("xenia", 3);

    /// <summary>All available speakers for the v5_4_ru model.</summary>
    public static IReadOnlyList<SileroVoice> All { get; } = [Aidar, Baya, Kseniya, Xenia];

    /// <summary>
    /// Get a voice by name (case-insensitive).
    /// </summary>
    public static SileroVoice FromName(string name)
    {
        foreach (var voice in All)
        {
            if (string.Equals(voice.Name, name, StringComparison.OrdinalIgnoreCase))
                return voice;
        }

        throw new ArgumentException($"Unknown speaker: '{name}'. Available: {string.Join(", ", All.Select(v => v.Name))}");
    }

    public override string ToString() => Name;
}
