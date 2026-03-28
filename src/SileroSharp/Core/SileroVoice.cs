namespace SileroSharp;

/// <summary>
/// Represents a Silero TTS speaker voice.
/// </summary>
public sealed class SileroVoice
{
    public string Name { get; }
    public int SpeakerId { get; }

    private SileroVoice(string name, int speakerId)
    {
        this.Name = name;
        this.SpeakerId = speakerId;
    }

    /// <summary>Create a voice with explicit speaker ID.</summary>
    public static SileroVoice Create(string name, int speakerId) => new(name, speakerId);

    // --- v5_4_ru speakers (CC BY-NC 4.0) ---

    /// <summary>Male voice — Aidar (v5_4_ru).</summary>
    public static readonly SileroVoice Aidar = new("aidar", 0);

    /// <summary>Female voice — Baya (v5_4_ru).</summary>
    public static readonly SileroVoice Baya = new("baya", 1);

    /// <summary>Female voice — Kseniya (v5_4_ru).</summary>
    public static readonly SileroVoice Kseniya = new("kseniya", 2);

    /// <summary>Female voice — Xenia (v5_4_ru).</summary>
    public static readonly SileroVoice Xenia = new("xenia", 3);

    // --- v5_cis_base speakers (MIT, Russian subset) ---

    /// <summary>Female voice — Aigul (CIS base, MIT).</summary>
    public static readonly SileroVoice RuAigul = new("ru_aigul", 1);

    /// <summary>Female voice — Albina (CIS base, MIT).</summary>
    public static readonly SileroVoice RuAlbina = new("ru_albina", 3);

    /// <summary>Male voice — Alexandr (CIS base, MIT).</summary>
    public static readonly SileroVoice RuAlexandr = new("ru_alexandr", 5);

    /// <summary>Male voice — Bogdan (CIS base, MIT).</summary>
    public static readonly SileroVoice RuBogdan = new("ru_bogdan", 12);

    /// <summary>Male voice — Dmitriy (CIS base, MIT).</summary>
    public static readonly SileroVoice RuDmitriy = new("ru_dmitriy", 15);

    /// <summary>Female voice — Ekaterina (CIS base, MIT).</summary>
    public static readonly SileroVoice RuEkaterina = new("ru_ekaterina", 17);

    /// <summary>Male voice — Eduard (CIS base, MIT).</summary>
    public static readonly SileroVoice RuEduard = new("ru_eduard", 58);

    /// <summary>Female voice — Zhadyra (CIS base, MIT).</summary>
    public static readonly SileroVoice RuZhadyra = new("ru_zhadyra", 53);

    public override string ToString() => this.Name;
}
