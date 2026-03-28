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

    // ================================================================
    // v5_4_ru speakers (CC BY-NC 4.0)
    // ================================================================

    /// <summary>Male voice — Aidar (v5_4_ru).</summary>
    public static readonly SileroVoice Aidar = new("aidar", 0);

    /// <summary>Female voice — Baya (v5_4_ru).</summary>
    public static readonly SileroVoice Baya = new("baya", 1);

    /// <summary>Female voice — Kseniya (v5_4_ru).</summary>
    public static readonly SileroVoice Kseniya = new("kseniya", 2);

    /// <summary>Female voice — Xenia (v5_4_ru).</summary>
    public static readonly SileroVoice Xenia = new("xenia", 3);

    // ================================================================
    // v5_cis_base Russian speakers (MIT)
    // ================================================================

    /// <summary>Female — Aigul (CIS, MIT).</summary>
    public static readonly SileroVoice RuAigul = new("ru_aigul", 1);

    /// <summary>Female — Albina (CIS, MIT).</summary>
    public static readonly SileroVoice RuAlbina = new("ru_albina", 3);

    /// <summary>Male — Alexandr (CIS, MIT).</summary>
    public static readonly SileroVoice RuAlexandr = new("ru_alexandr", 5);

    /// <summary>Female — Alfia (CIS, MIT).</summary>
    public static readonly SileroVoice RuAlfia = new("ru_alfia", 7);

    /// <summary>Female — Alfia 2 (CIS, MIT).</summary>
    public static readonly SileroVoice RuAlfia2 = new("ru_alfia2", 9);

    /// <summary>Male — Bogdan (CIS, MIT).</summary>
    public static readonly SileroVoice RuBogdan = new("ru_bogdan", 12);

    /// <summary>Male — Dmitriy (CIS, MIT).</summary>
    public static readonly SileroVoice RuDmitriy = new("ru_dmitriy", 14);

    /// <summary>Male — Eduard (CIS, MIT).</summary>
    public static readonly SileroVoice RuEduard = new("ru_eduard", 58);

    /// <summary>Female — Ekaterina (CIS, MIT).</summary>
    public static readonly SileroVoice RuEkaterina = new("ru_ekaterina", 16);

    /// <summary>Male — Gamat (CIS, MIT).</summary>
    public static readonly SileroVoice RuGamat = new("ru_gamat", 20);

    /// <summary>Male — Igor (CIS, MIT).</summary>
    public static readonly SileroVoice RuIgor = new("ru_igor", 22);

    /// <summary>Female — Karina (CIS, MIT).</summary>
    public static readonly SileroVoice RuKarina = new("ru_karina", 24);

    /// <summary>Male — Kejilgan (CIS, MIT).</summary>
    public static readonly SileroVoice RuKejilgan = new("ru_kejilgan", 26);

    /// <summary>Female — Kermen (CIS, MIT).</summary>
    public static readonly SileroVoice RuKermen = new("ru_kermen", 28);

    /// <summary>Male — Marat (CIS, MIT).</summary>
    public static readonly SileroVoice RuMarat = new("ru_marat", 31);

    /// <summary>Female — Miyau (CIS, MIT).</summary>
    public static readonly SileroVoice RuMiyau = new("ru_miyau", 33);

    /// <summary>Female — Nurgul (CIS, MIT).</summary>
    public static readonly SileroVoice RuNurgul = new("ru_nurgul", 35);

    /// <summary>Female — Oksana (CIS, MIT).</summary>
    public static readonly SileroVoice RuOksana = new("ru_oksana", 37);

    /// <summary>Male — Onaoy (CIS, MIT).</summary>
    public static readonly SileroVoice RuOnaoy = new("ru_onaoy", 39);

    /// <summary>Female — Ramilia (CIS, MIT).</summary>
    public static readonly SileroVoice RuRamilia = new("ru_ramilia", 41);

    /// <summary>Male — Roman (CIS, MIT).</summary>
    public static readonly SileroVoice RuRoman = new("ru_roman", 43);

    /// <summary>Male — Safarhuja (CIS, MIT).</summary>
    public static readonly SileroVoice RuSafarhuja = new("ru_safarhuja", 45);

    /// <summary>Female — Saida (CIS, MIT).</summary>
    public static readonly SileroVoice RuSaida = new("ru_saida", 47);

    /// <summary>Male — Sibday (CIS, MIT).</summary>
    public static readonly SileroVoice RuSibday = new("ru_sibday", 49);

    /// <summary>Female — Vika (CIS, MIT).</summary>
    public static readonly SileroVoice RuVika = new("ru_vika", 18);

    /// <summary>Female — Zara (CIS, MIT).</summary>
    public static readonly SileroVoice RuZara = new("ru_zara", 51);

    /// <summary>Female — Zhadyra (CIS, MIT).</summary>
    public static readonly SileroVoice RuZhadyra = new("ru_zhadyra", 53);

    /// <summary>Female — Zhazira (CIS, MIT).</summary>
    public static readonly SileroVoice RuZhazira = new("ru_zhazira", 55);

    /// <summary>Female — Zinaida (CIS, MIT).</summary>
    public static readonly SileroVoice RuZinaida = new("ru_zinaida", 57);

    public override string ToString() => this.Name;
}
