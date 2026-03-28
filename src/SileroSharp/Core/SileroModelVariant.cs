namespace SileroSharp;

/// <summary>
/// Identifies which Silero TTS model variant is in use.
/// </summary>
public enum SileroModelVariant
{
    /// <summary>
    /// v5_4_ru — 4 Russian speakers (aidar, baya, kseniya, xenia).
    /// License: CC BY-NC 4.0 (non-commercial).
    /// Includes built-in accentor.
    /// </summary>
    V5Russian,

    /// <summary>
    /// v5_cis_base — 60 speakers across CIS languages (Russian, Tatar, Bashkir, Kazakh, etc.).
    /// License: MIT (commercial use OK).
    /// No built-in accentor — stress must be provided externally.
    /// </summary>
    V5CisBase,
}
