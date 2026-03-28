namespace SileroSharp.Processing;

/// <summary>
/// Detects whether input text is SSML markup.
/// </summary>
internal static class SsmlDetector
{
    /// <summary>
    /// Returns true if the text appears to be SSML (starts with &lt;speak tag).
    /// </summary>
    public static bool IsSsml(ReadOnlySpan<char> text)
    {
        var trimmed = text.TrimStart();
        if (!trimmed.StartsWith("<speak", StringComparison.OrdinalIgnoreCase))
            return false;

        // Must be exactly "<speak>" or "<speak " (with attributes), not "<speaker>" etc.
        if (trimmed.Length <= 6)
            return true;

        var charAfter = trimmed[6];
        return charAfter is '>' or ' ' or '\t' or '\r' or '\n';
    }
}
