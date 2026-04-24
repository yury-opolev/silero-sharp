using System.Text;

namespace SileroSharp.Processing;

/// <summary>
/// Splits Russian text into sentence-sized chunks suitable for Silero TTS inference.
/// Respects abbreviations, asterisk markers, and the 130-character soft limit.
/// </summary>
internal static class SentenceSplitter
{
    private static readonly HashSet<string> Abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        "т.е", "т.д", "т.п", "т.н", "т.к", "т.о",
        "др", "пр", "пр-т",
        "г", "гг", "г.г",
        "ул", "д", "кв", "корп", "стр",
        "руб", "коп", "тыс",
        "млн", "млрд", "трлн",
        "км", "м", "см", "мм", "кг", "гр", "мг", "л", "мл",
        "н.э", "до н.э",
        "в", "вв",
        "проф", "акад", "доц", "канд",
        "им", "тел", "факс",
        "рис", "табл", "гл", "стр", "см",
        "р", "к", "с",
    };

    /// <summary>
    /// Split text into chunks, each under <paramref name="maxLength"/> characters.
    /// Each chunk ends with terminal punctuation where possible.
    /// </summary>
    public static List<string> Split(string text, int maxLength = 130)
    {
        if (string.IsNullOrWhiteSpace(text))
            return [];

        text = NormalizeWhitespace(text);

        var sentences = SplitOnSentenceBoundaries(text);
        var result = new List<string>();

        foreach (var sentence in sentences)
        {
            var trimmed = sentence.Trim();
            if (trimmed.Length == 0)
                continue;

            if (trimmed.Length <= maxLength)
            {
                result.Add(EnsureTerminalPunctuation(trimmed));
            }
            else
            {
                // Secondary split at commas, semicolons, or dashes.
                // Pass through the original sentence's terminator so only the LAST fragment
                // gets sentence-final intonation; non-final fragments keep ',' for continuing prosody.
                var terminator = ExtractTerminator(trimmed);
                var subChunks = SplitLongSentence(trimmed, maxLength, terminator);
                result.AddRange(subChunks);
            }
        }

        return result;
    }

    private static List<string> SplitOnSentenceBoundaries(string text)
    {
        var sentences = new List<string>();
        var current = new StringBuilder();

        for (var i = 0; i < text.Length; i++)
        {
            var ch = text[i];
            current.Append(ch);

            // Check for terminal punctuation
            if (ch is not ('.' or '!' or '?'))
                continue;

            // Handle ellipsis (...)
            if (ch == '.' && i + 1 < text.Length && text[i + 1] == '.')
                continue;

            // Check if this is an abbreviation
            if (ch == '.' && IsAbbreviation(text, i))
                continue;

            // Check if followed by whitespace + uppercase (or end of string)
            if (i + 1 >= text.Length || IsFollowedByNewSentence(text, i + 1))
            {
                sentences.Add(current.ToString());
                current.Clear();
            }
        }

        // Remainder
        if (current.Length > 0)
        {
            sentences.Add(current.ToString());
        }

        return sentences;
    }

    private static bool IsFollowedByNewSentence(string text, int startIndex)
    {
        // Skip whitespace
        var i = startIndex;
        while (i < text.Length && char.IsWhiteSpace(text[i]))
            i++;

        if (i >= text.Length)
            return true;

        // New sentence if next non-whitespace is uppercase or a quote/bracket
        var next = text[i];
        return char.IsUpper(next) || next is '"' or '\'' or '«' or '(' or '—' or '-';
    }

    private static bool IsAbbreviation(string text, int dotIndex)
    {
        // Find the word before the dot
        var wordStart = dotIndex - 1;
        while (wordStart >= 0 && char.IsLetter(text[wordStart]))
            wordStart--;
        wordStart++;

        if (wordStart >= dotIndex)
            return false;

        var word = text[wordStart..dotIndex];

        // Check against known abbreviations
        if (Abbreviations.Contains(word))
            return true;

        // Also check if the word+dot is part of a multi-dot abbreviation (e.g. "т.е.")
        // Look for pattern like "X.X" where both parts are 1-3 chars
        if (dotIndex + 1 < text.Length && char.IsLetter(text[dotIndex + 1]))
        {
            // Could be "т.е", "т.д", etc.
            var afterDot = dotIndex + 1;
            var nextWordEnd = afterDot;
            while (nextWordEnd < text.Length && char.IsLetter(text[nextWordEnd]))
                nextWordEnd++;

            var combined = word + "." + text[afterDot..nextWordEnd];
            if (Abbreviations.Contains(combined))
                return true;
        }

        // Single letter followed by dot is likely an abbreviation
        if (word.Length == 1 && char.IsUpper(text[wordStart]))
            return true;

        return false;
    }

    private static List<string> SplitLongSentence(string sentence, int maxLength, char terminator)
    {
        var chunks = new List<string>();
        var remaining = sentence.AsSpan();

        while (remaining.Length > maxLength)
        {
            var splitIndex = FindSecondarySplitPoint(remaining, maxLength);
            var isForceSplit = splitIndex <= 0;
            if (isForceSplit)
            {
                // Force split at maxLength
                splitIndex = maxLength;
                // Try not to split inside a word
                while (splitIndex > maxLength / 2 && !char.IsWhiteSpace(remaining[splitIndex]))
                    splitIndex--;
                if (splitIndex <= maxLength / 2)
                    splitIndex = maxLength;
            }

            var chunk = remaining[..splitIndex].ToString().Trim();
            if (chunk.Length > 0)
            {
                // Non-final fragment: end with ',' to suggest a continuing-thought pause.
                // For comma/semicolon/dash splits the chunk already ends with that punctuation;
                // normalize anything else (including force-splits with no trailing punctuation) to ','.
                chunks.Add(EnsureContinuingPunctuation(chunk));
            }

            remaining = remaining[splitIndex..].TrimStart();
        }

        if (remaining.Length > 0)
        {
            var last = remaining.ToString().Trim();
            if (last.Length > 0)
            {
                // Final fragment: apply the original sentence's terminator.
                chunks.Add(ApplyTerminator(last, terminator));
            }
        }

        return chunks;
    }

    private static int FindSecondarySplitPoint(ReadOnlySpan<char> text, int maxLength)
    {
        // Look for comma, semicolon, or dash within the maxLength limit
        // Search backwards from maxLength to find the best split point
        var bestSplit = -1;

        for (var i = Math.Min(maxLength, text.Length) - 1; i > maxLength / 3; i--)
        {
            var ch = text[i];
            if (ch is ',' or ';' or '—' or '–')
            {
                // Don't split inside asterisk markers
                if (!IsInsideAsteriskMarker(text, i))
                {
                    bestSplit = i + 1;
                    break;
                }
            }
        }

        return bestSplit;
    }

    private static bool IsInsideAsteriskMarker(ReadOnlySpan<char> text, int index)
    {
        // Check if index is between a pair of asterisks (*word*)
        var asterisksBefore = 0;
        for (var i = 0; i < index; i++)
        {
            if (text[i] == '*')
                asterisksBefore++;
        }

        // If odd number of asterisks before this point, we're inside a marker
        return asterisksBefore % 2 == 1;
    }

    private static string EnsureTerminalPunctuation(string text)
    {
        if (text.Length == 0)
            return text;

        var lastChar = text[^1];
        if (lastChar is '.' or '!' or '?' or '…')
            return text;

        // If ends with comma, semicolon, or other mid-sentence punctuation, replace with period
        if (lastChar is ',' or ';' or ':' or '—' or '–')
            return text[..^1] + ".";

        return text + ".";
    }

    /// <summary>
    /// Returns the terminator of an entire sentence ('.', '!' or '?'). Used to remember
    /// the original ending so a long sentence's last fragment can carry it after splitting.
    /// </summary>
    private static char ExtractTerminator(string sentence)
    {
        if (sentence.Length == 0)
            return '.';

        var lastChar = sentence[^1];
        return lastChar switch
        {
            '.' or '!' or '?' or '…' => lastChar,
            _ => '.',
        };
    }

    /// <summary>
    /// For non-final fragments of a long sentence: ensure ending punctuation suggests a
    /// continuing-thought pause (',') rather than sentence-final intonation.
    /// Preserves natural mid-sentence punctuation (',', ';', '—', '–'); replaces any
    /// terminator and appends ',' if none exists.
    /// </summary>
    private static string EnsureContinuingPunctuation(string text)
    {
        if (text.Length == 0)
            return text;

        var lastChar = text[^1];

        // Already mid-sentence punctuation — keep as-is for natural pause.
        if (lastChar is ',' or ';' or '—' or '–')
            return text;

        // Strip a colon (rarely appears in TTS output but keep behavior consistent).
        if (lastChar is ':')
            return text[..^1] + ",";

        // Strip any sentence-final punctuation that snuck in (e.g. abbreviation period
        // misclassified earlier) — we want continuing prosody here.
        if (lastChar is '.' or '!' or '?' or '…')
            return text[..^1] + ",";

        return text + ",";
    }

    /// <summary>
    /// Apply the original sentence's terminator to the final fragment.
    /// Replaces any continuing punctuation (',', ';', etc.) with the terminator.
    /// </summary>
    private static string ApplyTerminator(string text, char terminator)
    {
        if (text.Length == 0)
            return text;

        var lastChar = text[^1];

        // Already terminated correctly.
        if (lastChar == terminator || lastChar is '…')
            return text;

        // Replace any other punctuation with the terminator.
        if (lastChar is '.' or '!' or '?' or ',' or ';' or ':' or '—' or '–')
            return text[..^1] + terminator;

        return text + terminator;
    }

    private static string NormalizeWhitespace(string text)
    {
        var sb = new StringBuilder(text.Length);
        var prevWasSpace = false;

        foreach (var ch in text)
        {
            if (char.IsWhiteSpace(ch))
            {
                if (!prevWasSpace)
                {
                    sb.Append(' ');
                    prevWasSpace = true;
                }
            }
            else
            {
                sb.Append(ch);
                prevWasSpace = false;
            }
        }

        return sb.ToString().Trim();
    }
}
