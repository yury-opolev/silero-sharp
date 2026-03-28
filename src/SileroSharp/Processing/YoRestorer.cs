using System.Text;
using System.Text.Json;

namespace SileroSharp.Processing;

/// <summary>
/// Restores the Russian letter ё (yo) in place of е (ye) where appropriate.
/// Uses a dictionary of words that contain ё.
/// </summary>
internal sealed class YoRestorer
{
    private readonly Dictionary<string, string> _yoDict;

    public YoRestorer(Dictionary<string, string> yoDict)
    {
        _yoDict = yoDict;
    }

    /// <summary>
    /// Restore ё in the given text.
    /// </summary>
    public string Restore(string text)
    {
        if (string.IsNullOrWhiteSpace(text) || _yoDict.Count == 0)
            return text;

        var sb = new StringBuilder(text.Length);
        var wordStart = -1;

        for (var i = 0; i <= text.Length; i++)
        {
            var isLetter = i < text.Length && char.IsLetter(text[i]);

            if (isLetter && wordStart == -1)
            {
                wordStart = i;
            }
            else if (!isLetter && wordStart != -1)
            {
                var word = text[wordStart..i];
                sb.Append(RestoreWord(word));
                wordStart = -1;

                if (i < text.Length)
                    sb.Append(text[i]);
            }
            else if (!isLetter && i < text.Length)
            {
                sb.Append(text[i]);
            }
        }

        return sb.ToString();
    }

    private string RestoreWord(string word)
    {
        var lower = word.ToLowerInvariant();

        // Already has ё
        if (lower.Contains('ё'))
            return word;

        // Doesn't contain е — nothing to restore
        if (!lower.Contains('е'))
            return word;

        if (!_yoDict.TryGetValue(lower, out var yoForm))
            return word;

        // Transfer original casing
        var sb = new StringBuilder(yoForm.Length);
        for (var i = 0; i < yoForm.Length && i < word.Length; i++)
        {
            if (char.IsUpper(word[i]))
                sb.Append(char.ToUpperInvariant(yoForm[i]));
            else
                sb.Append(yoForm[i]);
        }

        // Append any remaining chars from yoForm if lengths differ
        if (yoForm.Length > word.Length)
            sb.Append(yoForm[word.Length..]);

        return sb.ToString();
    }

    /// <summary>
    /// Load yo dictionary from a JSON file.
    /// </summary>
    public static YoRestorer LoadFromJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        var dict = JsonSerializer.Deserialize<Dictionary<string, string>>(json)
            ?? throw new InvalidOperationException("Failed to parse yo dictionary");
        return new YoRestorer(dict);
    }

    /// <summary>
    /// Create a no-op restorer that doesn't change text.
    /// </summary>
    public static YoRestorer CreatePassthrough() => new(new Dictionary<string, string>());
}
