using System.Text.Json;

namespace SileroSharp.Processing;

/// <summary>
/// Maps characters to token IDs for the Silero TTS model.
/// The symbol table is loaded from an embedded JSON resource extracted from the model.
/// </summary>
internal sealed class SymbolTable
{
    private readonly Dictionary<char, int> _charToId;
    private readonly Dictionary<int, char> _idToChar;

    public SymbolTable(Dictionary<char, int> charToId)
    {
        _charToId = charToId;
        _idToChar = new Dictionary<int, char>(charToId.Count);
        foreach (var (ch, id) in charToId)
        {
            _idToChar.TryAdd(id, ch);
        }
    }

    /// <summary>Number of symbols in the table.</summary>
    public int Count => _charToId.Count;

    /// <summary>SOS token ID (| character, prepended to every sequence).</summary>
    public int SosTokenId => _charToId.GetValueOrDefault('|', 2);

    /// <summary>EOS token ID (~ character, appended to every sequence).</summary>
    public int EosTokenId => _charToId.GetValueOrDefault('~', 1);

    /// <summary>
    /// Encode a text string into token IDs with SOS/EOS framing.
    /// The Silero model expects: [SOS, ...tokens..., EOS]
    /// Characters not in the symbol table are silently skipped.
    /// </summary>
    public long[] Encode(string text)
    {
        var tokens = new List<long>(text.Length + 2) { SosTokenId };
        var lower = text.ToLowerInvariant();

        foreach (var ch in lower)
        {
            if (_charToId.TryGetValue(ch, out var id))
            {
                tokens.Add(id);
            }
            // Unknown characters are skipped (matching Silero's behavior)
        }

        tokens.Add(EosTokenId);
        return tokens.ToArray();
    }

    /// <summary>
    /// Decode token IDs back to a string (for debugging).
    /// </summary>
    public string Decode(ReadOnlySpan<long> tokens)
    {
        var chars = new char[tokens.Length];
        var len = 0;

        foreach (var id in tokens)
        {
            if (_idToChar.TryGetValue((int)id, out var ch))
            {
                chars[len++] = ch;
            }
        }

        return new string(chars, 0, len);
    }

    /// <summary>
    /// Load the symbol table from a JSON file.
    /// The JSON maps character strings to integer IDs.
    /// </summary>
    public static SymbolTable LoadFromJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        return LoadFromJsonString(json);
    }

    /// <summary>
    /// Load the symbol table from a JSON string.
    /// </summary>
    public static SymbolTable LoadFromJsonString(string json)
    {
        var raw = JsonSerializer.Deserialize<Dictionary<string, int>>(json)
            ?? throw new InvalidOperationException("Failed to parse symbols.json");

        var charToId = new Dictionary<char, int>(raw.Count);
        foreach (var (key, id) in raw)
        {
            if (key.Length == 1)
            {
                charToId[key[0]] = id;
            }
            // Multi-character keys are ignored (shouldn't happen for Silero)
        }

        return new SymbolTable(charToId);
    }

    /// <summary>
    /// Create the default symbol table for the v5_4_ru model.
    /// </summary>
    public static SymbolTable CreateDefault() => CreateForVariant(SileroModelVariant.V5Russian);

    /// <summary>
    /// Create the symbol table for a specific model variant.
    /// </summary>
    public static SymbolTable CreateForVariant(SileroModelVariant variant)
    {
        // v5_4_ru: '_~|!+,-.:;?абвгдежзийклмнопрстуфхцчшщъыьэюяё–… '
        // v5_cis_base: '|!\'+,-.:;?hабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїјўґғҕҗҙқҝҡңҥҫүұҳҷҹһӑӗәӝӟӣӥӧөӯӱӳӵӏ—… '
        var symbols = variant switch
        {
            SileroModelVariant.V5Russian =>
                "_~|!+,-.:;?\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u2013\u2026 ",
            // The symbols string from model.symbols omits _~ at the start, but symbol_to_id has them at 0,1
            SileroModelVariant.V5CisBase =>
                "_~|!'+,-.:;?h\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0458\u045e\u0491\u0493\u0495\u0496\u0499\u049b\u049d\u04a1\u04a3\u04a5\u04ab\u04af\u04b1\u04b3\u04b7\u04b9\u04bb\u04d1\u04d7\u04d9\u04dd\u04df\u04e3\u04e5\u04e7\u04e9\u04ef\u04f1\u04f3\u04f5\u04cf\u2014\u2026 ",
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };

        var charToId = new Dictionary<char, int>(symbols.Length);
        for (var i = 0; i < symbols.Length; i++)
        {
            charToId[symbols[i]] = i;
        }

        return new SymbolTable(charToId);
    }
}
