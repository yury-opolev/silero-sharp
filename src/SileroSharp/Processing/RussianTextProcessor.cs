namespace SileroSharp.Processing;

/// <summary>
/// Orchestrates the full Russian text preprocessing pipeline:
/// normalize → yo restore → stress → split sentences → tokenize.
/// </summary>
internal sealed class RussianTextProcessor
{
    private readonly SymbolTable _symbolTable;
    private readonly HomoSolver _homoSolver;
    private readonly StressAccentor _accentor;
    private readonly YoRestorer _yoRestorer;
    private readonly SileroOptions _options;

    public RussianTextProcessor(
        SymbolTable symbolTable,
        HomoSolver homoSolver,
        StressAccentor accentor,
        YoRestorer yoRestorer,
        SileroOptions options)
    {
        _symbolTable = symbolTable;
        _homoSolver = homoSolver;
        _accentor = accentor;
        _yoRestorer = yoRestorer;
        _options = options;
    }

    /// <summary>
    /// Process text into tokenized sentences ready for inference.
    /// </summary>
    public ProcessedText Process(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return ProcessedText.Empty;

        // SSML detection — do not apply text processing to SSML
        if (SsmlDetector.IsSsml(text))
        {
            return new ProcessedText(
                IsSsml: true,
                Sentences: [new TokenizedSentence(text, _symbolTable.Encode(text))]);
        }

        // 1. Homograph resolution (BERT-based, requires original casing).
        var processed = _options.PutAccent ? _homoSolver.Resolve(text) : text;

        // 2. Sentence splitting — done on the cased text so the splitter's "next char is
        //    uppercase" heuristic for sentence boundaries works correctly. Lowercasing,
        //    yo-restoration, and stress-placement are applied per-sentence below.
        //    Short texts skip splitting entirely, matching Silero's Python behavior.
        List<string> sentences;
        if (processed.Length <= _options.MaxChunkLength)
        {
            sentences = [processed];
        }
        else
        {
            sentences = SentenceSplitter.Split(processed, _options.MaxChunkLength);
        }

        // 3. Per-sentence: lowercase → yo-restore → stress-place → tokenize.
        var tokenized = new List<TokenizedSentence>(sentences.Count);
        foreach (var sentence in sentences)
        {
            var lowered = sentence.ToLowerInvariant();
            var withYo = _options.PutYo ? _yoRestorer.Restore(lowered) : lowered;
            var stressed = _options.PutAccent ? _accentor.PlaceStress(withYo) : withYo;

            var trimmed = stressed.Trim();
            if (trimmed.Length == 0)
            {
                continue;
            }

            var tokens = _symbolTable.Encode(trimmed);
            if (tokens.Length > 2) // More than just SOS+EOS
            {
                tokenized.Add(new TokenizedSentence(trimmed, tokens));
            }
        }

        return new ProcessedText(IsSsml: false, Sentences: tokenized);
    }

    /// <summary>
    /// Create a processor with default settings (no stress dict, no yo dict).
    /// Useful for testing or when dictionaries are not available.
    /// </summary>
    public static RussianTextProcessor CreateDefault(SileroOptions? options = null)
    {
        return new RussianTextProcessor(
            SymbolTable.CreateDefault(),
            HomoSolver.CreatePassthrough(),
            StressAccentor.CreatePassthrough(),
            YoRestorer.CreatePassthrough(),
            options ?? new SileroOptions());
    }
}

/// <summary>
/// Result of text preprocessing: a list of tokenized sentences.
/// </summary>
internal sealed record ProcessedText(bool IsSsml, IReadOnlyList<TokenizedSentence> Sentences)
{
    public static ProcessedText Empty { get; } = new(false, []);
}

/// <summary>
/// A single sentence with its text and corresponding token IDs.
/// </summary>
internal sealed record TokenizedSentence(string Text, long[] Tokens);
