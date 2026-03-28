using System.Text.Json;
using System.Text.RegularExpressions;
using TorchSharp;
using static TorchSharp.torch;

namespace SileroSharp.Processing;

/// <summary>
/// Resolves Russian homographs (words with context-dependent stress)
/// using a BERT-based classifier. Port of Silero's HomoSolver.
/// </summary>
internal sealed partial class HomoSolver : IDisposable
{
    private readonly jit.ScriptModule? model;
    private readonly Dictionary<string, List<string>>? homodict;
    private readonly Dictionary<string, int>? vocab;
    private readonly int homoStartId;
    private readonly int homoEndId;
    private readonly int padTokenId;
    private readonly int clsTokenId;
    private readonly int sepTokenId;
    private readonly int unkTokenId;

    private const string VowelChars = "аоуыэиеяёю";

    [GeneratedRegex(@"[А-Яа-яёЁ]+")]
    private static partial Regex WordFinder();

    private HomoSolver(
        jit.ScriptModule? model,
        Dictionary<string, List<string>>? homodict,
        Dictionary<string, int>? vocab,
        int homoStartId,
        int homoEndId,
        int padTokenId,
        int clsTokenId,
        int sepTokenId,
        int unkTokenId)
    {
        this.model = model;
        this.homodict = homodict;
        this.vocab = vocab;
        this.homoStartId = homoStartId;
        this.homoEndId = homoEndId;
        this.padTokenId = padTokenId;
        this.clsTokenId = clsTokenId;
        this.sepTokenId = sepTokenId;
        this.unkTokenId = unkTokenId;
    }

    /// <summary>
    /// Resolve homographs in a sentence by inserting stress marks.
    /// </summary>
    public string Resolve(string sentence, bool putStress = true, bool putYo = true, bool stressSingleVowel = true)
    {
        if (this.model is null || this.homodict is null || this.vocab is null)
        {
            return sentence;
        }

        // Find homographs in the sentence
        var tagged = this.FindAndTagHomos(sentence);
        if (tagged.Count == 0)
        {
            return sentence;
        }

        // Collect batch inputs for BERT
        var batchStarts = new List<long>();
        var batchEnds = new List<long>();
        var batchIds = new List<long[]>();
        var wordInfos = new List<(int Start, int End, string Word)>();

        foreach (var (start, end, word, isHomograph, rawMark) in tagged)
        {
            if (!isHomograph || rawMark is null)
            {
                continue;
            }

            var ids = this.Tokenize(rawMark);
            var homoStart = Array.IndexOf(ids, this.homoStartId);
            var homoEnd = Array.IndexOf(ids, this.homoEndId);

            if (homoStart < 0 || homoEnd < 0)
            {
                continue;
            }

            batchStarts.Add(homoStart);
            batchEnds.Add(homoEnd);
            batchIds.Add(ids);
            wordInfos.Add((start, end, word));
        }

        if (wordInfos.Count == 0)
        {
            return sentence;
        }

        // Pad sequences and run BERT
        var maxLen = 0;
        foreach (var ids in batchIds)
        {
            if (ids.Length > maxLen)
            {
                maxLen = ids.Length;
            }
        }

        var paddedBatch = new long[batchIds.Count * maxLen];
        for (var i = 0; i < batchIds.Count; i++)
        {
            var ids = batchIds[i];
            for (var j = 0; j < maxLen; j++)
            {
                paddedBatch[i * maxLen + j] = j < ids.Length ? ids[j] : this.padTokenId;
            }
        }

        using var noGrad = no_grad();
        using var inputTensor = tensor(paddedBatch, dtype: ScalarType.Int64)
            .reshape(batchIds.Count, maxLen);
        using var startsTensor = tensor(batchStarts.ToArray(), dtype: ScalarType.Int64);
        using var endsTensor = tensor(batchEnds.ToArray(), dtype: ScalarType.Int64);

        var result = this.model.forward(inputTensor, startsTensor, endsTensor);
        Tensor logits;
        if (result is Tensor t)
        {
            logits = t;
        }
        else
        {
            var tuple = (System.Runtime.CompilerServices.ITuple)result!;
            logits = (Tensor)tuple[0]!;
        }

        // Apply predictions
        var stressedSent = sentence;
        var offset = 0;

        for (var i = 0; i < wordInfos.Count; i++)
        {
            var (start, end, word) = wordInfos[i];
            start += offset;
            end += offset;

            var logitVal = logits[i].item<float>();
            var pred = logitVal > 0 ? 1 : 0; // round(sigmoid(x)) == (x > 0)

            var variants = this.homodict![word.ToLowerInvariant()];
            var sortedVariants = variants.OrderBy(v => v).ToList();
            var wordPred = sortedVariants[Math.Min(pred, sortedVariants.Count - 1)];

            if (!putYo)
            {
                wordPred = wordPred.Replace("ё", "е");
            }

            var numVowels = wordPred.Count(c => VowelChars.Contains(c));
            var stressIdx = wordPred.IndexOf('+');
            wordPred = wordPred.Replace("+", "");

            // Transfer case from original word
            var cased = new char[wordPred.Length];
            for (var j = 0; j < wordPred.Length && j < word.Length; j++)
            {
                cased[j] = char.IsUpper(word[j]) ? char.ToUpperInvariant(wordPred[j]) : wordPred[j];
            }

            for (var j = word.Length; j < wordPred.Length; j++)
            {
                cased[j] = wordPred[j];
            }

            var casedStr = new string(cased);

            if ((numVowels > 1 || stressSingleVowel) && putStress && stressIdx >= 0)
            {
                casedStr = casedStr[..stressIdx] + "+" + casedStr[stressIdx..];
                offset += 1;
            }

            stressedSent = stressedSent[..start] + casedStr + stressedSent[end..];
        }

        return stressedSent;
    }

    private List<(int Start, int End, string Word, bool IsHomograph, string? RawMark)> FindAndTagHomos(string sentence)
    {
        var results = new List<(int, int, string, bool, string?)>();

        foreach (Match match in WordFinder().Matches(sentence))
        {
            var start = match.Index;
            var end = start + match.Length;
            var word = match.Value;
            var isHomograph = this.homodict!.ContainsKey(word.ToLowerInvariant());

            string? rawMark = null;
            if (isHomograph)
            {
                rawMark = sentence[..start] + " [HOMO] " + sentence[start..end] + " [/HOMO] " + sentence[end..];
            }

            results.Add((start, end, word, isHomograph, rawMark));
        }

        return results;
    }

    /// <summary>
    /// Simple BERT WordPiece tokenizer: basic tokenize + wordpiece + [CLS]/[SEP].
    /// </summary>
    private long[] Tokenize(string text)
    {
        // Basic tokenization: split on whitespace and punctuation
        var basicTokens = BasicTokenize(text);

        // WordPiece tokenization
        var wpTokens = new List<string> { "[CLS]" };
        foreach (var token in basicTokens)
        {
            if (token == "[HOMO]" || token == "[/HOMO]")
            {
                wpTokens.Add(token);
                continue;
            }

            WordPieceTokenize(token, wpTokens);
        }

        wpTokens.Add("[SEP]");

        // Convert to IDs
        var ids = new long[wpTokens.Count];
        for (var i = 0; i < wpTokens.Count; i++)
        {
            ids[i] = this.vocab!.GetValueOrDefault(wpTokens[i], this.unkTokenId);
        }

        return ids;
    }

    private static readonly string[] NeverSplit = ["[HOMO]", "[/HOMO]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "[PAD]"];

    private static List<string> BasicTokenize(string text)
    {
        // First, protect never_split tokens by replacing them with placeholders
        var protected_ = new List<(string Token, string Placeholder)>();
        var processed = text;
        for (var i = 0; i < NeverSplit.Length; i++)
        {
            var ns = NeverSplit[i];
            if (processed.Contains(ns))
            {
                var placeholder = $"\x00NS{i}\x00";
                protected_.Add((ns, placeholder));
                processed = processed.Replace(ns, placeholder);
            }
        }

        // Basic tokenize: split on whitespace and punctuation
        var tokens = new List<string>();
        var current = new System.Text.StringBuilder();

        foreach (var ch in processed)
        {
            if (ch == '\x00')
            {
                // Start/end of placeholder — flush current token
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
            }
            else if (char.IsWhiteSpace(ch))
            {
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }
            }
            else if (char.IsPunctuation(ch) || char.IsSymbol(ch))
            {
                if (current.Length > 0)
                {
                    tokens.Add(current.ToString());
                    current.Clear();
                }

                tokens.Add(ch.ToString());
            }
            else
            {
                current.Append(ch);
            }
        }

        if (current.Length > 0)
        {
            tokens.Add(current.ToString());
        }

        // Restore never_split tokens
        for (var i = 0; i < tokens.Count; i++)
        {
            foreach (var (token, placeholder) in protected_)
            {
                // The placeholder chars (without \x00) remain as the token
                var inner = placeholder.Trim('\x00');
                if (tokens[i] == inner)
                {
                    tokens[i] = token;
                }
            }
        }

        return tokens;
    }

    private void WordPieceTokenize(string token, List<string> output)
    {
        if (token.Length > 200)
        {
            output.Add("[UNK]");
            return;
        }

        var chars = token.ToCharArray();
        var start = 0;

        while (start < chars.Length)
        {
            var end = chars.Length;
            string? found = null;

            while (start < end)
            {
                var substr = new string(chars, start, end - start);
                if (start > 0)
                {
                    substr = "##" + substr;
                }

                if (this.vocab!.ContainsKey(substr))
                {
                    found = substr;
                    break;
                }

                end--;
            }

            if (found is null)
            {
                output.Add("[UNK]");
                return;
            }

            output.Add(found);
            start = end;
        }
    }

    public static HomoSolver LoadFromDirectory(string directory)
    {
        var modelPath = Path.Combine(directory, "homosolver_bert.pt");
        if (!File.Exists(modelPath))
        {
            return new HomoSolver(null, null, null, 0, 0, 0, 0, 0, 0);
        }

        var bertModel = jit.load(modelPath);
        bertModel.eval();

        var homodictJson = File.ReadAllText(Path.Combine(directory, "homodict.json"), System.Text.Encoding.UTF8);
        var homodict = JsonSerializer.Deserialize<Dictionary<string, List<string>>>(homodictJson)!;

        var vocabJson = File.ReadAllText(Path.Combine(directory, "homo_vocab.json"), System.Text.Encoding.UTF8);
        var vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(vocabJson)!;

        var specialJson = File.ReadAllText(Path.Combine(directory, "homo_special.json"));
        var special = JsonSerializer.Deserialize<Dictionary<string, int>>(specialJson)!;

        return new HomoSolver(
            bertModel, homodict, vocab,
            special["homo_start_id"],
            special["homo_end_id"],
            special.GetValueOrDefault("pad_token_id", 0),
            special.GetValueOrDefault("cls_token_id", 2),
            special.GetValueOrDefault("sep_token_id", 3),
            special.GetValueOrDefault("unk_token_id", 1));
    }

    public static HomoSolver CreatePassthrough() => new(null, null, null, 0, 0, 0, 0, 0, 0);

    public void Dispose()
    {
        this.model?.Dispose();
    }
}
