using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace SileroSharp.Processing;

/// <summary>
/// Places stress markers ('+' before stressed vowel) and restores ё on Russian words.
/// Port of the Silero neural accentor: FastText n-gram embedding + MLP classifiers.
/// </summary>
internal sealed partial class StressAccentor : IDisposable
{
    private readonly Dictionary<string, int>? ngramDict;
    private readonly float[,]? embeddingWeight; // [vocabSize, embDim]
    private readonly MlpLayer[]? stressClf;
    private readonly MlpLayer[]? yoClf;
    private readonly Dictionary<string, (int Stress, int Yo)>? exceptions;
    private readonly Dictionary<string, string>? dictFallback;
    private readonly int embDim;

    private const string VowelChars = "аоуыэиеяёю";
    private static readonly HashSet<char> Vowels = [.. VowelChars];

    // Capturing group ensures Regex.Split preserves the delimiters in the output
    [GeneratedRegex(@"([\s.,!?;:<>=()/\\]+)")]
    private static partial Regex TokenSplitPattern();

    [GeneratedRegex(@"[^А-Яа-яёЁ]")]
    private static partial Regex NonCyrillicPattern();

    [GeneratedRegex(@"[А-Яа-яёЁ]+")]
    private static partial Regex WordPattern();

    private StressAccentor(
        Dictionary<string, int>? ngramDict,
        float[,]? embeddingWeight,
        MlpLayer[]? stressClf,
        MlpLayer[]? yoClf,
        Dictionary<string, (int Stress, int Yo)>? exceptions,
        Dictionary<string, string>? dictFallback = null)
    {
        this.ngramDict = ngramDict;
        this.embeddingWeight = embeddingWeight;
        this.stressClf = stressClf;
        this.yoClf = yoClf;
        this.exceptions = exceptions;
        this.dictFallback = dictFallback;
        this.embDim = embeddingWeight?.GetLength(1) ?? 0;
    }

    /// <summary>
    /// Place stress markers and restore ё on Russian text using the neural accentor.
    /// Follows the Silero Python accentor pipeline exactly.
    /// </summary>
    public string PlaceStress(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return text;
        }

        // Passthrough mode: no model and no fallback
        if (this.ngramDict is null && this.dictFallback is null)
        {
            return text;
        }

        // Neural accentor pipeline (full Silero port)
        if (this.ngramDict is not null)
        {
            return this.RunFullPipeline(text, putStress: true, putYo: true, stressSingleVowel: true);
        }

        // Dictionary fallback (simple mode, no tokenizer)
        if (text.Contains('+'))
        {
            return text;
        }

        return WordPattern().Replace(text, match =>
        {
            var word = match.Value;
            if (CountVowels(word) <= 1)
            {
                return word;
            }

            var lower = word.ToLowerInvariant();
            if (this.dictFallback!.TryGetValue(lower, out var stressed))
            {
                return TransferCase(word, stressed);
            }

            return word;
        });
    }

    /// <summary>
    /// Full Silero accentor pipeline: tokenize, predict, apply stress and ё.
    /// </summary>
    private string RunFullPipeline(string sentence, bool putStress, bool putYo, bool stressSingleVowel)
    {
        var (rawTokens, cleanTokens, predictionMask) = Tokenize(sentence);

        // Get model predictions (only for processable tokens)
        var (stressProbs, stressPreds, yoProbs, yoPreds) = this.GetModelPreds(cleanTokens, predictionMask);

        var accented = new StringBuilder(sentence.Length + rawTokens.Count);
        var predIdx = 0;

        for (var wordIdx = 0; wordIdx < rawTokens.Count; wordIdx++)
        {
            var rawWord = rawTokens[wordIdx];
            var cleanWord = cleanTokens[wordIdx];
            var needProcessing = predictionMask[wordIdx];

            if (!needProcessing)
            {
                accented.Append(rawWord);
                continue;
            }

            // This token was processed by the model; get its prediction index
            var myPredIdx = predIdx;
            predIdx++;

            var rawLower = rawWord.ToLowerInvariant();
            var haveStress = rawLower.Contains('+');
            var haveYo = rawLower.Contains('ё');

            // Already has both stress and ё -- nothing to do
            if (haveStress && haveYo)
            {
                accented.Append(rawWord);
                continue;
            }

            // Has ё but no stress -- place stress before each ё
            if (!haveStress && haveYo && putStress)
            {
                var modified = rawWord;
                var yoPositions = new List<int>();
                var lowerMod = modified.ToLowerInvariant();
                for (var i = 0; i < lowerMod.Length; i++)
                {
                    if (lowerMod[i] == 'ё')
                    {
                        yoPositions.Add(i);
                    }
                }

                for (var i = 0; i < yoPositions.Count; i++)
                {
                    var insertAt = yoPositions[i] + i; // offset for previously inserted '+'
                    modified = modified[..insertAt] + "+" + modified[insertAt..];
                }

                accented.Append(modified);
                continue;
            }

            // Check exceptions dict (char positions, not vowel indices)
            if (this.exceptions is not null && this.exceptions.TryGetValue(cleanWord, out var exc))
            {
                var modified = rawWord;
                var excStress = exc.Stress; // char position
                var excYo = exc.Yo;         // char position, -1 if none

                if (excYo != -1 && excYo < modified.Length)
                {
                    modified = modified[..excYo] + "ё" + modified[(excYo + 1)..];
                }

                if (excStress >= 0 && excStress <= modified.Length)
                {
                    modified = modified[..excStress] + "+" + modified[excStress..];
                }

                accented.Append(modified);
                continue;
            }

            // Neural model prediction
            var stressedVowelIdx = stressPreds[myPredIdx];
            var stressConfidence = stressProbs[myPredIdx][stressedVowelIdx];
            var setStress = stressConfidence > 0.5f && !haveStress;

            var yoVowelIdx = yoPreds[myPredIdx]; // 0 = no ё, 1+ = ye_index (1-based)
            var yoConfidence = yoProbs[myPredIdx][yoVowelIdx];
            var setYo = yoVowelIdx > 0 && yoConfidence > 0.5f;

            // Convert vowel indices to char positions
            var vowelPositions = new List<int>();
            var yePositions = new List<int>();
            var wordLower = rawWord.ToLowerInvariant();
            for (var i = 0; i < wordLower.Length; i++)
            {
                var ch = wordLower[i];
                if (Vowels.Contains(ch))
                {
                    vowelPositions.Add(i);
                }

                if (ch == 'е')
                {
                    yePositions.Add(i);
                }
            }

            var numVowels = vowelPositions.Count;
            if (numVowels == 0)
            {
                accented.Append(rawWord);
                continue;
            }

            var stressPos = stressedVowelIdx < vowelPositions.Count
                ? vowelPositions[stressedVowelIdx]
                : -1;

            var yoPos = (yoVowelIdx > 0 && yoVowelIdx - 1 < yePositions.Count)
                ? yePositions[yoVowelIdx - 1]
                : -1;

            var currentWord = rawWord;

            // Apply ё replacement
            if (yoPos >= 0 && yoPos == stressPos && setYo && putYo)
            {
                if (yoPos < currentWord.Length && char.ToLowerInvariant(currentWord[yoPos]) == 'е')
                {
                    var replacement = char.IsUpper(currentWord[yoPos]) ? 'Ё' : 'ё';
                    currentWord = currentWord[..yoPos] + replacement + currentWord[(yoPos + 1)..];
                }
            }

            // Single-vowel words always get stress
            if (numVowels == 1)
            {
                stressPos = vowelPositions[0];
                setStress = stressSingleVowel && putStress;
            }

            // Apply stress mark
            if (!haveStress && setStress && stressPos >= 0 && putStress)
            {
                currentWord = currentWord[..stressPos] + "+" + currentWord[stressPos..];
            }

            accented.Append(currentWord);
        }

        return accented.ToString();
    }

    /// <summary>
    /// Tokenize a sentence into raw tokens, cleaned tokens, and a prediction mask.
    /// Matches the Python _tokenize method exactly.
    /// </summary>
    private static (List<string> RawTokens, List<string> CleanTokens, List<bool> PredictionMask) Tokenize(string sentence)
    {
        var rawTokens = new List<string>();
        var cleanTokens = new List<string>();
        var predictionMask = new List<bool>();

        // Split on delimiters, keeping delimiters as separate tokens
        var parts = TokenSplitPattern().Split(sentence);

        foreach (var word in parts)
        {
            if (word.Length == 0)
            {
                continue;
            }

            // Check if this is a delimiter (matches the split pattern)
            if (TokenSplitPattern().IsMatch(word))
            {
                rawTokens.Add(word);
                cleanTokens.Add("");
                predictionMask.Add(false);
                continue;
            }

            // Split on hyphens
            var hyphenParts = word.Split('-');

            List<string> curTokens;
            List<bool> curMask;

            if (hyphenParts.Length == 1)
            {
                curTokens = [hyphenParts[0]];
                curMask = [true];
            }
            else
            {
                curTokens = new List<string>(hyphenParts.Length);
                curMask = new List<bool>(hyphenParts.Length);

                for (var i = 0; i < hyphenParts.Length - 1; i++)
                {
                    curTokens.Add(hyphenParts[i] + "-");
                    curMask.Add(true);
                }

                curTokens.Add(hyphenParts[^1]);
                // "то" after hyphen is not processed (e.g. "что-то", "кто-то")
                curMask.Add(hyphenParts[^1] != "то");
            }

            // Clean each token: remove non-Cyrillic, lowercase
            var curClean = new List<string>(curTokens.Count);
            for (var i = 0; i < curTokens.Count; i++)
            {
                curClean.Add(NonCyrillicPattern().Replace(curTokens[i].ToLowerInvariant(), ""));
            }

            // Update mask: need non-empty clean token AND original mask is true
            for (var i = 0; i < curTokens.Count; i++)
            {
                var finalMask = curClean[i].Length > 0 && curMask[i];
                rawTokens.Add(curTokens[i]);
                cleanTokens.Add(curClean[i]);
                predictionMask.Add(finalMask);
            }
        }

        return (rawTokens, cleanTokens, predictionMask);
    }

    /// <summary>
    /// Get model predictions for the processable tokens.
    /// Returns arrays indexed by the prediction index (consecutive processable tokens only).
    /// </summary>
    private (float[][] StressProbs, int[] StressPreds, float[][] YoProbs, int[] YoPreds) GetModelPreds(
        List<string> cleanTokens,
        List<bool> predictionMask)
    {
        // Count processable words and collect them
        var processableWords = new List<string>();
        for (var i = 0; i < cleanTokens.Count; i++)
        {
            if (predictionMask[i])
            {
                processableWords.Add(cleanTokens[i]);
            }
        }

        var count = processableWords.Count;
        var stressProbs = new float[count][];
        var stressPreds = new int[count];
        var yoProbs = new float[count][];
        var yoPreds = new int[count];

        for (var i = 0; i < count; i++)
        {
            var word = processableWords[i];
            if (word.Length == 0)
            {
                stressProbs[i] = [1.0f];
                stressPreds[i] = 0;
                yoProbs[i] = [1.0f];
                yoPreds[i] = 0;
                continue;
            }

            // Compute embedding
            var embedding = this.ComputeEmbedding(word);

            // Stress logits and prediction
            var stressLogits = RunMlp(embedding, this.stressClf!);
            stressProbs[i] = Softmax(stressLogits);
            stressPreds[i] = Argmax(stressLogits);

            // Yo logits and prediction
            if (this.yoClf is not null)
            {
                var yoLogits = RunMlp(embedding, this.yoClf);
                yoProbs[i] = Softmax(yoLogits);
                yoPreds[i] = Argmax(yoLogits);
            }
            else
            {
                yoProbs[i] = [1.0f];
                yoPreds[i] = 0;
            }
        }

        return (stressProbs, stressPreds, yoProbs, yoPreds);
    }

    private float[] ComputeEmbedding(string word)
    {
        // Pad with < and > like FastText
        var padded = "<" + word + ">";
        var sum = new float[this.embDim];
        var count = 0;

        // Generate all n-grams from length 1 to len(padded)
        // Python: word_ngrams(word, 1, len(word)+3) generates n-grams of size 1 to len(word)+2
        var maxN = word.Length + 2;
        for (var n = 1; n <= maxN; n++)
        {
            for (var i = 0; i <= padded.Length - n; i++)
            {
                var gram = padded.Substring(i, n);
                if (this.ngramDict!.TryGetValue(gram, out var idx))
                {
                    for (var d = 0; d < this.embDim; d++)
                    {
                        sum[d] += this.embeddingWeight![idx, d];
                    }

                    count++;
                }
            }
        }

        // Mean pooling
        if (count > 0)
        {
            for (var d = 0; d < this.embDim; d++)
            {
                sum[d] /= count;
            }
        }

        return sum;
    }

    private static float[] RunMlp(float[] input, MlpLayer[] layers)
    {
        var current = input;
        foreach (var layer in layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    private static float[] Softmax(float[] logits)
    {
        var max = logits[0];
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        var sum = 0.0f;
        var result = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp(logits[i] - max);
            sum += result[i];
        }

        for (var i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    private static int Argmax(float[] values)
    {
        var maxIdx = 0;
        var maxVal = values[0];
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > maxVal)
            {
                maxVal = values[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    private static string TransferCase(string original, string stressed)
    {
        var sb = new StringBuilder(stressed.Length);
        var origIdx = 0;
        foreach (var ch in stressed)
        {
            if (ch == '+')
            {
                sb.Append('+');
            }
            else
            {
                if (origIdx < original.Length && char.IsUpper(original[origIdx]))
                {
                    sb.Append(char.ToUpperInvariant(ch));
                }
                else
                {
                    sb.Append(ch);
                }

                origIdx++;
            }
        }

        return sb.ToString();
    }

    private static int CountVowels(string word)
    {
        var count = 0;
        foreach (var ch in word)
        {
            if (Vowels.Contains(char.ToLowerInvariant(ch)))
            {
                count++;
            }
        }

        return count;
    }

    /// <summary>
    /// Load the neural accentor from extracted model files.
    /// Expects: ngram_dict.json, stress_embedding.npy, stress_clf_*.npy,
    /// yo_clf_*.npy, and exceptions.json.
    /// </summary>
    /// <param name="directory">Directory containing the accentor model files.</param>
    public static StressAccentor LoadFromDirectory(string directory)
    {
        // Load n-gram dictionary
        var dictPath = Path.Combine(directory, "ngram_dict.json");
        var dictJson = File.ReadAllText(dictPath, Encoding.UTF8);
        var ngramDict = JsonSerializer.Deserialize<Dictionary<string, int>>(dictJson)
            ?? throw new InvalidOperationException("Failed to load ngram_dict.json");

        // Load embedding weight [vocabSize, embDim]
        var embWeight = NpyReader.LoadFloat2D(Path.Combine(directory, "stress_embedding.npy"));

        // Load stress classifier MLP layers: Linear(16->64)+ReLU, Linear(64->128)+ReLU,
        // Linear(128->64)+ReLU, Linear(64->10)
        var stressClf = new MlpLayer[]
        {
            MlpLayer.LoadLinearRelu(directory, "stress_clf_0_weight", "stress_clf_0_bias"),
            MlpLayer.LoadLinearRelu(directory, "stress_clf_2_weight", "stress_clf_2_bias"),
            MlpLayer.LoadLinearRelu(directory, "stress_clf_4_weight", "stress_clf_4_bias"),
            MlpLayer.LoadLinear(directory, "stress_clf_6_weight", "stress_clf_6_bias"),
        };

        // Load yo classifier MLP layers: Linear(16->64)+ReLU, Linear(64->128)+ReLU,
        // Linear(128->64)+ReLU, Linear(64->7)
        MlpLayer[]? yoClf = null;
        var yoClfWeightPath = Path.Combine(directory, "yo_clf_0_weight.npy");
        if (File.Exists(yoClfWeightPath))
        {
            yoClf =
            [
                MlpLayer.LoadLinearRelu(directory, "yo_clf_0_weight", "yo_clf_0_bias"),
                MlpLayer.LoadLinearRelu(directory, "yo_clf_2_weight", "yo_clf_2_bias"),
                MlpLayer.LoadLinearRelu(directory, "yo_clf_4_weight", "yo_clf_4_bias"),
                MlpLayer.LoadLinear(directory, "yo_clf_6_weight", "yo_clf_6_bias"),
            ];
        }

        // Load exceptions dictionary: { "word": [stress_char_pos, yo_char_pos] }
        Dictionary<string, (int Stress, int Yo)>? exceptions = null;
        var exceptionsPath = Path.Combine(directory, "exceptions.json");
        if (File.Exists(exceptionsPath))
        {
            var excJson = File.ReadAllText(exceptionsPath, Encoding.UTF8);
            var rawExc = JsonSerializer.Deserialize<Dictionary<string, int[]>>(excJson);
            if (rawExc is not null)
            {
                exceptions = new Dictionary<string, (int, int)>(rawExc.Count);
                foreach (var (key, value) in rawExc)
                {
                    if (value.Length >= 2)
                    {
                        exceptions[key] = (value[0], value[1]);
                    }
                }
            }
        }

        return new StressAccentor(ngramDict, embWeight, stressClf, yoClf, exceptions);
    }

    /// <summary>
    /// Load stress dictionary from a JSON file (fallback, less accurate).
    /// </summary>
    public static StressAccentor LoadFromJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath, Encoding.UTF8);
        var dict = JsonSerializer.Deserialize<Dictionary<string, string>>(json)
            ?? throw new InvalidOperationException("Failed to parse stress dictionary");

        return new StressAccentor(null, null, null, null, null, dict);
    }

    /// <summary>
    /// Create a no-op accentor that passes text through unchanged.
    /// </summary>
    public static StressAccentor CreatePassthrough() => new(null, null, null, null, null);

    public void Dispose() { }

    /// <summary>
    /// A single MLP layer: Linear + optional ReLU.
    /// Weight shape is [outDim, inDim], bias shape is [outDim].
    /// </summary>
    internal sealed class MlpLayer
    {
        private readonly float[,] weight; // [outDim, inDim]
        private readonly float[] bias;    // [outDim]
        private readonly bool relu;

        private MlpLayer(float[,] weight, float[] bias, bool relu)
        {
            this.weight = weight;
            this.bias = bias;
            this.relu = relu;
        }

        public float[] Forward(float[] input)
        {
            var outDim = this.weight.GetLength(0);
            var inDim = this.weight.GetLength(1);
            var output = new float[outDim];

            for (var o = 0; o < outDim; o++)
            {
                var sum = this.bias[o];
                for (var i = 0; i < inDim; i++)
                {
                    sum += this.weight[o, i] * input[i];
                }

                output[o] = this.relu ? Math.Max(0, sum) : sum;
            }

            return output;
        }

        public static MlpLayer LoadLinearRelu(string dir, string weightName, string biasName)
        {
            var w = NpyReader.LoadFloat2D(Path.Combine(dir, weightName + ".npy"));
            var b = NpyReader.LoadFloat1D(Path.Combine(dir, biasName + ".npy"));
            return new MlpLayer(w, b, relu: true);
        }

        public static MlpLayer LoadLinear(string dir, string weightName, string biasName)
        {
            var w = NpyReader.LoadFloat2D(Path.Combine(dir, weightName + ".npy"));
            var b = NpyReader.LoadFloat1D(Path.Combine(dir, biasName + ".npy"));
            return new MlpLayer(w, b, relu: false);
        }
    }
}
