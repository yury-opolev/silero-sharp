using SileroSharp.Processing;
using Xunit;

namespace SileroSharp.Tests.Processing;

public class RussianTextProcessorTests
{
    [Fact]
    public void Process_MultiSentenceCasedInput_SplitsAndLowercasesEachSentence()
    {
        // Multi-sentence Russian input over the 130-char default chunk length.
        // Each underlying sentence is short enough to remain a single fragment.
        // Verifies Option A: sentence boundaries are detected on the cased text,
        // then each sentence is lowercased independently.
        // Bug today: ToLowerInvariant() ran before SentenceSplitter, defeating its
        // "next char is uppercase" boundary heuristic.
        var processor = RussianTextProcessor.CreateDefault();
        var input = "Это первое предложение для теста интонации. " +
                    "Это второе предложение, которое не слишком длинное! " +
                    "Это третье предложение, и оно тоже довольно короткое?";

        Assert.True(input.Length > 130, "Test input must exceed default MaxChunkLength to exercise the splitter.");

        var result = processor.Process(input);

        var actualTexts = result.Sentences.Select(s => s.Text).ToList();
        Assert.Equal(
            [
                "это первое предложение для теста интонации.",
                "это второе предложение, которое не слишком длинное!",
                "это третье предложение, и оно тоже довольно короткое?",
            ],
            actualTexts);
    }

    [Fact]
    public void Process_ShortSingleSentence_PassesThroughLowercased()
    {
        // Regression guard for the short-circuit at length <= MaxChunkLength.
        // Text is not split but is still lowercased.
        var processor = RussianTextProcessor.CreateDefault();

        var result = processor.Process("Короткое Предложение.");

        var actualTexts = result.Sentences.Select(s => s.Text).ToList();
        Assert.Equal(["короткое предложение."], actualTexts);
    }

    [Fact]
    public void Process_SingleSentenceOver130Chars_SplitsAtCommasWithOriginalTerminator()
    {
        // One real sentence longer than the 130-char limit.
        // After Option A (split before lowercasing) + Option D (preserve original
        // mid-sentence punctuation), the sentence splits at commas, intermediate
        // fragments end with ',' and the last fragment keeps '?'. All lowercased.
        var processor = RussianTextProcessor.CreateDefault();
        var input = "Это очень длинное вопросительное предложение, которое содержит много " +
                    "слов и запятых, и оно должно быть разделено на несколько частей, потому " +
                    "что превышает максимальную длину чанка?";

        Assert.True(input.Length > 130);

        var result = processor.Process(input);

        var actualTexts = result.Sentences.Select(s => s.Text).ToList();
        Assert.Equal(
            [
                "это очень длинное вопросительное предложение, которое содержит много слов и запятых,",
                "и оно должно быть разделено на несколько частей, потому что превышает максимальную длину чанка?",
            ],
            actualTexts);
    }

    [Fact]
    public void Process_MultiSentenceWithOneLongSentence_OnlyLongOneIsSubdivided()
    {
        // Three real sentences where the middle one exceeds MaxChunkLength.
        // Sentence-boundary detection runs first, then length-based splitting
        // applies only to the long sentence. Short sentences pass through whole.
        var processor = RussianTextProcessor.CreateDefault();
        var input = "Короткое первое. " +
                    "Это очень длинное второе предложение, которое содержит много слов, " +
                    "и оно должно быть разделено на части, потому что оно превышает " +
                    "максимальную длину чанка! " +
                    "Короткое третье.";

        Assert.True(input.Length > 130);

        var result = processor.Process(input);

        var actualTexts = result.Sentences.Select(s => s.Text).ToList();
        Assert.Equal(
            [
                "короткое первое.",
                "это очень длинное второе предложение, которое содержит много слов, и оно должно быть разделено на части,",
                "потому что оно превышает максимальную длину чанка!",
                "короткое третье.",
            ],
            actualTexts);
    }

    [Fact]
    public void Process_EmptyOrWhitespace_ReturnsEmpty()
    {
        var processor = RussianTextProcessor.CreateDefault();

        Assert.Empty(processor.Process("").Sentences);
        Assert.Empty(processor.Process("   ").Sentences);
    }
}
