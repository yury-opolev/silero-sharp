using SileroSharp.Processing;
using Xunit;

namespace SileroSharp.Tests.Processing;

public class RussianTextProcessorTests
{
    [Fact]
    public void Process_MultiSentenceCasedInput_SplitsAtSentenceBoundaries()
    {
        // Multi-sentence Russian input over the 130-char default chunk length.
        // Each underlying sentence is short enough to remain a single fragment.
        // Bug today: ToLowerInvariant() runs before SentenceSplitter, so the splitter's
        // "next char is uppercase" heuristic never triggers and the whole paragraph collapses
        // into a single chunk that then gets comma-split instead of sentence-split.
        var processor = RussianTextProcessor.CreateDefault();
        var input = "Это первое предложение для теста интонации. " +
                    "Это второе предложение, которое не слишком длинное! " +
                    "Это третье предложение, и оно тоже довольно короткое?";

        Assert.True(input.Length > 130, "Test input must exceed default MaxChunkLength to exercise the splitter.");

        var result = processor.Process(input);

        Assert.Equal(3, result.Sentences.Count);
        Assert.EndsWith(".", result.Sentences[0].Text);
        Assert.EndsWith("!", result.Sentences[1].Text);
        Assert.EndsWith("?", result.Sentences[2].Text);
    }

    [Fact]
    public void Process_ShortSingleSentence_StaysSingle()
    {
        // Regression guard for the short-circuit at length <= MaxChunkLength.
        var processor = RussianTextProcessor.CreateDefault();

        var result = processor.Process("Короткое предложение.");

        Assert.Single(result.Sentences);
    }

    [Fact]
    public void Process_SingleSentenceOver130Chars_SplitsAtCommasNotPeriods()
    {
        // One real sentence longer than the 130-char limit.
        // After Option A (split before lowercasing) + Option D (preserve original
        // mid-sentence punctuation), non-final fragments should end with ','
        // and the last fragment should keep the original terminator '?'.
        var processor = RussianTextProcessor.CreateDefault();
        var input = "Это очень длинное вопросительное предложение, которое содержит много " +
                    "слов и запятых, и оно должно быть разделено на несколько частей, потому " +
                    "что превышает максимальную длину чанка?";

        Assert.True(input.Length > 130);

        var result = processor.Process(input);

        Assert.True(result.Sentences.Count > 1, "Long sentence must be split into multiple fragments.");
        for (var i = 0; i < result.Sentences.Count - 1; i++)
        {
            Assert.EndsWith(",", result.Sentences[i].Text);
        }
        Assert.EndsWith("?", result.Sentences[^1].Text);
    }

    [Fact]
    public void Process_LowercasesSentencesAfterSplitting()
    {
        // Verify Option A: sentence boundaries are detected on cased input,
        // then each sentence is lowercased independently.
        var processor = RussianTextProcessor.CreateDefault();
        var input = "Это первое предложение для теста интонации. " +
                    "Это второе предложение, которое не слишком длинное! " +
                    "Это третье предложение, и оно тоже довольно короткое?";

        var result = processor.Process(input);

        foreach (var sentence in result.Sentences)
        {
            Assert.Equal(sentence.Text, sentence.Text.ToLowerInvariant());
        }
    }
}
