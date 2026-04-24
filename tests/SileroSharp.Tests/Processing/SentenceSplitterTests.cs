using SileroSharp.Processing;
using Xunit;

namespace SileroSharp.Tests.Processing;

public class SentenceSplitterTests
{
    [Fact]
    public void Split_SimpleSentences_SplitsOnPeriod()
    {
        var result = SentenceSplitter.Split("Первое предложение. Второе предложение.");
        Assert.Equal(2, result.Count);
        Assert.Equal("Первое предложение.", result[0]);
        Assert.Equal("Второе предложение.", result[1]);
    }

    [Fact]
    public void Split_ExclamationAndQuestion_SplitsCorrectly()
    {
        var result = SentenceSplitter.Split("Привет! Как дела? Хорошо.");
        Assert.Equal(3, result.Count);
        Assert.Equal("Привет!", result[0]);
        Assert.Equal("Как дела?", result[1]);
        Assert.Equal("Хорошо.", result[2]);
    }

    [Fact]
    public void Split_Abbreviations_DoesNotSplit()
    {
        var result = SentenceSplitter.Split("Т.е. это важно. Другое предложение.");
        Assert.Equal(2, result.Count);
        Assert.StartsWith("Т.е.", result[0]);
    }

    [Fact]
    public void Split_EmptyInput_ReturnsEmpty()
    {
        Assert.Empty(SentenceSplitter.Split(""));
        Assert.Empty(SentenceSplitter.Split("  "));
    }

    [Fact]
    public void Split_SingleSentence_ReturnsOne()
    {
        var result = SentenceSplitter.Split("Одно предложение.");
        Assert.Single(result);
        Assert.Equal("Одно предложение.", result[0]);
    }

    [Fact]
    public void Split_NoTerminalPunctuation_AddsPeriod()
    {
        var result = SentenceSplitter.Split("Без точки");
        Assert.Single(result);
        Assert.EndsWith(".", result[0]);
    }

    [Fact]
    public void Split_LongSentence_SplitsAtSecondaryBoundaries()
    {
        // Create a sentence longer than 130 chars with commas
        var longText = "Это очень длинное предложение, которое содержит много слов, " +
                       "и оно должно быть разделено на части, потому что оно превышает " +
                       "максимальную длину в сто тридцать символов.";

        var result = SentenceSplitter.Split(longText, maxLength: 80);

        Assert.True(result.Count > 1, "Long sentence should be split into multiple chunks");
        foreach (var chunk in result)
        {
            Assert.True(chunk.Length <= 80 || !chunk.Contains(','),
                $"Chunk exceeds max length: '{chunk}' ({chunk.Length} chars)");
        }
    }

    [Fact]
    public void Split_AsteriskMarkers_PreservedIntact()
    {
        var result = SentenceSplitter.Split("Это *важный* вопрос. Да.");
        Assert.Contains(result, s => s.Contains("*важный*"));
    }

    [Fact]
    public void Split_MultipleWhitespace_Normalized()
    {
        var result = SentenceSplitter.Split("Слово   слово.  Другое   предложение.");
        Assert.Equal(2, result.Count);
        Assert.DoesNotContain("  ", result[0]);
    }

    [Fact]
    public void SplitLongSentence_CommaSplit_FullExpectedFragments()
    {
        // One real sentence, longer than maxLength, with comma boundaries.
        // Each non-final fragment must end with ',' (natural mid-sentence pause)
        // and the final fragment keeps the original terminator '.'.
        // Bug today: EnsureTerminalPunctuation replaced trailing commas with '.',
        // causing Silero to apply falling sentence-final intonation at every comma.
        var longText = "Это очень длинное предложение, которое содержит много слов, " +
                       "и оно должно быть разделено на части, потому что оно превышает " +
                       "максимальную длину в восемьдесят символов.";

        var result = SentenceSplitter.Split(longText, maxLength: 80);

        Assert.Equal(
            [
                "Это очень длинное предложение, которое содержит много слов,",
                "и оно должно быть разделено на части,",
                "потому что оно превышает максимальную длину в восемьдесят символов.",
            ],
            result);
    }

    [Fact]
    public void SplitLongSentence_QuestionTerminator_FullExpectedFragments()
    {
        // The original sentence ends with '?'. The last fragment must keep '?',
        // not have it replaced with '.'. Non-final fragments end with ','.
        var longQuestion = "Это очень длинное вопросительное предложение, которое содержит " +
                           "много слов, и оно должно быть разделено на части, потому что " +
                           "оно превышает максимальную длину?";

        var result = SentenceSplitter.Split(longQuestion, maxLength: 80);

        Assert.Equal(
            [
                "Это очень длинное вопросительное предложение, которое содержит много слов,",
                "и оно должно быть разделено на части,",
                "потому что оно превышает максимальную длину?",
            ],
            result);
    }

    [Fact]
    public void SplitLongSentence_ExclamationTerminator_FullExpectedFragments()
    {
        var longExclamation = "Это очень длинное восклицательное предложение, которое содержит " +
                              "много слов, и оно должно быть разделено на части, потому что " +
                              "оно превышает максимальную длину!";

        var result = SentenceSplitter.Split(longExclamation, maxLength: 80);

        Assert.Equal(
            [
                "Это очень длинное восклицательное предложение, которое содержит много слов,",
                "и оно должно быть разделено на части,",
                "потому что оно превышает максимальную длину!",
            ],
            result);
    }

    [Fact]
    public void SplitLongSentence_ForceSplit_FullExpectedFragments()
    {
        // Sentence with no commas/semicolons/dashes inside the maxLength window.
        // Force-split fragments should suggest "continuing" prosody —
        // append ',' rather than fake-terminal '.' on each non-final fragment.
        // The final fragment keeps the original sentence terminator '.'.
        var longText = "Это очень длинное предложение без запятых и других разделителей которое " +
                       "должно быть принудительно разделено на несколько фрагментов потому что оно " +
                       "превышает максимальную длину.";

        var result = SentenceSplitter.Split(longText, maxLength: 60);

        Assert.Equal(
            [
                "Это очень длинное предложение без запятых и других,",
                "разделителей которое должно быть принудительно разделено на,",
                "несколько фрагментов потому что оно превышает максимальную,",
                "длину.",
            ],
            result);
    }

    [Fact]
    public void Split_MultipleSentencesOneLong_OnlyLongOneIsSubdivided()
    {
        // Two short sentences and one long sentence in between.
        // Sentence boundary detection runs first, then length-based splitting
        // applies only to the long one. Short sentences pass through unchanged.
        var input = "Короткое первое. " +
                    "Это очень длинное предложение, которое содержит много слов, " +
                    "и оно должно быть разделено на части, потому что оно превышает " +
                    "максимальную длину! " +
                    "Короткое последнее.";

        var result = SentenceSplitter.Split(input, maxLength: 80);

        Assert.Equal(
            [
                "Короткое первое.",
                "Это очень длинное предложение, которое содержит много слов,",
                "и оно должно быть разделено на части,",
                "потому что оно превышает максимальную длину!",
                "Короткое последнее.",
            ],
            result);
    }
}
