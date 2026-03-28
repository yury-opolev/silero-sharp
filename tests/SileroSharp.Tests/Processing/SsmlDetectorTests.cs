using SileroSharp.Processing;
using Xunit;

namespace SileroSharp.Tests.Processing;

public class SsmlDetectorTests
{
    [Theory]
    [InlineData("<speak>Привет</speak>", true)]
    [InlineData("<speak version=\"1.0\">Текст</speak>", true)]
    [InlineData("<SPEAK>Текст</SPEAK>", true)]
    [InlineData("  <speak>с пробелами</speak>", true)]
    [InlineData("\t<speak>с табом</speak>", true)]
    [InlineData("Обычный текст", false)]
    [InlineData("", false)]
    [InlineData("<speaker>не SSML</speaker>", false)]
    [InlineData("<speaking>не SSML</speaking>", false)]
    [InlineData("Текст с <speak> внутри", false)]
    public void IsSsml_DetectsCorrectly(string input, bool expected)
    {
        Assert.Equal(expected, SsmlDetector.IsSsml(input));
    }
}
