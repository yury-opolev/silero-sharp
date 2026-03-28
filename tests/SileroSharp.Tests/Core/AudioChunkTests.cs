using Xunit;

namespace SileroSharp.Tests.Core;

public class AudioChunkTests
{
    [Fact]
    public void Duration_CalculatesCorrectly()
    {
        var chunk = new AudioChunk
        {
            Samples = new float[48000],
            SampleRate = 48000,
        };

        Assert.Equal(TimeSpan.FromSeconds(1), chunk.Duration);
    }

    [Fact]
    public void Duration_EmptySamples_ReturnsZero()
    {
        var chunk = new AudioChunk
        {
            Samples = [],
            SampleRate = 48000,
        };

        Assert.Equal(TimeSpan.Zero, chunk.Duration);
    }

    [Fact]
    public void Duration_ZeroSampleRate_ReturnsZero()
    {
        var chunk = new AudioChunk
        {
            Samples = new float[100],
            SampleRate = 0,
        };

        Assert.Equal(TimeSpan.Zero, chunk.Duration);
    }

    [Fact]
    public void ToPcm16Bytes_ConvertsCorrectly()
    {
        var chunk = new AudioChunk
        {
            Samples = [1.0f, -1.0f, 0.0f],
            SampleRate = 48000,
        };

        var bytes = chunk.ToPcm16Bytes();

        Assert.Equal(6, bytes.Length); // 3 samples * 2 bytes each

        // 1.0f -> 32767 -> 0xFF7F (little-endian)
        var sample0 = BitConverter.ToInt16(bytes, 0);
        Assert.Equal(32767, sample0);

        // -1.0f -> -32767 -> 0x0180
        var sample1 = BitConverter.ToInt16(bytes, 2);
        Assert.Equal(-32767, sample1);

        // 0.0f -> 0
        var sample2 = BitConverter.ToInt16(bytes, 4);
        Assert.Equal(0, sample2);
    }

    [Fact]
    public void ToFloat32Bytes_RoundTrips()
    {
        var original = new float[] { 0.5f, -0.3f, 0.0f, 1.0f };
        var chunk = new AudioChunk
        {
            Samples = original,
            SampleRate = 48000,
        };

        var bytes = chunk.ToFloat32Bytes();
        Assert.Equal(original.Length * sizeof(float), bytes.Length);

        // Verify round-trip
        var restored = new float[original.Length];
        Buffer.BlockCopy(bytes, 0, restored, 0, bytes.Length);

        for (var i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], restored[i]);
        }
    }
}
