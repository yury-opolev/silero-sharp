using NAudio.Wave;

namespace SileroSharp.NAudio;

/// <summary>
/// Audio sink that plays audio through NAudio's WaveOutEvent.
/// Windows-only. Uses IEEE float32 format to avoid int16 conversion.
/// </summary>
public sealed class NAudioWaveOutSink : IAudioSink, IDisposable
{
    private WaveOutEvent? _waveOut;
    private BufferedWaveProvider? _buffer;
    private bool _initialized;
    private bool _disposed;

    public async Task InitializeAsync(int sampleRate, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_initialized)
            return;

        var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels: 1);
        _buffer = new BufferedWaveProvider(format)
        {
            BufferDuration = TimeSpan.FromSeconds(10),
            DiscardOnBufferOverflow = false,
        };

        _waveOut = new WaveOutEvent();
        _waveOut.Init(_buffer);
        _waveOut.Play();
        _initialized = true;

        await Task.CompletedTask.ConfigureAwait(false);
    }

    public Task WriteAsync(AudioChunk chunk, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (!_initialized || _buffer is null)
            throw new InvalidOperationException("Sink not initialized. Call InitializeAsync first.");

        var bytes = chunk.ToFloat32Bytes();
        _buffer.AddSamples(bytes, 0, bytes.Length);

        return Task.CompletedTask;
    }

    public Task FlushAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    public async Task DrainAsync(CancellationToken cancellationToken = default)
    {
        if (_buffer is null)
            return;

        // Wait until the buffer has been consumed by WaveOut
        while (_buffer.BufferedDuration > TimeSpan.FromMilliseconds(50))
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Delay(50, cancellationToken).ConfigureAwait(false);
        }
    }

    public void StopImmediately()
    {
        _waveOut?.Stop();
        _buffer?.ClearBuffer();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _waveOut?.Stop();
        _waveOut?.Dispose();
    }

    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}
