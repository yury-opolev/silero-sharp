namespace SileroSharp;

/// <summary>
/// Abstraction for consuming synthesized audio chunks.
/// Implementations handle the actual audio output (playback, file writing, etc.).
/// </summary>
public interface IAudioSink : IAsyncDisposable
{
    /// <summary>
    /// Initialize the sink for audio at the given sample rate.
    /// Called once before the first <see cref="WriteAsync"/> call.
    /// </summary>
    Task InitializeAsync(int sampleRate, CancellationToken cancellationToken = default);

    /// <summary>
    /// Write an audio chunk to the sink.
    /// </summary>
    Task WriteAsync(AudioChunk chunk, CancellationToken cancellationToken = default);

    /// <summary>
    /// Signal that all audio has been written. The sink should finish
    /// playing/writing any buffered audio.
    /// </summary>
    Task FlushAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Wait until all queued audio has finished playing/writing.
    /// </summary>
    Task DrainAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Immediately stop all output, discarding any buffered audio.
    /// </summary>
    void StopImmediately();
}
