using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using SileroSharp.Processing;

namespace SileroSharp;

/// <summary>
/// High-level Silero TTS client providing streaming and single-shot speech synthesis.
/// </summary>
public sealed partial class SileroTTS : IAsyncDisposable
{
    private readonly SileroModel _model;
    private readonly RussianTextProcessor _textProcessor;
    private readonly SileroOptions _options;
    private readonly ILogger<SileroTTS> _logger;
    private readonly bool _ownsModel;
    private bool _disposed;

    /// <summary>
    /// Create a SileroTTS instance from a pre-loaded model.
    /// </summary>
    public SileroTTS(
        SileroModel model,
        SileroOptions? options = null,
        ILogger<SileroTTS>? logger = null,
        bool ownsModel = true)
        : this(model, null, options, logger, ownsModel)
    {
    }

    internal SileroTTS(
        SileroModel model,
        RussianTextProcessor? textProcessor,
        SileroOptions? options,
        ILogger<SileroTTS>? logger,
        bool ownsModel)
    {
        _model = model;
        _options = options ?? new SileroOptions();
        _textProcessor = textProcessor ?? RussianTextProcessor.CreateDefault(_options);
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<SileroTTS>.Instance;
        _ownsModel = ownsModel;
    }

    /// <summary>
    /// Synthesize the entire text and return a single audio chunk.
    /// For long text, this concatenates all sentence chunks into one.
    /// </summary>
    public async Task<AudioChunk> SynthesizeAsync(
        string text,
        SileroVoice voice,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var allSamples = new List<float>();
        var lastSampleRate = _options.SampleRate;

        await foreach (var chunk in SynthesizeStreamingAsync(text, voice, cancellationToken).ConfigureAwait(false))
        {
            allSamples.AddRange(chunk.Samples);
            lastSampleRate = chunk.SampleRate;
        }

        return new AudioChunk
        {
            Samples = allSamples.ToArray(),
            SampleRate = lastSampleRate,
            SentenceIndex = 0,
            SourceText = text,
            IsLast = true,
        };
    }

    /// <summary>
    /// Stream synthesized audio chunks sentence by sentence.
    /// Each chunk corresponds to one sentence from the input text.
    /// Chunks are yielded as soon as inference completes for each sentence.
    /// </summary>
    public async IAsyncEnumerable<AudioChunk> SynthesizeStreamingAsync(
        string text,
        SileroVoice voice,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var processed = _textProcessor.Process(text);

        if (processed.IsSsml)
        {
            LogSsmlDetected();
        }

        LogSynthesisStarted(text.Length, voice.Name, processed.Sentences.Count);

        for (var i = 0; i < processed.Sentences.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var sentence = processed.Sentences[i];
            var isLast = i == processed.Sentences.Count - 1;

            LogSentenceInference(i, sentence.Text.Length, sentence.Tokens.Length);

            // Run inference on a thread pool thread to avoid blocking the caller
            var samples = await Task.Run(
                () => _model.Infer(sentence.Tokens, voice.SpeakerId),
                cancellationToken).ConfigureAwait(false);

            var chunk = new AudioChunk
            {
                Samples = samples,
                SampleRate = _options.SampleRate,
                SentenceIndex = i,
                SourceText = sentence.Text,
                IsLast = isLast,
            };

            LogChunkProduced(i, samples.Length, chunk.Duration.TotalSeconds);

            yield return chunk;
        }
    }

    /// <summary>
    /// Stream audio directly to a sink with a producer/consumer pipeline.
    /// Uses a bounded channel (capacity 3) to overlap inference with playback.
    /// </summary>
    public async Task SpeakToSinkAsync(
        string text,
        SileroVoice voice,
        IAudioSink sink,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        await sink.InitializeAsync(_options.SampleRate, cancellationToken).ConfigureAwait(false);

        var channel = Channel.CreateBounded<AudioChunk>(new BoundedChannelOptions(3)
        {
            SingleWriter = true,
            SingleReader = true,
            FullMode = BoundedChannelFullMode.Wait,
        });

        // Producer: synthesize sentences and write to channel
        var producerTask = Task.Run(async () =>
        {
            try
            {
                await foreach (var chunk in SynthesizeStreamingAsync(text, voice, cancellationToken).ConfigureAwait(false))
                {
                    await channel.Writer.WriteAsync(chunk, cancellationToken).ConfigureAwait(false);
                }
            }
            finally
            {
                channel.Writer.TryComplete();
            }
        }, cancellationToken);

        // Consumer: read from channel and write to sink
        var consumerTask = Task.Run(async () =>
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                await sink.WriteAsync(chunk, cancellationToken).ConfigureAwait(false);
            }

            await sink.FlushAsync(cancellationToken).ConfigureAwait(false);
            await sink.DrainAsync(cancellationToken).ConfigureAwait(false);
        }, cancellationToken);

        try
        {
            await Task.WhenAll(producerTask, consumerTask).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            sink.StopImmediately();
            throw;
        }
    }

    [LoggerMessage(Level = LogLevel.Information, Message = "SSML input detected, passing through to server")]
    private partial void LogSsmlDetected();

    [LoggerMessage(Level = LogLevel.Information,
        Message = "Synthesis started: {TextLength} chars, speaker={Speaker}, sentences={SentenceCount}")]
    private partial void LogSynthesisStarted(int textLength, string speaker, int sentenceCount);

    [LoggerMessage(Level = LogLevel.Debug,
        Message = "Inferring sentence {Index}: {TextLength} chars, {TokenCount} tokens")]
    private partial void LogSentenceInference(int index, int textLength, int tokenCount);

    [LoggerMessage(Level = LogLevel.Debug,
        Message = "Chunk produced: sentence {Index}, {SampleCount} samples, {DurationSeconds:F3}s")]
    private partial void LogChunkProduced(int index, int sampleCount, double durationSeconds);

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        if (_ownsModel)
        {
            _model.Dispose();
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }
}
