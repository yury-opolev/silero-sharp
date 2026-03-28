using Microsoft.Extensions.Logging;
using TorchSharp;
using static TorchSharp.torch;

namespace SileroSharp;

/// <summary>
/// TorchSharp wrapper for the Silero TTS v5 JIT model.
/// Loads the original TorchScript model directly — bit-exact output.
/// </summary>
public sealed partial class SileroModel : IDisposable
{
    private readonly jit.ScriptModule _model;
    private readonly ILogger<SileroModel> _logger;
    private readonly Lock _lock = new();
    private bool _disposed;

    public SileroModel(string modelPath, ILogger<SileroModel>? logger = null)
    {
        _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<SileroModel>.Instance;

        _model = jit.load(modelPath);
        _model.eval();

        LogModelLoaded(modelPath);
    }

    /// <summary>
    /// Run TTS inference: convert token IDs to audio samples.
    /// Thread-safe — calls are serialized internally.
    /// </summary>
    /// <param name="tokens">Integer token IDs (with SOS/EOS) representing the input text.</param>
    /// <param name="speakerId">Speaker index (0=aidar, 1=baya, 2=kseniya, 3=xenia).</param>
    /// <returns>Float32 audio samples in the range [-1.0, 1.0] at 48kHz.</returns>
    public float[] Infer(long[] tokens, int speakerId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (tokens.Length == 0)
            return [];

        lock (_lock)
        {
            using var noGrad = no_grad();

            // Build input tensors
            using var tokensTensor = tensor(tokens, dtype: ScalarType.Int64).unsqueeze(0); // [1, N]
            using var speakerTensor = tensor(new long[] { speakerId }, dtype: ScalarType.Int64); // [1]

            // Run inference — returns (audio[1,T], durations[1,N]) tuple
            var result = _model.forward(tokensTensor, speakerTensor);

            // The JIT model returns a tuple. TorchSharp wraps it as object.
            // Try to extract the audio tensor (first element of the tuple).
            Tensor audioTensor;
            switch (result)
            {
                case Tensor t:
                    audioTensor = t;
                    break;
                case (Tensor audio, Tensor):
                    audioTensor = audio;
                    break;
                default:
                    // Generic tuple handling
                    var tuple = (System.Runtime.CompilerServices.ITuple)result!;
                    audioTensor = (Tensor)tuple[0]!;
                    break;
            }

            // Convert to float array: audio is [1, T]
            using var flat = audioTensor.squeeze(0);
            return flat.data<float>().ToArray();
        }
    }

    [LoggerMessage(Level = LogLevel.Information,
        Message = "Silero TorchSharp model loaded from {ModelPath}")]
    private partial void LogModelLoaded(string modelPath);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _model.Dispose();
    }
}
