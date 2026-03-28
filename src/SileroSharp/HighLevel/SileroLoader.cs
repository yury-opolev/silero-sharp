using Microsoft.Extensions.Logging;
using SileroSharp.Processing;

namespace SileroSharp;

/// <summary>
/// Loads and initializes the Silero TTS model and associated resources.
/// </summary>
public static class SileroLoader
{
    /// <summary>
    /// Load a SileroTTS instance, auto-discovering resource files next to the model.
    /// Expects: silero_v5_jit.pt + accentor/ directory with model assets.
    /// </summary>
    public static SileroTTS LoadWithAutoDiscovery(
        string modelPath,
        SileroOptions? options = null,
        ILogger<SileroTTS>? logger = null,
        ILogger<SileroModel>? modelLogger = null)
    {
        options ??= new SileroOptions();
        var dir = Path.GetDirectoryName(modelPath);
        if (string.IsNullOrEmpty(dir))
        {
            dir = ".";
        }

        // TTS model
        var model = new SileroModel(modelPath, modelLogger);

        // Symbol table
        var symbolsPath = Path.Combine(dir, "symbols.json");
        var symbolTable = File.Exists(symbolsPath)
            ? SymbolTable.LoadFromJson(symbolsPath)
            : SymbolTable.CreateForVariant(options.Variant);

        // Accentor directory (contains neural stress model + exceptions + homosolver)
        var accentorDir = Path.Combine(dir, "accentor");

        HomoSolver homoSolver;
        StressAccentor accentor;

        if (Directory.Exists(accentorDir))
        {
            homoSolver = HomoSolver.LoadFromDirectory(accentorDir);
            accentor = StressAccentor.LoadFromDirectory(accentorDir);
        }
        else
        {
            homoSolver = HomoSolver.CreatePassthrough();
            var stressMapPath = Path.Combine(dir, "stress_map.json");
            accentor = File.Exists(stressMapPath)
                ? StressAccentor.LoadFromJson(stressMapPath)
                : StressAccentor.CreatePassthrough();
        }

        // Yo restorer
        var yoPath = Path.Combine(dir, "yo_dict.json");
        var yoRestorer = File.Exists(yoPath)
            ? YoRestorer.LoadFromJson(yoPath)
            : YoRestorer.CreatePassthrough();

        var textProcessor = new RussianTextProcessor(symbolTable, homoSolver, accentor, yoRestorer, options);

        return new SileroTTS(model, textProcessor, options, logger, ownsModel: true);
    }
}
