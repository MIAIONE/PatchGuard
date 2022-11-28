global using Microsoft.ML;
global using Microsoft.ML.Data;
global using SixLabors.ImageSharp;
namespace PatchGuardService;

internal class Recognizer
{
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<InputFormat, OutputFormat> _predictionEngine;
    private const string ONNX_MODEL_PATH = @".\common_old.onnx";
    public Recognizer()
    {
        _mlContext = new();
        ITransformer predictionPipeline = GetPredictionPipeline(_mlContext);
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<InputFormat, OutputFormat>(predictionPipeline);
    }
    public long Predict(Image img)
    {
        img.Mutate(img => { img.Grayscale(); });
        int size = img.Height * img.Width;
        var rgb = new Rgb24[size];
        img.CopyPixelDataTo(rgb);
        List<float> gray = new List<Rgb24>(rgb).Select(x => { return (float)((x.B / 255) - 0.5) * 2; }).ToList();
        //TODO PRE PROCESSING
        return 0;
    }
    private OutputFormat Forward(float[] x) // TODO: FIX INPUT
    {
        var input = new InputFormat()
        {
            InputImage = x
        };
        return _predictionEngine.Predict(input);
    }
    private static ITransformer GetPredictionPipeline(MLContext mlContext)
    {
        var inputColumns = new string[] { "input1" };
        var outputColumns = new string[] { "output" };
        var onnxPredictionPipeline = mlContext
           .Transforms
           .ApplyOnnxModel(
                outputColumns,
                inputColumns,
                ONNX_MODEL_PATH
            );
        var emptyInp = mlContext.Data.LoadFromEnumerable(Array.Empty<InputFormat>());
        return onnxPredictionPipeline.Fit(emptyInp);
    }
    private class InputFormat
    {
        [VectorType(1, 64, 128 )] // TO FIX INPUT
        [ColumnName("input1")]
        public float[] InputImage { get; set; }
    }
    private class OutputFormat
    {
        [ColumnName("output")] //TO FIX OUTPUT
        public float[] OutPutImage { get; set; }
    }
}
