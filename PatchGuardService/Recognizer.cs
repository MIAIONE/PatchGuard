﻿global using Microsoft.ML;
global using Microsoft.ML.Data;
global using SixLabors.ImageSharp;
global using SixLabors.ImageSharp.PixelFormats;
global using SixLabors.ImageSharp.Processing;
global using SixLabors.ImageSharp.Advanced;
global using SixLabors.ImageSharp.Memory;
using SixLabors.ImageSharp.ColorSpaces;

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
    public long Predict(Image<Argb32> img)
    {
        img.Mutate(img_ => {
            img_.Resize((img.Height/img.Width)*64,64);
            img_.Grayscale(); 
        });
        Console.WriteLine($"width:{img.Width}, heigh:{img.Height}");
        /*
        var gray = new float[img.Width, img.Height];
        for(int i = 0; i < img.Width; i++)
        {
            for(int j = 0; j < img.Height; j++)
            {
                gray[i,j] = (float)((img[i,j].B / 255) -0.5) *2;
            }
        }
        Console.WriteLine(gray.ToString());
        Console.WriteLine(gray[5, 5]);
        */
        var rgb = new Argb32[img.Width * img.Height];
        img.CopyPixelDataTo(rgb);
        var gray = new List<Argb32>(rgb).Select(x => { return (float)((x.B / 255) - 0.5) * 2; }).ToArray();

        Console.WriteLine(Forward(gray).OutPutImage.Length);
        
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
        [VectorType(64)] // TO FIX INPUT
        [ColumnName("input1")]
        public float[] InputImage { get; set; }
    }
    private class OutputFormat
    {
        [ColumnName("output")] //TO FIX OUTPUT
        public float[] OutPutImage { get; set; }
    }
}
