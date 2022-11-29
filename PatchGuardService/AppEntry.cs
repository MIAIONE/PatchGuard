using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;



namespace PatchGuardService;


internal class AppEntry
{   
    internal static void Main()
    {
        var img = Image.Load<Argb32>(@".\Code.png");
        
        Recognizer ctl = new();
        ctl.Predict(img);

    }
}