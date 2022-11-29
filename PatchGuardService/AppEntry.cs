using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;



namespace PatchGuardService;


internal class AppEntry
{   
    internal static void Main()
    {
        var img = Image.Load<Argb32>("C:\\Users\\Thinkpad\\Pictures\\Pixel Studio\\captcha.jpg");
        
        Recognizer ctl = new();
        ctl.Predict(img);

    }
}