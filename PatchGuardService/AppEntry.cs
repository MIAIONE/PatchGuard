using SixLabors.ImageSharp;
using SixLabors.ImageSharp.ColorSpaces;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PatchGuardService;
internal class AppEntry
{
    internal static void Main()
    {
        var image = Image.Load<Argb32>(@"D:\Pictures\93811943.jpg");
        var destImage = new Argb32[image.Width * image.Height];

        image.Mutate(x => x
        .Grayscale()
        //.BlackWhite() // 别用黑白, 这个是二值化, 只有0和非0
        );

        image.CopyPixelDataTo(destImage);
        //image.SaveAsPng("example.png");
        var list = new List<Argb32>(destImage)
            .Select(x => (float)(x.R + x.G * 256 + x.B * 256 * 256)) //转换成color float色彩空间
            .ToArray();

        //list.ToList().ForEach(Console.WriteLine); // 输出
        Console.ReadKey();
    }
}