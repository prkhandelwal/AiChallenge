using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System;
using System.IO;

namespace ConsumeModel
{
    class Program
    {
        static readonly string _modelPath = Path.Combine("F:","MSFT AI Challenge","Model.zip");
        static ITransformer loadedModel;
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            Console.WriteLine("Loading Model...");
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);    
            }
            //Console.WriteLine("Model Loaded");
        }
    }
}
