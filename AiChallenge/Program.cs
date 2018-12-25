using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.IO;
using System.Linq;

namespace AiChallenge
{
    class Program
    {
        //Paths for data files
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "data.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "eval1_unlabelled.tsv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static TextLoader _textLoader;
        //Main Function
        static void Main(string[] args)
        {
            //Console.WriteLine("Hello World!");
            var mlContext = new MLContext();

            //Load Data
            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("QueryId",DataKind.Text,0),
                    new TextLoader.Column("Query",DataKind.Text,1),
                    new TextLoader.Column("PassageText",DataKind.Text,2),
                    new TextLoader.Column("Label",DataKind.Bool,3),
                    new TextLoader.Column("PassageID",DataKind.Text,4)
                }
            });

            var model = Train(mlContext, _trainDataPath);

            //Evaluate(mlContext, model);

            

            Console.WriteLine("Done!");
            Console.ReadLine();

        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            
            //throw new NotImplementedException();
            IDataView dataView = _textLoader.Read(_testDataPath);
            Console.WriteLine("Evaluating Model with test data");
            var predictions = model.Transform(dataView);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
        }

        private static ITransformer Train(MLContext mlContext, string TrainDataPath)
        {
            var data = _textLoader.Read(TrainDataPath);

            //Splitting data
            Console.WriteLine("Splitting Dataset");
            // Split the data 70:30 into train and test sets, train and evaluate.
            var (LeftData, TakenData) = mlContext.BinaryClassification.TrainTestSplit(data, testFraction: 0.3);

            var (trainData, testData) = mlContext.BinaryClassification.TrainTestSplit(TakenData, testFraction: 0.2);

            var pipeline =

                //Featurize Query
                mlContext.Transforms.Text.FeaturizeText("Query", "QueryFeatures")

                //Featurize Passage
                .Append(mlContext.Transforms.Text.FeaturizeText("PassageText", "PassageFeatures"))

                .Append(mlContext.Transforms.Concatenate("Features", "QueryFeatures", "PassageFeatures"))

                .Append(mlContext.BinaryClassification.Trainers.FastTree());

            ////Normalize Query
            //.Append(mlContext.Transforms.Text.NormalizeText("Query", "NormalizedQuery"))

            ////Normalize Passage
            //.Append(mlContext.Transforms.Text.NormalizeText("PassageText", "NormalizedPassage"))

            //// NLP pipeline 1: bag of words.
            //.Append(new WordBagEstimator(mlContext, "NormalizedQuery", "QueryBOW"))
            //.Append(new WordBagEstimator(mlContext, "NormalizedPassage", "PassageBOW"))

            //// NLP pipeline 2: bag of bigrams, using hashes instead of dictionary indices.
            ////BOB = Bag of Bigrams
            //.Append(new WordHashBagEstimator(mlContext, "NormalizedQuery", "QueryBOB", ngramLength: 2, allLengths: false))
            //.Append(new WordHashBagEstimator(mlContext, "NormalizedPassage", "PassageBOB", ngramLength: 2, allLengths: false))

            //// NLP pipeline 3: bag of tri-character sequences with TF-IDF weighting.
            //.Append(mlContext.Transforms.Text.TokenizeCharacters("PassageText", "PassageChars"))
            //.Append(new NgramExtractingEstimator(mlContext, "PassageChars", "BagOfTrichar", ngramLength: 3, weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf));

            // NLP pipeline 4: word embeddings.
            //.Append(mlContext.Transforms.Text.TokenizeWords("NormalizedQuery", "TokenizedQuery"))
            //.Append(mlContext.Transforms.Text.ExtractWordEmbeddings("TokenizedQuery", "QueryEmbeddings", WordEmbeddingsExtractingTransformer.PretrainedModelKind.GloVeTwitter25D))

            //.Append(mlContext.Transforms.Text.TokenizeWords("NormalizedPassage", "TokenizedPassage"))
            //.Append(mlContext.Transforms.Text.ExtractWordEmbeddings("TokenizedPassage", "PassageEmbeddings", WordEmbeddingsExtractingTransformer.PretrainedModelKind.FastTextWikipedia300D));

            //pipeline.Append(mlContext.Transforms.DropColumns(new string[] {"QueryId","PassageID"}));

            //var transformedData = pipeline.Fit(trainData).Transform(trainData);

            //pipeline.Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));



            //Console.WriteLine("Transforming Data...");
            //var transformedData = pipeline.Fit(trainData).Transform(trainData);


            //var pipeline2 = mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20);

            Console.WriteLine("Creating and Training the model...");

            var model = pipeline.Fit(trainData);

            SaveModelAsFile(mlContext, model);

            Console.WriteLine("Completed Training");

            //Evaluation
            Console.WriteLine("Starting Evaluation");
            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall}"); // Correctly predicted positive instances among all positive instances.
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision}"); //Correctly predicted Positive instances among all Positives Predictions
            Console.WriteLine($"Negetive Recall: {metrics.NegativeRecall}");
            Console.WriteLine($"Negetive Precision: {metrics.NegativePrecision}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            //Cross Validation
            //Console.WriteLine("Starting Cross Validation");

            //var cvResults = mlContext.BinaryClassification.CrossValidate(data,pipeline,numFolds:5);
            //var microAccuracies = cvResults.Select(r => r.metrics.Accuracy);
            //Console.WriteLine(microAccuracies.Average());

            return model;
        }
    }
}
