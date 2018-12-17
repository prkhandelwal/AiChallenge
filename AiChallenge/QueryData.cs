using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace AiChallenge
{
    public class QueryData
    {
        [Column(ordinal:"0")]
        public int QueryId { get; set; }
        [Column(ordinal:"1")]
        public string Query { get; set; }
        [Column(ordinal:"2")]
        public string PassageText { get; set; }
        [Column(ordinal:"3", name:"Label")]
        public float Label { get; set; }
        [Column(ordinal:"4")]
        public int PassageID { get; set; }
    }

    public class QueryPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        [ColumnName("Probability")]
        public float Probability { get; set; }
        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
