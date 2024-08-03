package org.example;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import java.io.IOException;
import java.util.Arrays;

public class WineQualityTraining {
    public static void main(String[] args){
        SparkSession sparkSession = SparkSession.builder().
                appName("WineQualityTraining").master("local[*]").getOrCreate();
        String path = "TrainingDataset.csv";
        String modelPath = "../model";


        Dataset<Row> trainData = sparkSession.read().option("delimiter", ";").option("inferSchema", "true")
                .option("header", "true").csv(path);
        trainData = preProcessData(trainData);

//
//        String[] columnNames = {"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
//                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
//                "pH", "sulphates", "alcohol", "quality"};
//
//        // VectorAssembler is a transformer that combines a given list of columns into a single vector column.
//        for (int i = 0; i < columnNames.length; i++) {
//            trainData = trainData.withColumnRenamed(trainData.columns()[i], columnNames[i]);
//        }
//        String[] inputCols = Arrays.copyOfRange(columnNames, 0, columnNames.length - 1);
//
//
//        VectorAssembler vectorAssembler = new VectorAssembler()
//                .setInputCols(inputCols)
//                .setOutputCol("trainingFeatures");
//
//        trainData = vectorAssembler.transform(trainData);
//        trainData.show();
        // Creating a logistic regression model

        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10)
                .setRegParam(0.3).setFeaturesCol("trainingFeatures").setLabelCol("quality") ;

        // Fit the model
        LogisticRegressionModel lrModel = lr.fit(trainData);

        // Trying with diff models
        DecisionTreeClassifier dTree = new DecisionTreeClassifier().
                setFeaturesCol("trainingFeatures").setLabelCol("quality") ;

        DecisionTreeClassificationModel dtModel = dTree.fit(trainData);

//        // Create and fit a Gradient-Boosted Trees model
//        GBTClassifier gbt = new GBTClassifier()
//                .setLabelCol("quality")
//                .setFeaturesCol("trainingFeatures")
//                .setMaxIter(10);




// Print the coefficients and intercept for multinomial logistic regression
        System.out.println("Coefficients: \n"
                + lrModel.coefficientMatrix() + " \nIntercept: " + lrModel.interceptVector());


        // Running on validation data file as well

        String validationPath = "ValidationDataset.csv";

        Dataset<Row> validationData = sparkSession.read().option("delimiter", ";").option("inferSchema", "true")
                .option("header", "true").csv(validationPath);
        validationData = preProcessData(validationData);
        Dataset<Row> LRPredictions = lrModel.transform(validationData);
        LRPredictions.show();

        Dataset<Row> dtModelPredictions = dtModel.transform(validationData);
        dtModelPredictions.show();
        try {
            dtModel.write().overwrite().save(modelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

//        Dataset<Row> gbtModelPredictions = gbtModel.transform(validationData);
//        gbtModelPredictions.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

// Compute the F1 score
        double f1ScoreLT = evaluator.evaluate(LRPredictions);
        System.out.println("Validation F1 Score of Linear Regression: " + f1ScoreLT);

        // Compute the F1 score of DTree
        double f1ScoreDT = evaluator.evaluate(dtModelPredictions);
        System.out.println("Validation F1 Score of Decision Tree: " + f1ScoreDT);

//        // Compute the F1 score of GBT
//        double f1ScoreGBT = evaluator.evaluate(gbtModelPredictions);
//        System.out.println("Validation F1 Score of Gradient Boosted Tree: " + f1ScoreGBT);
/// we can use decision tree
        sparkSession.stop();

    }
    public static Dataset<Row> preProcessData(Dataset<Row> data) {
        String[] columnNames = {"fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol", "quality"};

        // VectorAssembler is a transformer that combines a given list of columns into a single vector column.
        for (int i = 0; i < columnNames.length; i++) {
            data = data.withColumnRenamed(data.columns()[i], columnNames[i]);
        }
        String[] inputCols = Arrays.copyOfRange(columnNames, 0, columnNames.length - 1);

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(inputCols)
                .setOutputCol("trainingFeatures");

        data = vectorAssembler.transform(data);
        data.show();
        return data;
    }
}