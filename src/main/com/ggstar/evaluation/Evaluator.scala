package com.ggstar.evaluation

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame

class Evaluator {
  def evaluate(predictions:DataFrame):Unit = {

    import  predictions.sparkSession.implicits._

    /*
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")
    */
    //predictions.select("label", "probability").show(1000,false)
    //predictions.select("label", "probability").printSchema()

    val scoreAndLabels = predictions.select("label", "probability").map { row =>
      (row.apply(1).asInstanceOf[DenseVector](1), row.getAs[Int]("label").toDouble)
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels.rdd)

    println("AUC under PR = " + metrics.areaUnderPR())
    println("AUC under ROC = " + metrics.areaUnderROC())
  }
}
