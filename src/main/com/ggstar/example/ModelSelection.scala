package com.ggstar.example

import com.ggstar.ctrmodel.{LogisticRegressionCtrModel, NaiveBayesCtrModel}
import com.ggstar.evaluation.Evaluator
import com.ggstar.features.FeatureEngineering
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object ModelSelection {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("ctrModel")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder.appName("ctrModel").getOrCreate()
    import spark.implicits._

    val resourcesPath = this.getClass.getResource("/samples.snappy.orc")
    val rawSamples = spark.read.format("orc").option("compression", "snappy").load(resourcesPath.getPath)
    rawSamples.printSchema()
    rawSamples.show(10)

    val samples = new FeatureEngineering().preProcessSamples(rawSamples)

    samples.printSchema()
    samples.select($"scaledFeatures").show(10)

    val Array(trainingSamples, validationSamples) = samples.randomSplit(Array(0.7, 0.3))

    val lrModel = new LogisticRegressionCtrModel().train(trainingSamples)
    val nbModel = new NaiveBayesCtrModel().train(trainingSamples)

    val evaluator = new Evaluator
    evaluator.evaluate(nbModel.transform(validationSamples))
    evaluator.evaluate(lrModel.transform(validationSamples))
  }
}
