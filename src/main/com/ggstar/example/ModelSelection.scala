package com.ggstar.example

import com.ggstar.ctrmodel._
import com.ggstar.evaluation.Evaluator
import com.ggstar.features.FeatureEngineering
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}

object ModelSelection {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    val resourcesPath = this.getClass.getResource("/samples.snappy.orc")
    val rawSamples = spark.read.format("orc").option("compression", "snappy").load(resourcesPath.getPath)
    rawSamples.printSchema()
    rawSamples.show(10)

    //transform array to vector for following vectorAssembler
    val samples = FeatureEngineering.transferArray2Vector(rawSamples)

    //split samples into training samples and validation samples
    val Array(trainingSamples, validationSamples) = samples.randomSplit(Array(0.7, 0.3))
    val evaluator = new Evaluator

    /*
    println("Logistic Regression Ctr Prediction Model:")
    val lrModel = new LogisticRegressionCtrModel()
    lrModel.train(trainingSamples)
    evaluator.evaluate(lrModel.transform(validationSamples))

    println("FM Ctr Prediction Model:")
    val fmModel = new FactorizationMachineCtrModel()
    fmModel.train(trainingSamples)
    evaluator.evaluate(fmModel.transform(validationSamples))

    println("Neural Network Ctr Prediction Model:")
    val nnModel = new NeuralNetworkCtrModel()
    nnModel.train(trainingSamples)
    evaluator.evaluate(nnModel.transform(validationSamples))

    println("Random Forest Ctr Prediction Model:")
    val rfModel = new RandomForestCtrModel()
    rfModel.train(trainingSamples)
    evaluator.evaluate(rfModel.transform(validationSamples))

    println("GBDT Ctr Prediction Model:")
    val gbtModel = new GBDTCtrModel()
    gbtModel.train(trainingSamples)
    evaluator.evaluate(gbtModel.transform(validationSamples))

    println("GBDT+LR Ctr Prediction Model:")
    val gbtlrModel = new GBTLRCtrModel()
    gbtlrModel.train(trainingSamples)
    evaluator.evaluate(gbtlrModel.transform(validationSamples))*/


    println("IPNN Ctr Prediction Model:")
    val ipnnModel = new InnerProductNNCtrModel()
    ipnnModel.train(trainingSamples)
    evaluator.evaluate(ipnnModel.transform(validationSamples))

/*
    println("OPNN Ctr Prediction Model:")
    val opnnModel = new OuterProductNNCtrModel()
    opnnModel.train(trainingSamples)
    evaluator.evaluate(opnnModel.transform(validationSamples))*/
  }
}
