package com.ggstar.example

import com.ggstar.ctrmodel._
import com.ggstar.features.FeatureEngineering
import com.ggstar.serving.serialization.ModelSerializer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object ModelSerialization {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("ctrModel")
    new SparkContext(conf)
    val spark = SparkSession.builder.appName("ctrModel").getOrCreate()

    val resourcesPath = this.getClass.getResource("/samples.snappy.orc")
    val rawSamples = spark.read.format("orc").option("compression", "snappy").load(resourcesPath.getPath)


    //transform array to vector for following vectorAssembler
    val samples = new FeatureEngineering().transferArray2Vector(rawSamples)

    samples.printSchema()
    samples.show(5, false)


    //model training
    println("Neural Network Ctr Prediction Model:")
    val lrModel = new LogisticRegressionCtrModel()
    lrModel.train(samples)
    val transformedData = lrModel.transform(samples)

    //model serialization
    val modelSerializer = new ModelSerializer()
    modelSerializer.serializeModel(lrModel._pipelineModel, lrModel._model, "jar:file:/Users/zhwang/Workspace/CTRmodel/model/lr.model.zip", transformedData)
  }
}
