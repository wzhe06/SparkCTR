package com.ggstar.example

import com.ggstar.ctrmodel._
import com.ggstar.features.FeatureEngineering
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object ModelSerialization {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()

    val resourcesPath = this.getClass.getResource("/samples.snappy.orc")
    val rawSamples = spark.read.format("orc").option("compression", "snappy").load(resourcesPath.getPath)


    //transform array to vector for following vectorAssembler
    val samples = FeatureEngineering.transferArray2Vector(rawSamples)

    samples.printSchema()
    samples.show(5, false)


    //model training
    println("Neural Network Ctr Prediction Model:")
    val innModel = new InnerProductNNCtrModel()
    innModel.train(samples)
    val transformedData = innModel.transform(samples)

    transformedData.show(1,false)

    //model serialization by mleap
    val mleapModelSerializer = new com.ggstar.serving.mleap.serialization.ModelSerializer()
    mleapModelSerializer.serializeModel(innModel._pipelineModel, "jar:file:/Users/zhwang/Workspace/CTRmodel/model/inn.model.mleap.zip", transformedData)

    //model serialization by JPMML
    val jpmmlModelSerializer = new com.ggstar.serving.jpmml.serialization.ModelSerializer()
    jpmmlModelSerializer.serializeModel(innModel._pipelineModel, "model/inn.model.jpmml.xml", transformedData)
  }
}
