package com.ggstar.ctr

import com.ggstar.mllib.GetDomain
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by ggstar on 12/28/16.
 */
object CTRPredictModel {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //val impData = sc.textFile("/Users/ggstar/Desktop/DSP/log/imp_sample.seq")
    //hdfs:///user/root/flume/express/2016/12/15/*/impression_bid_*

    //val impData = sc.textFile("hdfs:///user/root/flume/express/2016/12/10/20/impression_bid_201612102014_0.seq")
    //val clkData = sc.textFile("hdfs:///user/root/flume/express/2016/12/10/20/click_bid_201612102014_0.seq")

    val impData = sc.textFile("/Users/ggstar/Desktop/DSP/log/impression_bid_201612102014_0.seq")
    val clkData = sc.textFile("/Users/ggstar/Desktop/DSP/log/click_bid_201612102014_0.seq")

    val clkimpLabelDF = sampleProcess(impData, clkData, sqlContext)

    val featureData = featureProcess(clkimpLabelDF)

    val Array(trainingData, testData) = featureData.randomSplit(Array(0.7, 0.3))
    //GBDTCTRModel(trainingData, testData)
    LRCTRModel(trainingData, testData)
  }


  def sampleProcess(impData: RDD[String], clkData: RDD[String], sqlContext: SQLContext): DataFrame = {
    val clickLabel = clkData.map { clklog =>
      val uuid = clklog.split(Delimiter.PARTION, -1)(1).split(Delimiter.SECTION, -1)(1).split(Delimiter.FIELD, -1)(0)
      (uuid, 1)
    }

    val impFeatures = impData.map { log =>
      val logs = log.split('\u0003')
      val implog = logs.apply(0).split('\u0001')
      val bidlog = logs.apply(1).split('\u0001')
      val platform = bidlog.apply(1).split('\u0002').apply(6)
      val url = bidlog.apply(4)
      var app = bidlog.apply(5)

      val urllen = bidlog.apply(4).split('\u0002').length
      val applen = bidlog.apply(5).split('\u0002').length

      var domain = "default"
      if (urllen > 1) {
        val tdomain = GetDomain.getdomain(bidlog.apply(4).split('\u0002').apply(1))
        if (tdomain.trim.length > 0)
          domain = tdomain
      } else if (applen > 1) {
        val adomain = bidlog.apply(5).split('\u0002').apply(6)
        if (adomain.trim.length > 0)
          domain = adomain
      }
      val uuid = log.split(Delimiter.PARTION, -1)(1).split(Delimiter.SECTION, -1)(1).split(Delimiter.FIELD, -1)(0)


      val site = bidlog.apply(7).split('\u0002')
      var adSlot = ""
      var width = ""
      var height = ""
      if (site.length > 0) {
        adSlot = bidlog.apply(7).split('\u0002').apply(0)
      }
      if (site.length > 6) {
        width = bidlog.apply(7).split('\u0002').apply(6)
      }
      if (site.length > 7) {
        height = bidlog.apply(7).split('\u0002').apply(7)
      }
      adSlot = platform + "#" + domain + "#" + adSlot + "#" + width + "#" + height


      (uuid, (platform, domain, adSlot))
    }

    val clkImp = impFeatures.leftOuterJoin(clickLabel)

    val clkImpLabel = clkImp.map { r =>
      var label = 1.0
      if (r._2._2.isEmpty)
        label = 0.0
      (r._2._1._1, r._2._1._2, r._2._1._3, label)
    }

    val clkImpLabelDF = sqlContext.createDataFrame(clkImpLabel).toDF("platform", "domain", "adSlot", "label")
    clkImpLabelDF
  }

  def featureProcess(sample: DataFrame): DataFrame = {
    val platformIndexer = new StringIndexer()
      .setInputCol("platform")
      .setOutputCol("platformIndex")

    val domainIndexer = new StringIndexer()
      .setInputCol("domain")
      .setOutputCol("domainIndex")

    val adSlotIndexer = new StringIndexer()
      .setInputCol("adSlot")
      .setOutputCol("adSlotIndex")

    val platformEncoder = new OneHotEncoder()
      .setInputCol("platformIndex")
      .setOutputCol("platformId")
      .setDropLast(false)

    val domainEncoder = new OneHotEncoder()
      .setInputCol("domainIndex")
      .setOutputCol("domainId")
      .setDropLast(false)

    val adSlotEncoder = new OneHotEncoder()
      .setInputCol("adSlotIndex")
      .setOutputCol("adSlotId")
      .setDropLast(false)

    val vectorAsCols = Array("platformId", "domainId", "adSlotId")
    val vectorAssembler = new VectorAssembler().setInputCols(vectorAsCols).setOutputCol("vectorFeature")

    /*将特征转换，特征聚合，模型等组成一个管道，并调用它的fit方法拟合出模型*/
    val pipelineStage: Array[PipelineStage] = Array(platformIndexer, domainIndexer, adSlotIndexer, platformEncoder, domainEncoder, adSlotEncoder, vectorAssembler)
    val featurePipline = new Pipeline().setStages(pipelineStage)

    val featureModel = featurePipline.fit(sample)
    val featureData = featureModel.transform(sample)
    featureData
  }

  def LRCTRModel(trainingSample: DataFrame, testSample: DataFrame) = {
    val logModel = new LogisticRegression().setMaxIter(100).setRegParam(0.1).setElasticNetParam(0.0)
      .setFeaturesCol("vectorFeature").setLabelCol("label")



    val ctrModel = logModel.fit(trainingSample)

    val testResult = ctrModel.transform(testSample)

    val output = testResult.select("vectorFeature", "label", "prediction", "rawPrediction", "probability")
    val prediction = output.select("label", "prediction", "rawPrediction", "probability")


    prediction.printSchema()
    val scoreAndLabels = prediction.select("label", "probability").map { row =>
      ((row.apply(1).asInstanceOf[DenseVector]).apply(1), row.apply(0).asInstanceOf[Double])
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("AUC = " + auROC)
    //ctrModel.save("/Users/ggstar/Desktop/ctrmodel.dat")
    ctrModel.explainParams()

    //ctrModel.save(sc, "myModelPath")
    //val sameModel = LinearRegressionModel.load(sc, "myModelPath")



    //.toPMML("/tmp/kmeans.xml")


    //prediction.show();
    //result.select("lable", "platform", "domain", "prediction", "probability").toJavaRDD.repartition(1).saveAsTextFile("/Users/ggstar/Desktop/platform")

  }

  def GBDTCTRModel(trainingSample: DataFrame, testSample: DataFrame) = {

    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("vectorFeature")
      .setMaxIter(10)

    val ctrModel = gbt.fit(trainingSample)

    val testResult = ctrModel.transform(testSample)

    val output = testResult.select("vectorFeature", "label", "prediction", "rawPrediction", "probability")
    val prediction = output.select("label", "prediction", "rawPrediction", "probability")


    prediction.printSchema()
    val scoreAndLabels = prediction.select("label", "probability").map { row =>
      ((row.apply(1).asInstanceOf[DenseVector]).apply(1), row.apply(0).asInstanceOf[Double])
    }

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("AUC = " + auROC)
    //prediction.show();
    //result.select("lable", "platform", "domain", "prediction", "probability").toJavaRDD.repartition(1).saveAsTextFile("/Users/ggstar/Desktop/platform")

  }

}
