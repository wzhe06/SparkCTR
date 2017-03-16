package com.ipinyou.mllib

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by ggstar on 12/19/16.
 */
object LR {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val data:RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    val splits = data.randomSplit(Array(0.8,0.2))
    val training = splits(0).cache()
    val test = splits(1)

    val iterNum =1

    val model = SVMWithSGD.train(training, iterNum)
    model.clearThreshold()

    val score = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val metrics = new BinaryClassificationMetrics(score)
    val auc = metrics.areaUnderROC()

    println("auc=" + auc)

    val lrmodel = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)
    val lrresult = test.map{ point =>
      (lrmodel.predict(point.features), point.label)
    }

    val lrmetrics = new MulticlassMetrics(lrresult)
    println("lr precision:\t" + lrmetrics.precision)
    //model.save(sc, "/Users/ggstar/Desktop")



    val sgdlrmodel = LinearRegressionWithSGD.train(training, 100)

    val sgdresult = test.map { point =>
      (point.label, sgdlrmodel.predict(point.features))
    }

    val MSE = sgdresult.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("sgd mse:\t" + MSE)

  }
}
