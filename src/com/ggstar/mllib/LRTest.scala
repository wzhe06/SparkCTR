package com.ggstar.mllib

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * A test of logistic regression model
  *
  * @author zhe.wang
  */
object LRTest {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val data:RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    //split samples into training samples and testing samples
    val splits = data.randomSplit(Array(0.8,0.2))
    val training = splits(0).cache()
    val test = splits(1)

    val iterNum =10

    //build a SVM model
    val model = SVMWithSGD.train(training, iterNum)
    model.clearThreshold()

    val score = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    //build metrics to evaluate the SVM model
    val metrics = new BinaryClassificationMetrics(score)
    val auc = metrics.areaUnderROC()
    println("auc=" + auc)


    //build a logistic regression model, and use BFGS to train the model
    val lrmodel = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)
    val lrresult = test.map{ point =>
      (lrmodel.predict(point.features), point.label)
    }

    val lrmetrics = new MulticlassMetrics(lrresult)
    println("lr precision:\t" + lrmetrics.precision)
    lrmodel.save(sc, "./modelresult")


    val sgdlrmodel = LinearRegressionWithSGD.train(training, 100)

    val sgdresult = test.map { point =>
      (point.label, sgdlrmodel.predict(point.features))
    }

    val MSE = sgdresult.map{case(v, p) => v-p}.mean()
    println("sgd mse:\t" + MSE)

  }
}
