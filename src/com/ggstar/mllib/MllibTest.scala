package com.ggstar.mllib

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD


import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
/**
 * Created by ggstar on 12/19/16.
 */
object MllibTest {
  def main(args: Array[String]) {
    val dv:Vector = Vectors.dense(1.0,2.0,3.0)
    val sv:Vector = Vectors.sparse(2, Array(0,2), Array(1.0, 3.0))

    val pos:LabeledPoint = LabeledPoint(1, dv)
    val neg:LabeledPoint = LabeledPoint(0, sv)

    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val rddPoints:RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")


    val dm:Matrix = Matrices.dense(3, 2, Array(1,2,3,4,5,6))


    val sm:Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))

    val sqlContext=new SQLContext(sc)


    val df = sqlContext.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)
    indexed.show()

    val encoder = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")
    encoder.setDropLast(false)
    val encoded = encoder.transform(indexed)
    encoded.select("id", "categoryVec").show()


    //val rddsv:RDD[Vector] = sc.parallelize(sv)

    //val summary : MultivariateStatisticalSummary = Statistics.colStats(rddsv)
    //println(summary)

  }

}
