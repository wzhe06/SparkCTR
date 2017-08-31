package com.ggstar.mllib

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

/**
  * A standard process of transforming id feature to one hot feature
  * @author zhe.wang
  */

object OneHotFeatureTest {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val sqlContext=new SQLContext(sc)

    //create a data frame
    val df = sqlContext.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")
    df.show()

    //a standard transform process
    //transform category to category index
    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)
    indexed.show()

    //transfrom category index to one hot vector
    val encoder = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")
    encoder.setDropLast(false)

    val encoded = encoder.transform(indexed)
    encoded.select("id", "category", "categoryVec").show()
  }
}
