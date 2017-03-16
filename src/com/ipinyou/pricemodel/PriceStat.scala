package com.ipinyou.pricemodel

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by ggstar on 1/6/17.
 */
object PriceStat {

  val COL_SPLIT = '\t'
  val KEY_SPLIT = '@'

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    //val sqlContext = new SQLContext(sc)

    //hdfs:///user/root/flume/express/2016/12/10/20/impression_bid_201612102014_0.seq
    val priceData = sc.textFile("hdfs:///user/data-userprofile/cpm/2017/01/09/*")
    //val priceData = sc.textFile("/Users/ggstar/Desktop/DSP/log/cpm.log")
    val adslot2bidprice = priceData.map{ data =>
      val datas = data.split(COL_SPLIT)
      val adslotKey = datas(3) + KEY_SPLIT + datas(4) + KEY_SPLIT + datas(5) + KEY_SPLIT + datas(6) + KEY_SPLIT + datas(7) + KEY_SPLIT + datas(8)
      val timeinterval = datas(9)
      val bidprice = datas(10)
      val winprice = datas(11)
      val wincount = datas(12)
      val bidcount = datas(13)

      val key = adslotKey + COL_SPLIT + bidprice
      val value = bidcount
      (key, value.toDouble.toInt)
    }.reduceByKey((a,b) => a+b).map{ a =>
      (a._1.split(COL_SPLIT)(0), a._1.split(COL_SPLIT)(1) + ":" + a._2)
    }.aggregateByKey("")(
      (a:String, b:String) => b,
      (price1:String, price2:String) => price1 + " " +price2)


    val adslot2winprice = priceData.map{ data =>
      val datas = data.split(COL_SPLIT)
      val adslotKey = datas(3) + KEY_SPLIT + datas(4) + KEY_SPLIT + datas(5) + KEY_SPLIT + datas(6) + KEY_SPLIT + datas(7) + KEY_SPLIT + datas(8)
      val timeinterval = datas(9)
      val bidprice = datas(10)
      val winprice = datas(11)
      val wincount = datas(12)
      val bidcount = datas(13)

      val key = adslotKey + COL_SPLIT + winprice
      val value = wincount
      (key, value.toDouble.toInt)
    }.reduceByKey((a,b) => a+b).map{ a =>
      (a._1.split(COL_SPLIT)(0), a._1.split(COL_SPLIT)(1) + ":" + a._2)
    }.aggregateByKey("")(
        (a:String, b:String) => b,
        (price1:String, price2:String) => price1 + " " +price2)

    val bidpricewinprice = adslot2bidprice.join(adslot2winprice).map(a => a._2)

    bidpricewinprice.saveAsTextFile("/user/data-usermodel/price_dist/adslot_20170109_bidwin")

    //adslot2pricecounts.repartition(1).saveAsTextFile("/Users/ggstar/Desktop/DSP/pricedata/20170106")
  }
}
