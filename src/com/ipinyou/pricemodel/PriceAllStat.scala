package com.ipinyou.pricemodel

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by ggstar on 1/6/17.
 */
object PriceAllStat {

  val COL_SPLIT = '\t'
  val KEY_SPLIT = '@'

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    //val sqlContext = new SQLContext(sc)

    //hdfs:///user/root/flume/express/2016/12/10/20/impression_bid_201612102014_0.seq
    //one week data
    /*
      adslot	day	hour	E(bidprice)	E(winprice)
      adslot	day	bidprice	count
      adslot	day	winprice	count
      adslot	day	hour	bidprice	count
      adslot	day	hour	winprice	count
      adslot	day	priceDiff	count
      adslot	day	hour	priceDiff	count
    */


    val priceData = sc.textFile("hdfs:///user/data-userprofile/cpm/cpmbyday/2017/01/10/*")
    //val priceData = sc.textFile("/Users/ggstar/Desktop/DSP/log/20170110price.log")



    val adslothour2bidprice = priceData.map{ data =>
      val datas = data.split(COL_SPLIT)

      val timeinterval = datas(10)

      val day = datas(0)
      val hour = timeinterval.substring(0,2)

      val bidprice = datas(11)
      val winprice = datas(12)
      val wincount = datas(13)
      val bidcount = datas(14)

      val adslotKey = datas(4) + KEY_SPLIT + datas(5) + KEY_SPLIT + datas(6) + KEY_SPLIT + datas(7) + KEY_SPLIT +
        datas(8) + KEY_SPLIT + datas(9)

      val key = adslotKey + COL_SPLIT + day + COL_SPLIT + hour + COL_SPLIT + bidprice
      val value = bidcount
      (key, value.toDouble.toInt)
    }.reduceByKey((a,b) => a+b).map(a => a._1 + COL_SPLIT + a._2)


    val adslothour2winprice = priceData.map{ data =>
      val datas = data.split(COL_SPLIT)

      val timeinterval = datas(10)

      val day = datas(0)
      val hour = timeinterval.substring(0,2)

      val bidprice = datas(11)
      val winprice = datas(12)
      val wincount = datas(13)
      val bidcount = datas(14)

      val adslotKey = datas(4) + KEY_SPLIT + datas(5) + KEY_SPLIT + datas(6) + KEY_SPLIT + datas(7) + KEY_SPLIT +
        datas(8) + KEY_SPLIT + datas(9)

      val key = adslotKey + COL_SPLIT + day + COL_SPLIT + hour + COL_SPLIT + winprice
      val value = wincount
      (key, value.toDouble.toInt)
    }.reduceByKey((a,b) => a+b).filter(a => a._2 > 0).map(a => a._1 + COL_SPLIT + a._2)

    /*
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
        (price1:String, price2:String) => price1 + " " +price2)*/

    //val bidpricewinprice = adslothour2bidprice.join(adslot2winprice).map(a => a._2)

    adslothour2bidprice.saveAsTextFile("/user/data-usermodel/price_dist/adslot_20170110_bidpricedist")
    adslothour2winprice.saveAsTextFile("/user/data-usermodel/price_dist/adslot_20170109_winpricedist")

    //adslothour2winprice.repartition(1).saveAsTextFile("/Users/ggstar/Desktop/DSP/pricedata/20170116_winpricehourly")
  }
}
