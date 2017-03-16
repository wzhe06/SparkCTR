package com.ipinyou.pricemodel

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by ggstar on 1/6/17.
 */
object WinrateStat {

  val COL_SPLIT = '\t'
  val KEY_SPLIT = '@'

  //taobao@www.wasu.cn@null@mm_32875384_3406520_40324525@300@250
  //taobao#mm_26632162_2469125_22350506
  //baidu@null@com.diosapp.nhb@7581330927849651016@640@100


  val PLATFORM = "baidu"
  val DOMAIN = "com.diosapp.nhb"
  val ADSLOTID = "7581330927849651016"
  val HOUR = "14"

  val datapath = "hdfs:///user/data-userprofile/cpm/cpmbyday/2017/01/17/*"

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    winpricedist(sc)
  }

  def winpricedist(sc:SparkContext): Unit = {
    val priceData = sc.textFile(datapath)

    val adslotdata = priceData.filter{ data =>

      var result = false
      val datas = data.split(COL_SPLIT)

      val platform = datas(4).trim
      val domain = datas(5).trim
      val app = datas(6).trim
      val adslotid = datas(7).trim
      val hour = datas(10).trim.substring(0,2)
      val bidprice = datas(11).toInt

      if(bidprice > 470 && platform.equals(PLATFORM) && (domain.equals(DOMAIN) || app.equals(DOMAIN)) && adslotid.equals(ADSLOTID) && hour.equals(HOUR))
        result = true
      result
    }.map{ data =>

      val datas = data.split(COL_SPLIT)

      val timeinterval = datas(10)

      val day = datas(0)
      val bidprice = datas(11)
      val winprice = datas(12)
      val wincount = datas(13)
      val bidcount = datas(14)

      val adslotKey = datas(4) + KEY_SPLIT + datas(5) + KEY_SPLIT + datas(6) + KEY_SPLIT + datas(7) + KEY_SPLIT +
        datas(8) + KEY_SPLIT + datas(9)

      val key = adslotKey + COL_SPLIT + day + COL_SPLIT + timeinterval

      (winprice.toInt, wincount.toInt)
    }.reduceByKey{(a,b) =>
      a+b
    }.sortByKey().collect()

    var count = 0


    adslotdata.foreach(a => count += a._2)

    var curcount = 0
    adslotdata.foreach{a =>
      curcount += a._2
      println(a._1, curcount.toDouble / count.toDouble, a._2)
    }
    //adslotdata.repartition(1).saveAsTextFile("/user/data-usermodel/price_dist/windist_20170117")
  }

  def winrate(sc:SparkContext): Unit ={
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


    val priceData = sc.textFile(datapath)
    //val priceData = sc.textFile("/Users/ggstar/Desktop/DSP/log/20170110price.log")

    val adslotdata = priceData.filter{ data =>

      var result = false
      val datas = data.split(COL_SPLIT)

      val platform = datas(4).trim
      val domain = datas(5).trim
      val app = datas(6).trim
      val adslotid = datas(7).trim
      val hour = datas(10).trim.substring(0,2)

      if(platform.equals(PLATFORM) && (domain.equals(DOMAIN) || app.equals(DOMAIN)) && adslotid.equals(ADSLOTID) && hour.equals(HOUR))
        result = true
      result
    }.map{ data =>

      val datas = data.split(COL_SPLIT)

      val timeinterval = datas(10)

      val day = datas(0)
      val bidprice = datas(11)
      val winprice = datas(12)
      val wincount = datas(13)
      val bidcount = datas(14)

      val adslotKey = datas(4) + KEY_SPLIT + datas(5) + KEY_SPLIT + datas(6) + KEY_SPLIT + datas(7) + KEY_SPLIT +
        datas(8) + KEY_SPLIT + datas(9)

      val key = adslotKey + COL_SPLIT + day + COL_SPLIT + timeinterval

      (bidprice.toInt ,(key, wincount.toInt, bidcount.toInt))
    }.reduceByKey{(a,b) =>
      val key = a._1
      var wincount = 0
      var bidcount = 0
      if(a._1.equals(b._1))
        bidcount = a._3
      else
        bidcount = a._3 + b._3
      (key, a._2 + b._2, bidcount)
    }.sortByKey()
      .map{ data =>
      (data._1, data._2._2.toDouble / data._2._3.toDouble, data._2._2, data._2._3)
    }

    adslotdata.repartition(1).saveAsTextFile("/user/data-usermodel/price_dist/winrate_20170117")
  }
}
