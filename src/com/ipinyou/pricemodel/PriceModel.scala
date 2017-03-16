package com.ipinyou.pricemodel

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by wang zhe on 1/6/17.
 */
object PriceModel {

  val COL_SPLIT = '\t'
  val KEY_SPLIT = '@'
  val dataRoot = "hdfs:///user/data-userprofile/cpm/cpmbyday"
  val modelRoot = "/user/data-usermodel/price_model"

  def main(args: Array[String]) {
    val year = args(0)
    val month = args(1)
    val day = args(2)

    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    adSlotMean(sc, year, month, day)
  }

  def adSlotMean(sc:SparkContext, year:String, month:String, day:String) = {
    val dataPath = dataRoot + "/" + year + "/" + month + "/" + day + "/*"

    val priceData = sc.textFile(dataPath)

    val adSlotHourWinData = priceData.map{ data =>

      val dataCols = data.split(COL_SPLIT)

      val platform = dataCols(4).trim
      val domain = dataCols(5).trim
      val app = dataCols(6).trim
      val adSlotId = dataCols(7).trim
      val width = dataCols(8).trim
      val height = dataCols(9).trim
      val hour = dataCols(10).trim.substring(0,2)
      val clock = dataCols(10).trim

      val bidPrice = dataCols(11)
      val winPrice = dataCols(12)
      val winCount = dataCols(13)
      val bidCount = dataCols(14)

      val adSlotKey = platform + KEY_SPLIT + domain + KEY_SPLIT + app + KEY_SPLIT + adSlotId + KEY_SPLIT +
        width + KEY_SPLIT + height

      ((adSlotKey, hour.toInt, winPrice.toDouble), winCount.toInt)
    }.reduceByKey(_+_).map{data =>
      ((data._1._1, data._1._2), (data._1._3, data._2))
    }.reduceByKey{ (a,b) =>
      val count = a._2 + b._2
      var mean = 0.0
      if(count == 0)
        mean = 0
      else
        mean = a._1 * (a._2.toDouble / count.toDouble) + b._1 * (b._2.toDouble / count.toDouble)

      (mean, count)
    }

    val adSlotDayWinData = adSlotHourWinData.map{ data =>
      ((data._1._1, 100), (data._2._1, data._2._2))
    }.reduceByKey{ (a,b) =>
      val count = a._2 + b._2
      var mean = 0.0
      if(count == 0)
        mean = 0
      else
        mean = a._1 * (a._2.toDouble / count.toDouble) + b._1 * (b._2.toDouble / count.toDouble)

      (mean, count)
    }

    val adSlotWinData = adSlotHourWinData.union(adSlotDayWinData)


    val adSlotHourBidData = priceData.map{ data =>

      val dataCols = data.split(COL_SPLIT)

      val platform = dataCols(4).trim
      val domain = dataCols(5).trim
      val app = dataCols(6).trim
      val adSlotId = dataCols(7).trim
      val width = dataCols(8).trim
      val height = dataCols(9).trim
      val hour = dataCols(10).trim.substring(0,2)
      val clock = dataCols(10).trim

      val bidPrice = dataCols(11)
      val winPrice = dataCols(12)
      val winCount = dataCols(13)
      val bidCount = dataCols(14)

      val adSlotKey = platform + KEY_SPLIT + domain + KEY_SPLIT + app + KEY_SPLIT + adSlotId + KEY_SPLIT +
        width + KEY_SPLIT + height

      ((adSlotKey, clock, bidPrice.toDouble), bidCount.toInt)
    }.reduceByKey((a,b) => a).map{data =>
      val hour = data._1._2.trim.substring(0,2)
      ((data._1._1, hour.toInt), (data._1._3, data._2))
    }.reduceByKey{ (a,b) =>
      val count = a._2 + b._2
      var mean = 0.0
      if(count == 0)
        mean = 0
      else
        mean = a._1 * (a._2.toDouble / count.toDouble) + b._1 * (b._2.toDouble / count.toDouble)

      (mean, count)
    }


    val adSlotDayBidData = adSlotHourBidData.map{ data =>
      ((data._1._1, 100), (data._2._1, data._2._2))
    }.reduceByKey{ (a,b) =>
      val count = a._2 + b._2
      var mean = 0.0
      if(count == 0)
        mean = 0
      else
        mean = a._1 * (a._2.toDouble / count.toDouble) + b._1 * (b._2.toDouble / count.toDouble)

      (mean, count)
    }

    val adSlotBidData = adSlotHourBidData.union(adSlotDayBidData)

    val adSlotData = adSlotBidData.join(adSlotWinData).sortByKey().filter(data => data._2._2._2 > 5).map{ data =>
      data._1._1 + COL_SPLIT + data._1._2 +
        COL_SPLIT + "%6.2f".format(data._2._1._1) + COL_SPLIT + data._2._1._2 +
        COL_SPLIT + "%6.2f".format(data._2._2._1) + COL_SPLIT + data._2._2._2 +
        COL_SPLIT + "%6.2f".format(data._2._2._2.toDouble / data._2._1._2.toDouble)
    }

    adSlotData.saveAsTextFile(modelRoot + "/" + year + month + day)
  }
}
