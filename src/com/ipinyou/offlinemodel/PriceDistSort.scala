package com.ipinyou.offlinemodel

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by ggstar on 12/7/16.
 */
object PriceDistSort {


  val platform = "inmobi"
  val domain = "883891897"
  val adslot = "74f09f3a87534fd58fc192b079e47dc0_320_50"
  val width = "320"
  val height = "50"

  def main (args: Array[String]) {
    val conf = new SparkConf().setAppName("adslot_price").setMaster("local[4]")
    val sc = new SparkContext(conf)
    //val data = sc.textFile("/Users/ggstar/Desktop/DSP/log/impression_bid_201612102014_0.seq")

    val data = sc.textFile("hdfs:///user/root/flume/express/2016/12/15/*/impression_bid_*")


    val filterdata = data.filter(adslotfilter)
    val pricedata = filterdata.map(priceProcess)

    val pricegroup = pricedata.sortByKey()

    val priceresult = pricegroup.map(flatPricePair)

    //priceresult.repartition(1).saveAsTextFile("/Users/ggstar/Desktop/pricedataadslot")
    priceresult.repartition(1).saveAsTextFile("/user/data-usermodel/price_dist/adslot_15_all")

    //val datalength = data.map(func1)

    //val count = datalength.reduce((x,y) => x+y)

    ///println(count)
    //datalength.collect().foreach(x => println(x))


    //println(count)
    //val clickdata = sc.newAPIHadoopFile[LongWritable, Text, LzoTextInputFormat]("hdfs:///user/root/flume/express/2016/12/07/13/click_201612071329_0.seq");

    //val splitdata = clickdata.flatMap(x => x._2.toString.split('\u0001'))
  }

  def flatPricePair(log:(String, String)): String ={
    log._1 + "\t" + log._2
  }

  def priceProcess(log:String): (String, String) ={
    var logs = log.split('\u0003')
    var implog = logs.apply(0).split('\u0001')
    var bidlog = logs.apply(1).split('\u0001')

    var requestTime = implog.apply(1).split('\u0002').apply(10)
    var winprice = implog.apply(8).split('\u0002').apply(2)

    var platform = bidlog.apply(1).split('\u0002').apply(6)

    var url = bidlog.apply(4)//bidlog.apply(4).split('\u0002').apply(1)
    var app = bidlog.apply(5)//bidlog.apply(5).split('\u0002').apply(6)

    var urllen = bidlog.apply(4).split('\u0002').length//bidlog.apply(4).split('\u0002').apply(1)
    var applen = bidlog.apply(5).split('\u0002').length//bidlog.apply(5).split('\u0002').apply(6)

    var domain = ""
    if(urllen > 1){
      domain = GetDomain.getdomain(bidlog.apply(4).split('\u0002').apply(1))
    }else if(applen > 1){
      domain = bidlog.apply(5).split('\u0002').apply(6)
    }

    var site = bidlog.apply(7).split('\u0002')
    var adslot = ""
    var width = ""
    var height = ""
    var floorprice = 0
    if(site.length > 0){
      adslot = bidlog.apply(7).split('\u0002').apply(0)
    }
    if(site.length > 6){
      width = bidlog.apply(7).split('\u0002').apply(6)
    }
    if(site.length > 7){
      height = bidlog.apply(7).split('\u0002').apply(7)
    }
    if(site.length > 14){
      floorprice = bidlog.apply(7).split('\u0002').apply(14).toDouble.toInt
    }

    var bidPrice = bidlog.apply(8).split('\u0002').apply(1)

    var pricelog = requestTime + "\t" + platform + "\t" + domain + "\t" +
      adslot + "\t" + width + "\t" + height + "\t" + floorprice + "\t" + bidPrice + "\t" + winprice

    var timedate = timeFormat(requestTime)
    var day = timedate.substring(0,8)
    //var month = timedate.substring(4,6)
    // var day = timedate.substring(6,8)
    var hour = timedate.substring(8,10)
    var minu = timedate.substring(10,12)
    var sec = timedate.substring(12,14)
    var msec = timedate.substring(14,17)

    var priceKey = platform + "$" + domain + "$" + adslot + "$" + width + "$" + height + "\t" + day + "\t" + hour + "\t" + minu + "\t" + sec + "\t" + msec
    var priceValue = floorprice + "\t" + bidPrice.toDouble.toInt + "\t" + winprice.toDouble.toInt

    (priceKey, priceValue)
  }


  def adslotfilter(log:String): Boolean ={
    var logs = log.split('\u0003')
    var implog = logs.apply(0).split('\u0001')
    var bidlog = logs.apply(1).split('\u0001')

    var requestTime = implog.apply(1).split('\u0002').apply(10)
    var winprice = implog.apply(8).split('\u0002').apply(2)

    var platform = bidlog.apply(1).split('\u0002').apply(6)

    var url = bidlog.apply(4)//bidlog.apply(4).split('\u0002').apply(1)
    var app = bidlog.apply(5)//bidlog.apply(5).split('\u0002').apply(6)

    var urllen = bidlog.apply(4).split('\u0002').length//bidlog.apply(4).split('\u0002').apply(1)
    var applen = bidlog.apply(5).split('\u0002').length//bidlog.apply(5).split('\u0002').apply(6)

    var domain = ""
    if(urllen > 1){
      domain = GetDomain.getdomain(bidlog.apply(4).split('\u0002').apply(1))
    }else if(applen > 6){
      domain = bidlog.apply(5).split('\u0002').apply(6)
    }

    var site = bidlog.apply(7).split('\u0002')
    var adslot = ""
    var width = ""
    var height = ""
    var floorprice = 0
    if(site.length > 0){
      adslot = bidlog.apply(7).split('\u0002').apply(0)
    }
    if(site.length > 6){
      width = bidlog.apply(7).split('\u0002').apply(6)
    }
    if(site.length > 7){
      height = bidlog.apply(7).split('\u0002').apply(7)
    }
    if(site.length > 14){
      floorprice = bidlog.apply(7).split('\u0002').apply(14).toDouble.toInt
    }
    var bidPrice = bidlog.apply(8).split('\u0002').apply(1)

    platform.trim.equals(this.platform) && domain.trim.equals(this.domain)  && adslot.trim.equals(this.adslot) && width.trim.equals(this.width) && height.trim.equals(this.height)
  }

  def timeFormat(time:String):String={
    var sdf:SimpleDateFormat = new SimpleDateFormat("yyyyMMddHHmmssSSS")
    var date:String = sdf.format(new Date((time.toLong)))
    date
  }


  def func1(s:String):Array[String]={
    s.split('\u0003')
  }
}
