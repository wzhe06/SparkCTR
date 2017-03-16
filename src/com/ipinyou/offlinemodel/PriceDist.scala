package com.ipinyou.offlinemodel

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by ggstar on 12/7/16.
 */
object PriceDist {
  def main (args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val data = sc.textFile("/Users/ggstar/Desktop/DSP/log/impression_bid_201612102014_0.seq")


    val pricedata = data.map(priceProcess)

    val pricegroup = pricedata.reduceByKey((a,b) => a+b)

    val priceresult = pricegroup.map(flatPricePair)


    priceresult.repartition(1).saveAsTextFile("/Users/ggstar/Desktop/pricedata")




    //val datalength = data.map(func1)

    //val count = datalength.reduce((x,y) => x+y)

    ///println(count)
    //datalength.collect().foreach(x => println(x))


    //println(count)
    //val clickdata = sc.newAPIHadoopFile[LongWritable, Text, LzoTextInputFormat]("hdfs:///user/root/flume/express/2016/12/07/13/click_201612071329_0.seq");

    //val splitdata = clickdata.flatMap(x => x._2.toString.split('\u0001'))
  }

  def flatPricePair(log:(String, Int)): String ={
    log._1 + "#" + log._2
  }

  def priceProcess(log:String): (String, Int) ={
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
    var floorprice = 0.0
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
      floorprice = bidlog.apply(7).split('\u0002').apply(14).toDouble
    }


    var bidPrice = bidlog.apply(8).split('\u0002').apply(1)
    var pricelog = requestTime + "\t" + platform + "\t" + domain + "\t" +
      adslot + "\t" + width + "\t" + height + "\t" + floorprice + "\t" + bidPrice + "\t" + winprice

    var priceKey = platform + "#" + domain + "#" + adslot + "#" + width + "#" + height + "#" + floorprice+ "#" + bidPrice.toDouble+ "#" + winprice.toDouble
    //var priceValue = requestTime + "#" + floorprice.toDouble.toInt + "#" + bidPrice.toDouble.toInt + "#" + winprice.toDouble.toInt

    (priceKey, 1)
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
