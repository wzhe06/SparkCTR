package com.ipinyou.offlinemodel

import com.hadoop.mapreduce.LzoTextInputFormat
import org.apache.hadoop.io.{Text, LongWritable}
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * Created by ggstar on 12/7/16.
 */
object BaseData {
  def main (args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val data = sc.textFile("hdfs:///user/root/flume/express/2016/12/07/13/click_201612071329_0.seq")


    ///user/root/flume/express/2016/12/14/*/impression_bid_*
    val datalength = data.map(func1)

    val count = datalength.reduce((x,y) => x+y)

    println(count)
    //datalength.collect().foreach(x => println(x))


    //println(count)
    //val clickdata = sc.newAPIHadoopFile[LongWritable, Text, LzoTextInputFormat]("hdfs:///user/root/flume/express/2016/12/07/13/click_201612071329_0.seq");

    //val splitdata = clickdata.flatMap(x => x._2.toString.split('\u0001'))
  }


  def func1(s:String):Int={
    s.length
  }
}
