package com.ipinyou.offlinemodel

import java.text.SimpleDateFormat
import java.util.Calendar

object DateUtils {
  
  def getYesterDay (dateFormat: String) : String = {
    val sdf : SimpleDateFormat = new SimpleDateFormat(dateFormat)
    val cal : Calendar = Calendar.getInstance
    cal.add(Calendar.DAY_OF_MONTH, -1)
    val dateTime = sdf.format(cal.getTime)
    dateTime
  }
  
  def getCurrentDay (dateFormat: String) = {
    val sdf : SimpleDateFormat = new SimpleDateFormat(dateFormat)
    val cal : Calendar = Calendar.getInstance
    val dateTime = sdf.format(cal.getTime)
    dateTime
  }
}