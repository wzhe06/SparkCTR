package com.ggstar.util

/**
  * Extract main domain from a href
  * @author zhe.wang
  */
import java.net.URLDecoder

object GetDomain {

  def getdomain(url:String) : String={
    if (url == null || url.length() == 0) return ""

    try
    {
      val agentUrl = URLDecoder.decode(url, "UTF-8")
      if(agentUrl == null || url.length() == 0) return ""
      val proto = "://";
      var start = agentUrl.indexOf(proto);
      if (start >= 0) {
        start += proto.length();// 包含协议
      } else {
        start = 0;// 不包含协议
      }
      var end = start;
      var length = agentUrl.length()
      while (end < agentUrl.length() && agentUrl.charAt(end) != '?' && agentUrl.charAt(end) != '/')
      {end = end +1}
      return agentUrl.substring(start, end).trim();

    }
    catch {
      case ex:Exception => ""
    }

  }

}
