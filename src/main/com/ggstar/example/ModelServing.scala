package com.ggstar.example

import com.ggstar.serving.load.ModelServer

object ModelServing {
  def main(args: Array[String]): Unit = {
    //model load
    val modelServer = new ModelServer
    modelServer.loadModel("jar:file:/Users/zhwang/Workspace/CTRmodel/model/lr.model.zip")
  }
}
