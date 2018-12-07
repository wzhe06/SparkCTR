package com.ggstar.serving.mleap.load

import ml.combust.bundle.BundleFile
import ml.combust.mleap.core.types._
import resource.managed
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row, Transformer}


class ModelServer(var modelPath:String, var dataSchema:StructType) {

  var _model:Transformer = _

  def loadModel(): Unit ={
    // load the Spark pipeline we saved in the previous section
    val bundle = (for(bundleFile <- managed(BundleFile(modelPath))) yield {
      bundleFile.loadMleapBundle().get
    }).opt.get
    this._model = bundle.root
  }

  def forecast(features:Row): Row ={
    if (this._model == null){
      loadModel()
    }
    if (features == null){
      throw
      null
    }

    val frame = DefaultLeapFrame(dataSchema, Seq(features))
    val resultFrame = this._model.transform(frame).get
    resultFrame.dataset.head
  }

}
