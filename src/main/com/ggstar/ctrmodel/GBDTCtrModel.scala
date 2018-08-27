package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.sql.DataFrame

class GBDTCtrModel {

  var _pipelineModel:PipelineModel = _
  var _model:GBTClassificationModel = _

  def train(samples:DataFrame) : Unit = {
    _pipelineModel = new FeatureEngineering().preProcessSamples(samples)

    _model = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setMaxIter(10)
      .setFeatureSubsetStrategy("auto")
      .fit(_pipelineModel.transform(samples))
  }

  def transform(samples:DataFrame):DataFrame = {
    _model.transform(_pipelineModel.transform(samples))
  }
}
