package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.gbtlr.{GBTLRClassificationModel, GBTLRClassifier}
import org.apache.spark.sql.DataFrame

class GBTLRCtrModel {

  var _pipelineModel:PipelineModel = null
  var _model:GBTLRClassificationModel = null

  def train(samples:DataFrame) : Unit = {
    val fe = new FeatureEngineering()
    val samplesWithInnerProduct = fe.calculateEmbeddingInnerProduct(samples)
    _pipelineModel = fe.preProcessInnerProductSamples(samplesWithInnerProduct)

    _model = new GBTLRClassifier()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .setGBTMaxIter(10)
      .setLRMaxIter(100)
      .setRegParam(0.01)
      .setElasticNetParam(0.5)
      .fit(_pipelineModel.transform(samplesWithInnerProduct))
  }

  def transform(samples:DataFrame):DataFrame = {
    val samplesWithInnerProduct = new FeatureEngineering().calculateEmbeddingInnerProduct(samples)
    _model.transform(_pipelineModel.transform(samplesWithInnerProduct))
  }
}
