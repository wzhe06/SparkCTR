package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

class LogisticRegressionCtrModel {

  var _pipelineModel:PipelineModel = null
  var _model:LogisticRegressionModel = null

  def train(samples:DataFrame) : Unit = {
    _pipelineModel = new FeatureEngineering().preProcessSamples(samples)

    _model = new LogisticRegression().setMaxIter(20).setRegParam(0.0).setElasticNetParam(0.0)
      .setFeaturesCol("scaledFeatures").setLabelCol("label")
      .fit(_pipelineModel.transform(samples))
  }

  def transform(samples:DataFrame):DataFrame = {
    _model.transform(_pipelineModel.transform(samples))
  }
}
