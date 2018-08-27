package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.DataFrame

class RandomForestCtrModel {

  var _pipelineModel:PipelineModel = _
  var _model:RandomForestClassificationModel = _

  def train(samples:DataFrame) : Unit = {
    _pipelineModel = new FeatureEngineering().preProcessSamples(samples)

    _model = new RandomForestClassifier()
      .setNumTrees(10)    //the number of trees
      .setMaxDepth(4)     //the max depth of each tree
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .fit(_pipelineModel.transform(samples))
  }

  def transform(samples:DataFrame):DataFrame = {
    _model.transform(_pipelineModel.transform(samples))
  }
}
