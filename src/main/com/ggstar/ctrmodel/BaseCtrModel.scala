package com.ggstar.ctrmodel

import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

trait BaseCtrModel {
  var _pipelineModel:PipelineModel = _

  def train(samples:DataFrame) : Unit

  def transform(samples:DataFrame):DataFrame = {
    _pipelineModel.transform(samples)
  }
}
