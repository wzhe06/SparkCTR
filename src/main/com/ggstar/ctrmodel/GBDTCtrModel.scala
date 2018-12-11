package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.sql.DataFrame

class GBDTCtrModel extends BaseCtrModel {

  def train(samples:DataFrame) : Unit = {

    val featureEngineeringStages:Array[PipelineStage] = FeatureEngineering.preProcessSamplesStages()

    val model:GBTClassifier = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
      .setMaxIter(10)
      .setFeatureSubsetStrategy("auto")

    val pipelineStages = featureEngineeringStages ++ Array(model)

    _pipelineModel = new Pipeline().setStages(pipelineStages).fit(samples)
  }
}
