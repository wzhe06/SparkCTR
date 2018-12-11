package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame

class LogisticRegressionCtrModel extends BaseCtrModel {

  def train(samples:DataFrame) : Unit = {

    val featureEngineeringStages:Array[PipelineStage] = FeatureEngineering.preProcessSamplesStages()

    val model:LogisticRegression = new LogisticRegression()
      .setMaxIter(20)           //max iteration
      .setRegParam(0.0)         //regularization parameter
      .setElasticNetParam(0.0)  //0-L2 regularization 1-L1 regularization
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")

    val pipelineStages = featureEngineeringStages ++ Array(model)

    _pipelineModel = new Pipeline().setStages(pipelineStages).fit(samples)
  }
}
