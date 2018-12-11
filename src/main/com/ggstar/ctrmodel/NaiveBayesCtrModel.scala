package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.sql.DataFrame

class NaiveBayesCtrModel extends BaseCtrModel {

  def train(samples:DataFrame) : Unit = {

    val featureEngineeringStages:Array[PipelineStage] = FeatureEngineering.preProcessSamplesStages()

    val model:NaiveBayesModel = new NaiveBayes().setFeaturesCol("scaledFeatures").setLabelCol("label")
      .fit(_pipelineModel.transform(samples))

    val pipelineStages = featureEngineeringStages ++ Array(model)

    _pipelineModel = new Pipeline().setStages(pipelineStages).fit(samples)
  }
}
