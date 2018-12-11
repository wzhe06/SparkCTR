package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.gbtlr.GBTLRClassifier
import org.apache.spark.sql.DataFrame

class GBTLRCtrModel extends BaseCtrModel {

  def train(samples:DataFrame) : Unit = {
    val samplesWithInnerProduct = FeatureEngineering.calculateEmbeddingInnerProduct(samples)

    val featureEngineeringStages:Array[PipelineStage] = FeatureEngineering.preProcessInnerProductSamplesStages()

    val model = new GBTLRClassifier()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .setGBTMaxIter(10)
      .setLRMaxIter(100)
      .setRegParam(0.01)
      .setElasticNetParam(0.5)

    val pipelineStages = featureEngineeringStages ++ Array(model)

    _pipelineModel = new Pipeline().setStages(pipelineStages).fit(samplesWithInnerProduct)
  }

  override def transform(samples:DataFrame):DataFrame = {
    val samplesWithInnerProduct = FeatureEngineering.calculateEmbeddingInnerProduct(samples)
    _pipelineModel.transform(samplesWithInnerProduct)
  }
}
