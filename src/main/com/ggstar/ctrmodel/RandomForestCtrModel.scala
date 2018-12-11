package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.DataFrame

class RandomForestCtrModel extends BaseCtrModel {

  def train(samples:DataFrame) : Unit = {

    val featureEngineeringStages:Array[PipelineStage] = FeatureEngineering.preProcessSamplesStages()

    val model:RandomForestClassifier = new RandomForestClassifier()
      .setNumTrees(10)    //the number of trees
      .setMaxDepth(4)     //the max depth of each tree
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")

    val pipelineStages = featureEngineeringStages ++ Array(model)

    _pipelineModel = new Pipeline().setStages(pipelineStages).fit(samples)
  }
}
