package com.ggstar.ctrmodel

import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.sql.DataFrame

class NeuralNetworkCtrModel {
  def train(samples:DataFrame) : MultilayerPerceptronClassificationModel = {

    val layers = Array[Int](12, 12, 2)

    val logModel = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(150).setStepSize(0.005)
      .setFeaturesCol("featureVector").setLabelCol("label")

    logModel.fit(samples)
  }
}
