package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame

class NeuralNetworkCtrModel extends BaseCtrModel {

  def train(samples:DataFrame) : Unit = {
    val prePipelineModel = FeatureEngineering.preProcessSamples(samples)

    val preparedSamples = prePipelineModel.transform(samples)

    //network architecture, better to keep tuning it until metrics converge
    val layers = Array[Int](preparedSamples.first().getAs[DenseVector]("scaledFeatures").toArray.length,
      preparedSamples.first().getAs[DenseVector]("scaledFeatures").toArray.length / 2, 2)

    val nnModel = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(150)                //max iterations, keep increasing it if loss function or metrics don't converge
      .setStepSize(0.005)             //learning step size, larger size will lead to loss vibration
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")

    val pipelineStages = prePipelineModel.stages ++ Array(nnModel)

    _pipelineModel = new Pipeline().setStages(pipelineStages).fit(samples)
  }
}
