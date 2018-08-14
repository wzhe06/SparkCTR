package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame

class OuterProductNNCtrModel {
  var _pipelineModel:PipelineModel = null
  var _model:MultilayerPerceptronClassificationModel = null

  def train(samples:DataFrame) : Unit = {

    val fe = new FeatureEngineering()
    val samplesWithOuterProduct = fe.calculateEmbeddingOuterProduct(samples)
    _pipelineModel = fe.preProcessOuterProductSamples(samplesWithOuterProduct)

    val preparedSamples = _pipelineModel.transform(samplesWithOuterProduct)

    val layers = Array[Int](preparedSamples.first().getAs[DenseVector]("scaledFeatures").toArray.length,
      preparedSamples.first().getAs[DenseVector]("scaledFeatures").toArray.length / 2, 2)

    val nnModel = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(150).setStepSize(0.005)
      .setFeaturesCol("scaledFeatures").setLabelCol("label")

    _model = nnModel.fit(preparedSamples)
  }

  def transform(samples:DataFrame):DataFrame = {
    val samplesWithOuterProduct = new FeatureEngineering().calculateEmbeddingOuterProduct(samples)
    _model.transform(_pipelineModel.transform(samplesWithOuterProduct))
  }
}
