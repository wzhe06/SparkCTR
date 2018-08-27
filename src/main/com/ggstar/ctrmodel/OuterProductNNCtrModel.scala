package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame

class OuterProductNNCtrModel {
  var _pipelineModel:PipelineModel = _
  var _model:MultilayerPerceptronClassificationModel = _

  def train(samples:DataFrame) : Unit = {

    val fe = new FeatureEngineering()

    //calculate outer product between item embedding and user embedding
    val samplesWithOuterProduct = fe.calculateEmbeddingOuterProduct(samples)
    _pipelineModel = fe.preProcessOuterProductSamples(samplesWithOuterProduct)

    val preparedSamples = _pipelineModel.transform(samplesWithOuterProduct)

    //network architecture, better to keep tuning it until metrics converge
    val layers = Array[Int](preparedSamples.first().getAs[DenseVector]("scaledFeatures").toArray.length,
      preparedSamples.first().getAs[DenseVector]("scaledFeatures").toArray.length / 2, 2)

    val nnModel = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(150)      //max iterations, keep increasing it if loss function or metrics don't converge
      .setStepSize(0.005)   //learning step size, larger size will lead to loss vibration
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")

    _model = nnModel.fit(preparedSamples)
  }

  def transform(samples:DataFrame):DataFrame = {
    val samplesWithOuterProduct = new FeatureEngineering().calculateEmbeddingOuterProduct(samples)
    _model.transform(_pipelineModel.transform(samplesWithOuterProduct))
  }
}
