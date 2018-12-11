package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{FMModel, FMWithSGD, LabeledPoint}
import org.apache.spark.sql.DataFrame

class FactorizationMachineCtrModel extends BaseCtrModel {
  var _model:FMModel = _

  def train(samples:DataFrame) : Unit = {
    //calculate inner product between item embedding and user embedding
    val samplesWithInnerProduct = FeatureEngineering.calculateEmbeddingInnerProduct(samples)
    _pipelineModel = FeatureEngineering.preProcessInnerProductSamples(samplesWithInnerProduct)

    val preparedSamples = _pipelineModel.transform(samplesWithInnerProduct)

    val formatSamples = preparedSamples.rdd.map( row =>{
      new LabeledPoint(row.getAs[Int]("label").toDouble, Vectors.fromML(row.getAs[DenseVector]("scaledFeatures")))
    })

    _model = FMWithSGD.train(formatSamples, task = 1, numIterations = 200, stepSize = 0.15, miniBatchFraction = 1, dim = (true, true, 2), regParam = (0, 0, 0), initStd = 0.1)
  }

  override def transform(samples:DataFrame):DataFrame = {
    val samplesWithInnerProduct = FeatureEngineering.calculateEmbeddingInnerProduct(samples)
    val preparedSamples = _pipelineModel.transform(samplesWithInnerProduct)

    _model.predict(preparedSamples)
  }
}
