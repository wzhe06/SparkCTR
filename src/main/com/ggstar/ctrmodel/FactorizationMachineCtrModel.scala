package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{FMModel, FMWithSGD, LabeledPoint}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class FactorizationMachineCtrModel {
  var _pipelineModel:PipelineModel = _
  var _model:FMModel = _

  def train(samples:DataFrame) : Unit = {

    val fe = new FeatureEngineering()

    //calculate inner product between item embedding and user embedding
    val samplesWithInnerProduct = fe.calculateEmbeddingInnerProduct(samples)
    _pipelineModel = fe.preProcessInnerProductSamples(samplesWithInnerProduct)

    val preparedSamples = _pipelineModel.transform(samplesWithInnerProduct)

    val formatSamples = preparedSamples.rdd.map( row =>{
      new LabeledPoint(row.getAs[Int]("label").toDouble, Vectors.fromML(row.getAs[DenseVector]("scaledFeatures")))
    })

    _model = FMWithSGD.train(formatSamples, task = 1, numIterations = 150, stepSize = 0.015, miniBatchFraction = 1, dim = (true, true, 2), regParam = (0, 0, 0), initStd = 0.1)
    //_model = FMWithLBFGS.train(formatSamples, task = 1, numIterations = 150, numCorrections = 5, dim = (true, true, 2), regParam = (0, 0, 0), initStd = 0.1)
  }

  def transform(samples:DataFrame):DataFrame = {
    val samplesWithInnerProduct = new FeatureEngineering().calculateEmbeddingInnerProduct(samples)
    val preparedSamples = _pipelineModel.transform(samplesWithInnerProduct)

    _model.predict(preparedSamples)
  }


  def transformRdd(samples:DataFrame):RDD[Double] = {
    val samplesWithInnerProduct = new FeatureEngineering().calculateEmbeddingInnerProduct(samples)
    val preparedSamples = _pipelineModel.transform(samplesWithInnerProduct)

    val formatSamples = preparedSamples.rdd.map( row =>{
      Vectors.fromML(row.getAs[DenseVector]("scaledFeatures"))
    })

    _model.predict(formatSamples)
  }
}
