package com.ggstar.ctrmodel

import com.ggstar.features.FeatureEngineering
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame

class OuterProductNNCtrModel extends BaseCtrModel {

  def train(samples:DataFrame) : Unit = {
    //calculate inner product between item embedding and user embedding
    val samplesWithOuterProduct = FeatureEngineering.calculateEmbeddingOuterProduct(samples)

    val prePipelineModel = FeatureEngineering.preProcessOuterProductSamples(samplesWithOuterProduct)

    val preparedSamples = prePipelineModel.transform(samplesWithOuterProduct)

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
    val pipelineStages = prePipelineModel.stages ++ Array(nnModel)

    _pipelineModel = new Pipeline().setStages(pipelineStages).fit(samplesWithOuterProduct)
  }

  override def transform(samples:DataFrame):DataFrame = {
    val samplesWithOuterProduct = FeatureEngineering.calculateEmbeddingOuterProduct(samples)
    _pipelineModel.transform(samplesWithOuterProduct)
  }
}
