package com.ggstar.features

import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame

class FeatureEngineering {
  def preProcessSamples(samples:DataFrame):DataFrame = {
    import samples.sparkSession.implicits._

    val samplesVector = samples
      .map(row => {
        (row.getAs[Int]("user_id"),
          row.getAs[Int]("item_id"),
          row.getAs[Int]("category_id"),
          row.getAs[String]("content_type"),
          row.getAs[String]("timestamp"),
          row.getAs[Long]("user_item_click"),
          row.getAs[Double]("user_item_imp"),
          row.getAs[Double]("item_ctr"),
          row.getAs[Int]("is_new_user"),
          Vectors.dense(row.getAs[Seq[Double]]("user_embedding").toArray),
          Vectors.dense(row.getAs[Seq[Double]]("item_embedding").toArray),
          row.getAs[Int]("label")
        )
      }).toDF(
      "user_id",
      "item_id",
      "category_id",
      "content_type",
      "timestamp",
      "user_item_click",
      "user_item_imp",
      "item_ctr",
      "is_new_user",
      "user_embedding",
      "item_embedding",
      "label")

    val contentTypeIndexer = new StringIndexer().setInputCol("content_type").setOutputCol("content_type_index")

    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("content_type_index"))
      .setOutputCols(Array("content_type_vector"))
      .setDropLast(false)

    val ctr_discretizer = new QuantileDiscretizer()
      .setInputCol("item_ctr")
      .setOutputCol("ctr_bucket")
      .setNumBuckets(100)

    val vectorAsCols = Array("content_type_vector", "ctr_bucket", "user_item_click", "user_item_imp", "is_new_user", "user_embedding", "item_embedding")
    val vectorAssembler = new VectorAssembler().setInputCols(vectorAsCols).setOutputCol("vectorFeature")

    val scaler = new MinMaxScaler().setInputCol("vectorFeature").setOutputCol("scaledFeatures")

    /*
    val scaler = new StandardScaler()
      .setInputCol("vectorFeature")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)
    */

    val pipelineStage: Array[PipelineStage] = Array(contentTypeIndexer, oneHotEncoder, ctr_discretizer, vectorAssembler, scaler)
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    val featureModel = featurePipeline.fit(samplesVector)

    featureModel.transform(samplesVector)
  }
}
