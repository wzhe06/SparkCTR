package com.ggstar.serving.mleap.serialization

import ml.combust.bundle.BundleFile
import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.sql.DataFrame
import resource.managed

class ModelSerializer {
  def serializeModel(pipelineModel:PipelineModel, model:Transformer, modelSavePath:String, transformedData:DataFrame): Unit ={
    val pipeline = SparkUtil.createPipelineModel(uid = "pipeline", Array(pipelineModel, model))

    val sbc = SparkBundleContext().withDataset(transformedData)
    for(bf <- managed(BundleFile(modelSavePath))) {
      pipeline.writeBundle.save(bf)(sbc).get
    }
  }

}
