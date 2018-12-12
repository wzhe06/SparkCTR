package com.ggstar.example

import com.ggstar.serving.mleap.load.ModelServer
import ml.combust.mleap.core.types.{ScalarType, StructField, StructType}
import ml.combust.mleap.runtime.frame.Row
import ml.combust.mleap.tensor.Tensor

object MLeapModelServing {
  def main(args: Array[String]): Unit = {
    //model load
    val dataSchema = StructType(
      StructField("user_id", ScalarType.Int),
      StructField("item_id", ScalarType.Int),
      StructField("category_id", ScalarType.Int),
      StructField("content_type", ScalarType.String),
      StructField("timestamp", ScalarType.String),
      StructField("user_item_click", ScalarType.Long),
      StructField("user_item_imp", ScalarType.Double),
      StructField("item_ctr", ScalarType.Double),
      StructField("is_new_user", ScalarType.Int),
      StructField("embedding_inner_product", ScalarType.Double)
    ).get
    val modelServer = new ModelServer("jar:file:/Users/zhwang/Workspace/CTRmodel/model/inn.model.zip", dataSchema)
    modelServer.loadModel()

    val data = Row(20143, 52, 16, "movie", "1533487890", 0l, 0.6931472d, 0.00617256d, 0, 0.5)

    val result = modelServer.forecast(data)

    for(x <- 0 until result.size){
      println(x, result.get(x))
    }

    val probabilities:Tensor[Double] = result.getTensor[Double](16)
    println("ctr", probabilities.get(1).head)

  }
}
