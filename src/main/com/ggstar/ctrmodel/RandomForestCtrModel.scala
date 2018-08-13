package com.ggstar.ctrmodel

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.DataFrame

class RandomForestCtrModel {
  def train(samples:DataFrame) : RandomForestClassificationModel = {
    val rfModel = new RandomForestClassifier().setNumTrees(10).setMaxDepth(4)
      .setFeaturesCol("vectorFeature").setLabelCol("label")

    rfModel.fit(samples)
  }
}
