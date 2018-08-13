package com.ggstar.ctrmodel

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

class LogisticRegressionCtrModel {
  def train(samples:DataFrame) : LogisticRegressionModel = {
    val lrModel = new LogisticRegression().setMaxIter(20).setRegParam(0.0).setElasticNetParam(0.0)
      .setFeaturesCol("scaledFeatures").setLabelCol("label")

    lrModel.fit(samples)
  }
}
