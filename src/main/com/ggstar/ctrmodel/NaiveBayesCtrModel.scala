package com.ggstar.ctrmodel

import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.sql.DataFrame

class NaiveBayesCtrModel {
  def train(samples:DataFrame) : NaiveBayesModel = {
    val nbModel = new NaiveBayes().setFeaturesCol("scaledFeatures").setLabelCol("label")
    nbModel.fit(samples)
  }
}
