package com.ggstar.serving.jpmml.serialization

import java.io.{File, FileOutputStream}

import javax.xml.transform.stream.StreamResult
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame
import org.dmg.pmml.PMML
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.PMMLBuilder

class ModelSerializer {
  def serializeModel(pipelineModel:PipelineModel, modelSavePath:String, transformedData:DataFrame): Unit ={

    val pmml:PMML = new PMMLBuilder(transformedData.schema, pipelineModel).build()

    val output = new FileOutputStream(new File(modelSavePath))
    JAXBUtil.marshalPMML(pmml, new StreamResult(output))
  }
}
