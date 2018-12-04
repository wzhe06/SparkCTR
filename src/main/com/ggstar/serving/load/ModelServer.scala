package com.ggstar.serving.load

import breeze.linalg.DenseVector
import ml.combust.bundle.BundleFile
import ml.combust.mleap.core.types
import ml.combust.mleap.core.types._
import resource.managed
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import ml.combust.mleap.tensor.DenseTensor

class ModelServer {

  def loadModel(modelPath:String): Unit ={
    // load the Spark pipeline we saved in the previous section
    val bundle = (for(bundleFile <- managed(BundleFile(modelPath))) yield {
      bundleFile.loadMleapBundle().get
    }).opt.get

    val mleapPipeline = bundle.root
    //mleapPipeline.transform()

    /*
 |-- user_id: integer (nullable = false)
 |-- item_id: integer (nullable = false)
 |-- category_id: integer (nullable = false)
 |-- content_type: string (nullable = true)
 |-- timestamp: string (nullable = true)
 |-- user_item_click: long (nullable = false)
 |-- user_item_imp: double (nullable = false)
 |-- item_ctr: double (nullable = false)
 |-- is_new_user: integer (nullable = false)
 |-- user_embedding: vector (nullable = true)
 |-- item_embedding: vector (nullable = true)
 |-- label: integer (nullable = false)
     */

    val schema = StructType(
      StructField("user_id", ScalarType.Int),
      StructField("item_id", ScalarType.Int),
      StructField("category_id", ScalarType.Int),
      StructField("content_type", ScalarType.String),
      StructField("timestamp", ScalarType.String),
      StructField("user_item_click", ScalarType.Long),
      StructField("user_item_imp", ScalarType.Double),
      StructField("item_ctr", ScalarType.Double),
      StructField("is_new_user", ScalarType.Int),
      StructField("user_embedding", TensorType.Double(50)),
      StructField("item_embedding", TensorType.Double(50))
    ).get

    val data = Seq(Row(
      20143,
      52,
      16,
      "movie",
      "1533487890",
      0l,
      0.6931471805599453d,
      0.0061725628301584d,
      0,
      DenseTensor( Array(0.0648365244269371,0.06820321083068848,0.15572160482406616,0.07844140380620956,-0.08746618032455444,-0.04901598021388054,-0.17239613831043243,0.048738326877355576,-0.04094882681965828,-0.039663463830947876,-0.134134903550148,0.2068929672241211,0.2723809480667114,-0.07512179017066956,0.05858059227466583,0.3259638547897339,-0.2890375852584839,-0.1909564733505249,0.24588070809841156,0.07095829397439957,0.050444766879081726,0.028928957879543304,0.07975737005472183,-0.14088886976242065,0.10529613494873047,-0.005059529095888138,-0.19789129495620728,-0.0768406093120575,0.0017437433125451207,0.10173505544662476,-0.03718102350831032,0.06624675542116165,0.07223014533519745,-0.1690087467432022,0.013677106238901615,0.022961946204304695,0.006647794507443905,0.09317808598279953,-0.07613910734653473,0.0383693166077137,-0.5037217140197754,0.11296124011278152,0.011819849722087383,0.09127316623926163,-0.01910974271595478,-0.06737814843654633,-0.06524071842432022,0.017754212021827698,0.15760280191898346,-0.13364417850971222), Seq(50)),
      DenseTensor( Array(-0.0905480831861496,-0.013500932604074478,0.27808094024658203,0.035076357424259186,-0.1065259799361229,-0.0584576390683651,0.011284944601356983,-0.11926604062318802,0.18736493587493896,0.08999132364988327,-0.1331675499677658,0.24956414103507996,0.23744040727615356,-0.12897568941116333,0.18764284253120422,0.052384231239557266,-0.11342161893844604,-0.07042761892080307,0.3005072772502899,-0.18694230914115906,-0.0070617711171507835,-0.12645111978054047,-0.049484848976135254,0.3104129731655121,0.20119524002075195,0.0840258002281189,-0.21941445767879486,-0.0672524943947792,-0.01718808338046074,-0.0115513876080513,-0.18150198459625244,0.03447090834379196,-0.2834555208683014,-0.0027743238024413586,-0.08593913167715073,-0.11656057834625244,-0.01464556623250246,-0.05811721086502075,0.008251870982348919,-0.10781512409448624,-0.08287085592746735,-0.015524163842201233,0.03674255311489105,0.15884864330291748,0.12991121411323547,-0.09415002167224884,-0.16861139237880707,-0.21277128159999847,0.07131009548902512,0.04732019454240799), Seq(50))
    ))
    val frame = DefaultLeapFrame(schema, data)

    val resultFrame = mleapPipeline.transform(frame).get

    val resultData = resultFrame.dataset

    println(resultData(0).size)

    var x = 0
    for(x <- 0 until resultData.head.size){
      println(resultData.head.get(x))
    }


  }

}
