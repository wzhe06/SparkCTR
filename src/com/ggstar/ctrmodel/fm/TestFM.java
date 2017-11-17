/**
 * Created by zrf on 4/18/15.
 */


object TestFM extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTFM"))

    //    "hdfs://ns1/whale-tmp/url_combined"
    val training = MLUtils.loadLibSVMFile(sc, "hdfs://ns1/whale-tmp/url_combined").cache()

    //    val task = args(1).toInt
    //    val numIterations = args(2).toInt
    //    val stepSize = args(3).toDouble
    //    val miniBatchFraction = args(4).toDouble

    val fm1 = FMWithSGD.train(training, task = 1, numIterations = 100, stepSize = 0.15, miniBatchFraction = 1.0, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)


    val fm2 = FMWithLBFGS.train(training, task = 1, numIterations = 20, numCorrections = 5, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1)

  }
}
