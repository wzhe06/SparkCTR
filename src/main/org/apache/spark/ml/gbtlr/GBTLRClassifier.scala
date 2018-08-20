package org.apache.spark.ml.gbtlr

import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger
import org.apache.spark.SparkException
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.linalg.{DenseVector => OldDenseVector}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.{FeatureType, Algo => OldAlgo, BoostingStrategy => OldBoostingStrategy, Strategy => OldStrategy}
import org.apache.spark.mllib.tree.impurity.{Variance => OldVariance}
import org.apache.spark.mllib.tree.loss.{LogLoss => OldLogLoss, Loss => OldLoss}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, GradientBoostedTreesModel, Node => OldNode}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.immutable.HashMap
import scala.collection.mutable


trait GBTLRClassifierParams extends Params {

  // =====below are GBTClassifier params=====
  /**
    * Param for set checkpoint interval (&gt;= 1) or disable checkpoint (-1).
    *
    * E.g. 10 means that the cache will get checkpointed every 10 iterations.
    * @group param
    */
  val checkpointInterval: IntParam = new IntParam(this, "checkpointInterval", "set" +
      "checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache " +
      "will get checkpointed every 10 iterations",
    (interval: Int) => interval == -1 || interval >= 1)

  /** @group getParam */
  def getCheckpointInterval: Int = $(checkpointInterval)

  /**
    * Loss function which GBT tries to minimize. (case-insensitive)
    *
    * Supported: "logistic"
    * @group param
    */
  val lossType: Param[String] = new Param[String](this, "lossType", "Loss funtion which GBT" +
      " tries to minimize (case-insensitive). Supported options: logistic, squared, absolute",
    (value: String) => value == "logistic")

  /** @group getParam */
  def getLossType: String = $(lossType).toLowerCase

  /**
    * Maximum number of bins used for discretizing continuous features and for choosing how to split
    * on features at each node.  More bins give higher granularity.
    * Must be >= 2 and >= number of categories in any categorical feature.
    *
    * (default = 32)
    * @group param
    */
  val maxBins: IntParam = new IntParam(this, "maxBins", "Max number of bins for" +
      " discretizing continuous features. Must be >= 2 and >= number of categories for any" +
      "categorical feature.", ParamValidators.gtEq(2))

  /** @group getParam */
  def getMaxBins: Int = $(maxBins)

  /**
    * Maximum depth of the tree ( >= 0).
    * E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
    *
    * (default = 5)
    * @group param
    */
  val maxDepth: IntParam =
    new IntParam(this, "maxDepth", "Maximum depth of the tree. (>= 0)" +
        " E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.",
      ParamValidators.gtEq(0))

  /** @group getParam */
  def getMaxDepth: Int = $(maxDepth)

  /**
    * If false, the algorithm will pass trees to executors to match instances with nodes.
    *
    * If true, the algorithm will cache node IDs for each instance.
    *
    * Caching can speed up training of deeper trees. Users can set how often should the
    * cache be checkpointed or disable it by setting checkpointInterval.
    *
    * (default = false)
    * @group param
    */
  val cacheNodeIds: BooleanParam = new BooleanParam(this, "cacheNodeIds", "If" +
      "false, the algorithm will pass trees to executors to match instances with nodes." +
      "If true, the algorithm will cache node IDs for each instance. Caching can speed" +
      "up training of deeper trees.")

  /** @group getParam */
  def getCacheNodeIds: Boolean = $(cacheNodeIds)

  /**
    * Maximum memory in MB allocated to histogram aggregation. If too small,
    * then 1 node will be split per iteration, and its aggregates may exceed this size.
    *
    * (default = 256 MB)
    * @group param
    */
  val maxMemoryInMB: IntParam = new IntParam(this, "maxMemoryInMB",
    "Maximum memory in MB allocated to histogram aggregation.",
    ParamValidators.gtEq(0))

  /** @group getParam */
  def getMaxMemoryInMB: Int = $(maxMemoryInMB)

  /**
    * Minimum number of instances each child must have after split.
    * If a split causes the left or right child to have fewer than minInstancesPerNode,
    * the split will be discarded as invalid.
    * Should be >= 1.
    *
    * (default = 1)
    * @group param
    */
  val minInstancesPerNode: IntParam = new IntParam(this, "minInstancesPerNode",
    "Minimum number of instances each child must have after split.  If a split causes" +
        " the left or right child to have fewer than minInstancesPerNode, the split" +
        "will be discarded as invalid. Should be >= 1.", ParamValidators.gtEq(1))

  /** @group getParam */
  def getMinInstancePerNode: Int = $(minInstancesPerNode)

  /**
    * Minimum information gain for a split to be considered at a tree node.
    * Should be >= 0.0.
    *
    * (default = 0.0)
    * @group param
    */
  val minInfoGain: DoubleParam = new DoubleParam(this, "minInfoGain",
    "Minimum information gain for a split to be considered at a tree node.",
    ParamValidators.gtEq(0.0))

  /** @group getParam */
  def getMinInfoGain: Double = $(minInfoGain)

  /**
    * Param for maximum number of iterations (&gt;= 0) of GBT.
    * @group param
    */
  val GBTMaxIter: IntParam = new IntParam(this, "GBTMaxIter",
    "maximum number of iterations (>= 0) of GBT",
    ParamValidators.gtEq(0))

  /** @group getParam */
  def getGBTMaxIter: Int = $(GBTMaxIter)

  /**
    * Param for Step size (a.k.a. learning rate) in interval (0, 1] for shrinking
    * the contribution of each estimator.
    *
    * (default = 0.1)
    * @group param
    */
  val stepSize: DoubleParam = new DoubleParam(this, "stepSize", "Step size " +
      "(a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of" +
      "each estimator.",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  /** @group getParam */
  def getStepSize: Double = $(stepSize)

  /**
    * Fraction of the training data used for learning each decision tree, in range (0, 1].
    *
    * (default = 1.0)
    * @group param
    */
  val subsamplingRate: DoubleParam = new DoubleParam(this, "subsamplingRate",
    "Fraction of the training data used for learning each decision tree, in range (0, 1].",
    ParamValidators.inRange(0, 1, lowerInclusive = false, upperInclusive = true))

  /** @group getParam */
  def getSubsamplingRate: Double = $(subsamplingRate)

  /**
    * Param for random seed.
    * @group param
    */
  val seed: LongParam = new LongParam(this, "seed", "random seed")

  /** @group getParam */
  def getSeed: Long = $(seed)

  // =====below are LR params=====

  /**
    * Param for the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0,
    * the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
    * @group param
    */
  val elasticNetParam: DoubleParam = new DoubleParam(this, "elasticNetParam",
    "the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is" +
        " an L2 penalty. For alpha = 1, it is an L1 penalty", ParamValidators.inRange(0, 1))

  /** @group getParam */
  def getElasticNetParam: Double = $(elasticNetParam)

  /**
    * Param for the name of family which is a description of the label distribution
    * to be used in the model.
    *
    * Supported options:
    *
    *  - "auto": Automatically select the family based on the number of classes:
    *            If numClasses == 1 || numClasses == 2, set to "binomial".
    *            Else, set to "multinomial"
    *
    *  - "binomial": Binary logistic regression with pivoting.
    *
    *  - "multinomial": Multinomial logistic (softmax) regression without pivoting.
    *
    * Default is "auto".
    *
    * @group param
    */
  val family: Param[String] = new Param(this, "family",
    "The name of family which is a description of the label distribution to be used in the " +
        s"model. Supported options: " +
        s"${Array("auto", "binomial", "multinomial").map(_.toLowerCase).mkString(", ")}.",
    ParamValidators.inArray[String](
      Array("auto", "binomial", "multinomial").map(_.toLowerCase)))

  /** @group getParam */
  def getFamily: String = $(family)

  /**
    * Param for whether to fit an intercept term.
    * @group param
    */
  val fitIntercept: BooleanParam = new BooleanParam(this, "fitIntercept",
    "whether to fit an intercept term")

  /** @group getParam */
  def getFitIntercept: Boolean = $(fitIntercept)

  /**
    * Param for maximum number of iterations (&gt;= 0) of LR.
    * @group param
    */
  val LRMaxIter: IntParam = new IntParam(this, "LRMaxIter",
    "maximum number of iterations (>= 0) of LR",
    ParamValidators.gtEq(0))

  /** @group getParam */
  def getLRMaxIter: Int = $(LRMaxIter)

  /**
    * Param for Column name for predicted class conditional probabilities.
    *
    * '''Note''': Not all models output well-calibrated probability estimates!
    *
    * These probabilities should be treated as confidences, not precise probabilities.
    *
    * @group param
    */
  val probabilityCol: Param[String] = new Param[String](this, "probabilityCol",
    "Column name for predicted class conditional probabilities. Note: Not all models output" +
        " well-calibrated probability estimates! These probabilities should be treated as" +
        " confidences, not precise probabilities")

  /** @group getParam */
  def getProbabilityCol: String = $(probabilityCol)

  /**
    * Param for raw prediction (a.k.a. confidence) column name.
    *
    * @group param
    */
  val rawPredictionCol: Param[String] = new Param[String](this, "rawPredictionCol",
    "raw prediction (a.k.a. confidence) column name")

  /** @group getParam */
  def getRawPredictionCol: String = $(rawPredictionCol)

  /**
    * Param for gbt generated features column name.
    *
    * @group param
    */
  val gbtGeneratedFeaturesCol: Param[String] = new Param[String](this, "gbtGeneratedCol",
    "gbt generated features column name")

  /** @group getParam */
  def getGbtGeneratedFeaturesCol: String = $(gbtGeneratedFeaturesCol)

  /**
    * Param for regularization parameter (&gt;= 0).
    * @group param
    */
  val regParam: DoubleParam = new DoubleParam(this, "regParam",
    "regularization parameter (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  def getRegParam: Double = $(regParam)

  /**
    * Param for whether to standardize the training features before fitting the model.
    * @group param
    */
  val standardization: BooleanParam = new BooleanParam(this, "standardization",
    "whether to standardize the training features before fitting the model")

  /** @group getParam */
  def getStandardization: Boolean = $(standardization)

  /**
    * Param for threshold in binary classification prediction, in range [0, 1].
    * @group param
    */
  val threshold: DoubleParam = new DoubleParam(this, "threshold",
    "threshold in binary classification prediction, in range [0, 1]",
    ParamValidators.inRange(0, 1))

  /**
    * Get threshold for binary classification.
    *
    * If `thresholds` is set with length 2 (i.e., binary classification),
    * this returns the equivalent threshold: {{{ 1 / (1 + thresholds(0) / thresholds(1)) }}}
    * Otherwise, returns `threshold` if set, or its default value if unset.
    *
    * @group getParam
    * @throws IllegalArgumentException if `thresholds` is set to an array of length other than 2.
    */
  def getThreshold: Double = {
    checkThresholdConsistency()
    if (isSet(thresholds)) {
      val ts = $(thresholds)
      require(ts.length == 2, "Logistic Regression getThreshold only applies to" +
          " binary classification, but thresholds has length != 2.  thresholds: " +
          ts.mkString(","))
      1.0 / (1.0 + ts(0) / ts(1))
    } else {
      $(threshold)
    }
  }

  /**
    * Param for Thresholds in multi-class classification to adjust the probability
    * of predicting each class. Array must have length equal to the number of classes,
    * with values > 0 excepting that at most one value may be 0. The class with largest
    * value p/t is predicted, where p is the original probability of that class and t is
    * the class's threshold.
    * @group param
    */
  val thresholds: DoubleArrayParam = new DoubleArrayParam(this, "thresholds",
    "Thresholds in multi-class classification to adjust the probability of predicting" +
        " each class. Array must have length equal to the number of classes, with values > 0" +
        " excepting that at most one value may be 0. The class with largest value p/t is" +
        " predicted, where p is the original probability of that class and t is the class's" +
        " threshold", (t: Array[Double]) => t.forall(_ >= 0) && t.count(_ == 0) <= 1)

  /**
    * Get thresholds for binary or multiclass classification.
    *
    * If `thresholds` is set, return its value.
    * Otherwise, if `threshold` is set, return the equivalent thresholds for binary
    * classification: (1-threshold, threshold).
    * If neither are set, throw an exception.
    *
    * @group getParam
    */
  def getThresholds: Array[Double] = {
    checkThresholdConsistency()
    if (!isSet(thresholds) && isSet(threshold)) {
      val t = $(threshold)
      Array(1-t, t)
    } else {
      $(thresholds)
    }
  }

  /**
    * Param for the convergence tolerance for iterative algorithms (&gt;= 0).
    * @group param
    */
  val tol: DoubleParam = new DoubleParam(this, "tol",
    "the convergence tolerance for iterative algorithms (>= 0)", ParamValidators.gtEq(0))

  /** @group getParam */
  def getTol: Double = $(tol)

  /**
    * Param for weight column name. If this is not set or empty, we treat all instance
    * weights as 1.0.
    * @group param
    */
  val weightCol: Param[String] = new Param[String](this, "weightCol",
    "weight column name. If this is not set or empty, we treat all instance weights as 1.0")

  /** @group getParam */
  def getWeightCol: String = $(weightCol)

  /**
    * Param for suggested depth for treeAggregate (&gt;= 2).
    * @group expertParam
    */
  val aggregationDepth: IntParam = new IntParam(this, "aggregationDepth",
    "suggested depth for treeAggregate (>= 2)", ParamValidators.gtEq(2))

  /** @group expertGetParam */
  def getAggregationDepth: Int = $(aggregationDepth)

  /**
    * If `threshold` and `thresholds` are both set, ensures they are consistent.
    *
    * @throws IllegalArgumentException if `threshold` and `thresholds` are not equivalent
    */
  private def checkThresholdConsistency(): Unit = {
    if (isSet(threshold) && isSet(thresholds)) {
      val ts = $(thresholds)
      require(ts.length == 2, "Logistic Regression found inconsistent values for threshold and" +
          s" thresholds.  Param threshold is set (${$(threshold)}), indicating binary" +
          s" classification, but Param thresholds is set with length ${ts.length}." +
          " Clear one Param value to fix this problem.")
      val t = 1.0 / (1.0 + ts(0) / ts(1))
      require(math.abs($(threshold) - t) < 1E-5, "Logistic Regression getThreshold found" +
          s" inconsistent values for threshold (${$(threshold)}) and thresholds (equivalent to $t)")
    }
  }

  setDefault(seed -> this.getClass.getName.hashCode.toLong,
    subsamplingRate -> 1.0, GBTMaxIter -> 20, stepSize -> 0.1, maxDepth -> 5, maxBins -> 32,
    minInstancesPerNode -> 1, minInfoGain -> 0.0, checkpointInterval -> 10, fitIntercept -> true,
    probabilityCol -> "probability", rawPredictionCol -> "rawPrediction", standardization -> true,
    threshold -> 0.5, lossType -> "logistic", cacheNodeIds -> false, maxMemoryInMB -> 256,
    regParam -> 0.0, elasticNetParam -> 0.0, family -> "auto", LRMaxIter -> 100, tol -> 1E-6,
    aggregationDepth -> 2, gbtGeneratedFeaturesCol -> "gbt_generated_features")
}


/**
  * GBTLRClassifier is a hybrid model of Gradient Boosting Trees and Logistic Regression.
  * Input features are transformed by means of boosted decision trees. The output of each individual tree is treated
  * as a categorical input feature to a sparse linear classifer. Boosted decision trees prove to be very powerful
  * feature transforms.
  *
  * Model details about GBTLR can be found in the following paper:
  * <a href="https://dl.acm.org/citation.cfm?id=2648589">Practical Lessons from Predicting Clicks on Ads at Facebook</a>
  *
  * GBTLRClassifier on Spark is designed and implemented by combining GradientBoostedTrees and Logistic Regressor in
  * Spark MLlib. Features are firstly trained and transformed into sparse vectors via GradientBoostedTrees, and then
  * the generated sparse features will be trained and predicted in Logistic Regression model.
  *
  * @param uid unique ID for Model
  */
class GBTLRClassifier (override val uid: String)
    extends Predictor[Vector, GBTLRClassifier, GBTLRClassificationModel]
        with GBTLRClassifierParams with DefaultParamsWritable {

  import GBTLRClassifier._
  import GBTLRUtil._

  def this() = this(Identifiable.randomUID("gbtlr"))

  // Set GBTClassifier params

  /** @group setParam */
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group setParam */
  def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group setParam */
  def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /** @group setParam */
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /**
    * The impurity setting is ignored for GBT models.
    * Individual trees are build using impurity "Variance."
    *
    * @group setParam
    */
  def setImpurity(value: String): this.type = {
    logger.warn("GBTClassifier in the GBTLRClassifier should NOT be used")
    this
  }

  /** @group setParam */
  def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setGBTMaxIter(value: Int): this.type = set(GBTMaxIter, value)

  /** @group setParam */
  def setStepSize(value: Double): this.type = set(stepSize, value)

  /** @group setParam */
  def setLossType(value: String): this.type = set(lossType, value)

  /** @group setParam */
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)

  /** @group setParam */
  def setFamily(value: String): this.type = set(family, value)

  /** @group setParam */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

  /** @group setParam */
  def setLRMaxIter(value: Int): this.type = set(LRMaxIter, value)

  /** @group setParam */
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)

  /** @group setParam */
  def setRawPredictionCol(value: String): this.type = set(rawPredictionCol, value)

  /** @group setParam */
  def setRegParam(value: Double): this.type = set(regParam, value)

  /** @group setParam */
  def setStandardization(value: Boolean): this.type = set(standardization, value)

  /** @group setParam */
  def setTol(value: Double): this.type = set(tol, value)

  /** @group setParam */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /** @group setParam */
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  /** @group setParam */
  def setGbtGeneratedFeaturesCol(value: String): this.type = set(gbtGeneratedFeaturesCol, value)

  /**
    * Set threshold in binary classification, in range [0, 1].
    *
    * If the estimated probability of class label 1 is greater than threshold, then predict 1,
    * else 0. A high threshold encourages the model to predict 0 more often;
    * a low threshold encourages the model to predict 1 more often.
    *
    * '''Note''': Calling this with threshold p is equivalent to calling
    * `setThresholds(Array(1-p, p))`.
    *       When `setThreshold()` is called, any user-set value for `thresholds` will be cleared.
    *       If both `threshold` and `thresholds` are set in a ParamMap, then they must be
    *       equivalent.
    *
    * Default is 0.5.
    *
    * @group setParam
    */
  // TODO: Implement SPARK-11543?
  def setThreshold(value: Double): this.type = {
    if (isSet(thresholds)) clear(thresholds)
    set(threshold, value)
  }

  /**
    * Set thresholds in multiclass (or binary) classification to adjust the probability of
    * predicting each class. Array must have length equal to the number of classes,
    * with values greater than 0, excepting that at most one value may be 0.
    * The class with largest value p/t is predicted, where p is the original probability of that
    * class and t is the class's threshold.
    *
    * '''Note''': When `setThresholds()` is called, any user-set value for `threshold`
    *       will be cleared.
    *       If both `threshold` and `thresholds` are set in a ParamMap, then they must be
    *       equivalent.
    *
    * @group setParam
    */
  def setThresholds(value: Array[Double]): this.type = {
    if (isSet(threshold)) clear(threshold)
    set(thresholds, value)
  }

  /**
    * Examine a schema to identify categorical (Binary and Nominal) features
    * @param featuresSchema Schema of the fetaures column.
    *
    *                       If a feature does not have metadata, it is assumed to be continuous.
    *
    *                       If a feature is Nominal, then it must have the number of values
    *                       specified.
    * @return Map: feature index to number of categories.
    *
    *         The map's set of keys will be the set of categorical feature indices.
    */
  private def getCategoricalFeatures(featuresSchema: StructField): Map[Int, Int] = {
    val metadata = AttributeGroup.fromStructField(featuresSchema)
    if (metadata.attributes.isEmpty) {
      HashMap.empty[Int, Int]
    } else {
      metadata.attributes.get.zipWithIndex.flatMap{ case (attr, idx) =>
        if (attr == null) {
          Iterator()
        } else {
          attr match {
            case _: NumericAttribute | UnresolvedAttribute => Iterator()
            case binAttr: BinaryAttribute => Iterator(idx -> 2)
            case nomAttr: NominalAttribute =>
              nomAttr.getNumValues match {
                case Some(numValues: Int) => Iterator(idx -> numValues)
                case None => throw new IllegalArgumentException(s"Feature $idx is " +
                    s"marked as Nominal (categorical), but it does not have the number" +
                    s" of values specified.")
              }
          }
        }
      }.toMap
    }
  }

  /**
    * Create a Strategy instance to use with the old API.
    * @param categoricalFeatures Map: feature index to number of categories.
    * @return Strategy instance
    */
  private def getOldStrategy(categoricalFeatures: Map[Int, Int]): OldStrategy = {
    val strategy = OldStrategy.defaultStrategy(OldAlgo.Classification)
    strategy.impurity = OldVariance
    strategy.checkpointInterval = getCheckpointInterval
    strategy.maxBins = getMaxBins
    strategy.maxDepth = getMaxDepth
    strategy.maxMemoryInMB = getMaxMemoryInMB
    strategy.minInfoGain = getMinInfoGain
    strategy.minInstancesPerNode = getMinInstancePerNode
    strategy.useNodeIdCache = getCacheNodeIds
    strategy.numClasses = 2
    strategy.categoricalFeaturesInfo = categoricalFeatures
    strategy.subsamplingRate = getSubsamplingRate
    strategy
  }

  /**
    * Get old Gradient Boosting Loss type
    * @return Loss type
    */
  private def getOldLossType: OldLoss = {
    getLossType match {
      case "logistic" => OldLogLoss
      case _ =>
        // Should never happen because of check in setter method.
        throw new RuntimeException(s"GBTClassifier was given bad loss type:" +
            s" $getLossType")
    }
  }

  /**
    * Train a GBTLRClassification Model which consists of GradientBoostedTreesModel
    * and LogisticRegressionModel.
    * @param dataset Input data.
    * @return GBTLRClassification model.
    */
  override def train(dataset: Dataset[_]): GBTLRClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      getCategoricalFeatures(dataset.schema($(featuresCol)))

    // GBT only supports 2 classes now.
    val oldDataset: RDD[OldLabeledPoint] =
      dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          require(label == 0 || label == 1, s"GBTClassifier was given" +
              s" dataset with invalid label $label.  Labels must be in {0,1}; note that" +
              s" GBTClassifier currently only supports binary classification.")
          OldLabeledPoint(label, new OldDenseVector(features.toArray))
      }

    val numFeatures = oldDataset.first().features.size
    val strategy = getOldStrategy(categoricalFeatures)
    val boostingStrategy = new OldBoostingStrategy(strategy, getOldLossType,
      getGBTMaxIter, getStepSize)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(params: _*)
    instr.logNumFeatures(numFeatures)
    instr.logNumClasses(2)

    // train a gradient boosted tree model using boostingStrategy.
    val gbtModel = GradientBoostedTrees.train(oldDataset, boostingStrategy)

    // udf for creating a feature column which consists of original features
    // and gbt model generated features.
    val addFeatureUDF = udf { (features: Vector) =>
      val gbtFeatures = getGBTFeatures(gbtModel, features)
      Vectors.dense(features.toArray ++ gbtFeatures.toArray)
    }

    val datasetWithCombinedFeatures = dataset.withColumn($(gbtGeneratedFeaturesCol),
      addFeatureUDF(col($(featuresCol))))

    // create a logistic regression instance.
    val logisticRegression = new LogisticRegression()
        .setRegParam($(regParam))
        .setElasticNetParam($(elasticNetParam))
        .setMaxIter($(LRMaxIter))
        .setTol($(tol))
        .setLabelCol($(labelCol))
        .setFeaturesCol($(featuresCol))
        .setFitIntercept($(fitIntercept))
        .setFamily($(family))
        .setStandardization($(standardization))
        .setPredictionCol($(predictionCol))
        .setProbabilityCol($(probabilityCol))
        .setRawPredictionCol($(rawPredictionCol))
        .setAggregationDepth($(aggregationDepth))
        .setFeaturesCol($(gbtGeneratedFeaturesCol))

    if (isSet(weightCol)) logisticRegression.setWeightCol($(weightCol))
    if (isSet(threshold)) logisticRegression.setThreshold($(threshold))
    if (isSet(thresholds)) logisticRegression.setThresholds($(thresholds))

    // train a logistic regression model with new combined features.
    val lrModel = logisticRegression.fit(datasetWithCombinedFeatures)

    val model = copyValues(new GBTLRClassificationModel(uid, gbtModel, lrModel).setParent(this))
    val summary = new GBTLRClassifierTrainingSummary(datasetWithCombinedFeatures, lrModel.summary,
      gbtModel.trees, gbtModel.treeWeights)
    model.setSummary(Some(summary))
    instr.logSuccess(model)
    model
  }

  override def copy(extra: ParamMap): GBTLRClassifier = defaultCopy(extra)
}

object GBTLRClassifier extends DefaultParamsReadable[GBTLRClassifier] {

  val logger = Logger.getLogger(GBTLRClassifier.getClass)

  override def load(path: String): GBTLRClassifier = super.load(path)
}

class GBTLRClassificationModel (
    override val uid: String,
    val gbtModel: GradientBoostedTreesModel,
    val lrModel: LogisticRegressionModel)
    extends PredictionModel[Vector, GBTLRClassificationModel]
        with GBTLRClassifierParams with MLWritable {

  import GBTLRUtil._

  private var trainingSummary: Option[GBTLRClassifierTrainingSummary] = None

  private[gbtlr] def setSummary(
      summary: Option[GBTLRClassifierTrainingSummary]): this.type = {
    this.trainingSummary = summary
    this
  }

  /**
    * Return true if there exists summary of model
    */
  def hasSummary: Boolean = trainingSummary.nonEmpty

  def summary: GBTLRClassifierTrainingSummary = trainingSummary.getOrElse {
    throw new SparkException(
      s"No training summary available for the ${this.getClass.getSimpleName}"
    )
  }

  override def write: MLWriter =
    new GBTLRClassificationModel.GBTLRClassificationModelWriter(this)

  /**
    * Get a combined feature point through gbdt when given a specific feature point.
    * @param point Original one point.
    * @return A combined feature point.
    */
  def getComibinedFeatures(
      point: OldLabeledPoint): OldLabeledPoint = {
    val numTrees = gbtModel.trees.length
    val treeLeafArray = new Array[Array[Int]](numTrees)
    for (i <- 0 until numTrees)
      treeLeafArray(i) = getLeafNodes(gbtModel.trees(i).topNode)

    var newFeature = new Array[Double](0)
    val label = point.label
    val features = point.features
    for (i <- 0 until numTrees) {
      val treePredict = predictModify(gbtModel.trees(i).topNode, features.toDense)
      val treeArray = new Array[Double]((gbtModel.trees(i).numNodes + 1) / 2)
      treeArray(treeLeafArray(i).indexOf(treePredict)) = 1
      newFeature = newFeature ++ treeArray
    }
    OldLabeledPoint(label.toInt, new OldDenseVector(features.toArray ++ newFeature))
  }

  // udf for creating a feature column which consists of original features
  // and gbt model generated features.
  private val addFeatureUDF = udf { (features: Vector) =>
    val gbtFeatures = getGBTFeatures(gbtModel, features)
    Vectors.dense(features.toArray ++ gbtFeatures.toArray)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val datasetWithCombinedFeatures = dataset.withColumn($(gbtGeneratedFeaturesCol),
      addFeatureUDF(col($(featuresCol))))
    val predictions = lrModel.transform(datasetWithCombinedFeatures)
    predictions
  }

  // just implements the abstract method in PredictionModel, but is not used.
  override def predict(features: Vector): Double = 0.0

  /**
    * Evaluations the model on a test dataset.
    * @param dataset Test dataset to evalute model on.
    */
  def evaluate(dataset: Dataset[_]): GBTLRClassifierSummary = {
    val datasetWithCombinedFeatures = dataset.withColumn($(gbtGeneratedFeaturesCol),
      addFeatureUDF(col($(featuresCol))))
    val lrSummary = lrModel.evaluate(datasetWithCombinedFeatures)
    new GBTLRClassifierSummary(lrSummary)
  }

  override def copy(extra: ParamMap): GBTLRClassificationModel = {
    val copied = copyValues(new GBTLRClassificationModel(uid, gbtModel, lrModel), extra)
    copied.setSummary(trainingSummary).setParent(this.parent)
  }

  /**
    * Get a set of rules which can reach the different leaf nodes.
    * @param node Root node.
    * @param rule Current set of rules
    * @param rules Final set of rules of all leaf node.
    */
  private def getLeafRules(
      node: OldNode,
      rule: String,
      rules: mutable.ArrayBuilder[String]) {
    val split = node.split
    if (node.isLeaf) {
      rules += rule
    } else {
      if (split.get.featureType == FeatureType.Continuous) {
        val leftRule = rule + s", feature#${split.get.feature} < ${split.get.threshold}"
        getLeafRules(node.leftNode.get, leftRule, rules)
        val rightRule = rule + s", feature#${split.get.feature} > ${split.get.threshold}"
        getLeafRules(node.rightNode.get, rightRule, rules)
      } else {
        val leftRule = rule + s", feature#${split.get.feature}'s value is in the Set" +
            split.get.categories.mkString("[", ",", "]")
        getLeafRules(node.leftNode.get, leftRule, rules)
        val rightRule = rule + s", feature#${split.get.feature}'s value is not in the Set" +
            split.get.categories.mkString("[", ",", "]")
        getLeafRules(node.rightNode.get, rightRule, rules)
      }
    }
  }

  /**
    * Get a description of each dimension of extra feature with a trained weight through lr.
    * @return An array of tuple2, in each tuple, the first elem indicates the weight of extra
    *         feature, the second elem is the description of how to get this feature.
    */
  def getRules: Array[Tuple2[Double, String]] = {
    val numTrees = gbtModel.trees.length
    val rules = new Array[Array[String]](numTrees)
    var numExtraFeatures = 0
    for (i <- 0 until numTrees) {
      val rulesInEachTree = mutable.ArrayBuilder.make[String]
      getLeafRules(gbtModel.trees(i).topNode, "", rulesInEachTree)
      val rule = rulesInEachTree.result()
      numExtraFeatures += rule.length
      rules(i) = rule
    }
    val weightsInLR = lrModel.coefficients.toArray
    val extraWeights =
      weightsInLR.slice(weightsInLR.length - numExtraFeatures, weightsInLR.length)
    extraWeights.zip(rules.flatMap(x => x))
  }

}

object GBTLRClassificationModel extends MLReadable[GBTLRClassificationModel] {

  val logger = Logger.getLogger(GBTLRClassificationModel.getClass)


  override def read: MLReader[GBTLRClassificationModel] = new GBTLRClassificationModelReader

  override def load(path: String): GBTLRClassificationModel = super.load(path)

  private[GBTLRClassificationModel] class GBTLRClassificationModelWriter(
      instance: GBTLRClassificationModel) extends MLWriter {
    override def saveImpl(path: String) {
      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data
      val gbtDataPath = new Path(path, "gbtData").toString
      instance.gbtModel.save(sc, gbtDataPath)
      val lrDataPath = new Path(path, "lrData").toString
      instance.lrModel.save(lrDataPath)
    }
  }

  private class GBTLRClassificationModelReader
      extends MLReader[GBTLRClassificationModel] {

    private val className = classOf[GBTLRClassificationModel].getName

    override def load(path: String): GBTLRClassificationModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val gbtDataPath = new Path(path, "gbtData").toString
      val lrDataPath = new Path(path, "lrData").toString
      val gbtModel = GradientBoostedTreesModel.load(sc, gbtDataPath)
      val lrModel = LogisticRegressionModel.load(lrDataPath)
      val model = new GBTLRClassificationModel(metadata.uid, gbtModel, lrModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}

class GBTLRClassifierTrainingSummary (
    @transient val newDataset: DataFrame,
    val logRegSummary: LogisticRegressionTrainingSummary,
    val gbtTrees: Array[DecisionTreeModel],
    val treeWeights: Array[Double]) extends Serializable {
}

class GBTLRClassifierSummary (
    val binaryLogisticRegressionSummary: LogisticRegressionSummary)
    extends Serializable {
}


object GBTLRUtil {
  /**
    * Get an array of leaf nodes according to the root node of a tree.
    * The order of nodes in the array is from left to right.
    *
    * @param node Root node of a tree.
    * @return An array stores the leaf node ids.
    */
  def getLeafNodes(node: OldNode): Array[Int] = {
    var treeLeafNodes = new Array[Int](0)
    if (node.isLeaf) {
      treeLeafNodes = treeLeafNodes :+ (node.id)
    } else {
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.leftNode.get)
      treeLeafNodes = treeLeafNodes ++ getLeafNodes(node.rightNode.get)
    }
    treeLeafNodes
  }

  /**
    * Get the leaf node id at which the features will be located.
    *
    * @param node Root node of a tree.
    * @param features Dense Vector features.
    * @return Leaf node id.
    */
  def predictModify(node: OldNode, features: OldDenseVector): Int = {
    val split = node.split
    if (node.isLeaf) {
      node.id
    } else {
      if (split.get.featureType == FeatureType.Continuous) {
        if (features(split.get.feature) <= split.get.threshold) {
          predictModify(node.leftNode.get, features)
        } else {
          predictModify(node.rightNode.get, features)
        }
      } else {
        if (split.get.categories.contains(features(split.get.feature))) {
          predictModify(node.leftNode.get, features)
        } else {
          predictModify(node.rightNode.get, features)
        }
      }
    }
  }

  /**
    *Get GBT generated features from gbt model
    *
    * @param gbtModel
    * @param features
    * @return
    */
  def getGBTFeatures(gbtModel: GradientBoostedTreesModel, features: Vector): Vector = {
    val GBTMaxIter = gbtModel.trees.length
    val oldFeatures = new OldDenseVector(features.toArray)
    val treeLeafArray = new Array[Array[Int]](GBTMaxIter)
    for (i <- 0 until GBTMaxIter)
      treeLeafArray(i) = getLeafNodes(gbtModel.trees(i).topNode)
    var newFeature = new Array[Double](0)
    for (i <- 0 until GBTMaxIter) {
      val treePredict = predictModify(gbtModel.trees(i).topNode, oldFeatures.toDense)
      val treeArray = new Array[Double]((gbtModel.trees(i).numNodes + 1) / 2)
      treeArray(treeLeafArray(i).indexOf(treePredict)) = 1
      newFeature = newFeature ++ treeArray
    }
    Vectors.dense(newFeature)
  }
}
