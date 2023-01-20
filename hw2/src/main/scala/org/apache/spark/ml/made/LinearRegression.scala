package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import breeze.linalg.{sum, DenseVector => BreezeDenseVector}

trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol
  with HasPredictionCol with HasMaxIter with HasStepSize {
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)
  setDefault(maxIter -> 500, stepSize -> 0.01)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val ones = dataset.withColumn("bias", lit(1)) //add bias

    val matrix_assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), "bias", $(labelCol)))
      .setOutputCol("features_out")

    val vectors: Dataset[Vector] = matrix_assembler
      .transform(ones)
      .select("features_out").as[Vector]

    val lenFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(featuresCol))

    var weights: BreezeDenseVector[Double] = BreezeDenseVector.rand[Double](lenFeatures + 1)

    for (i <- 0 until $(maxIter)) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(vec => {
          val X = vec.asBreeze(0 until weights.size).toDenseVector // features
          val y = vec.asBreeze(-1)
          val y_curr = sum(X * weights)
          val grads = X * (y_curr - y)
          summarizer.add(fromBreeze(grads))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      weights = weights - $(stepSize) * summary.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(weights(0 until weights.size - 1)).toDense,
      weights(weights.size - 1)
    )).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String, val weights: DenseVector, val bias: Double)
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(
      uid + "_transform",
      (x: Vector) => {
        Vectors.fromBreeze(BreezeDenseVector(weights.asBreeze.dot(x.asBreeze) + bias))
      })

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) = weights.asInstanceOf[Vector] -> Vectors.fromBreeze(BreezeDenseVector(bias))

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val weights = vectors.select(vectors("_1").as[Vector]).first()
      val bias = vectors.select(vectors("_2").as[Vector]).first()(0)

      val model = new LinearRegressionModel(weights.toDense, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}