package org.apache.spark.ml.made

import org.scalatest.flatspec._
import org.scalatest.matchers._

import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.stats.mean


object LinearRegressionTest extends WithSpark {

  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand(100000, 3)
  lazy val _weights: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
  lazy val _bias: Double = 1.0
  lazy val _y: DenseVector[Double] = _X * _weights + _bias + DenseVector.rand(100000) * 0.001 //
  lazy val _df: DataFrame = {
    import sqlc.implicits._

    val tmp = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)
    val df = tmp(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x_1", "x_2", "x_3", "label")

    val matrix_assembler = new VectorAssembler()
      .setInputCols(Array("x_1", "x_2", "x_3"))
      .setOutputCol("features")

    val out = matrix_assembler.transform(df).select("features", "label")

    out
  }

}

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {


  val delta: Double = 0.01
  val weights: DenseVector[Double] = LinearRegressionTest._weights
  val bias: Double = LinearRegressionTest._bias
  val y_true: DenseVector[Double] = LinearRegressionTest._y
  val df: DataFrame = LinearRegressionTest._df

  "Estimator" should "should produce functional model" in {
    val estimator: LinearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMaxIter(200)
      .setStepSize(1.0)


    val model = estimator.fit(df)

    model.weights.size should be (3)
    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(1) +- delta)
    model.weights(2) should be(weights(2) +- delta)
    model.bias should be(bias +- delta)
  }

  "Model" should "should predict right" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(weights).toDense,
      bias = bias
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val pred = DenseVector(model.transform(df).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))

    sqrt(mean(pow(pred - y_true, 2))) should be(0.0 +- delta)
  }

  "Estimator" should "should work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMaxIter(200)
        .setStepSize(1.0)
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(df).stages(0).asInstanceOf[LinearRegressionModel]

    model.weights.size should be (3)
    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(1) +- delta)
    model.weights(2) should be(weights(2) +- delta)
    model.bias should be(bias +- delta)
  }

  "Model" should "should work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMaxIter(200)
        .setStepSize(1.0)
    ))

    val model = pipeline.fit(df)
    val weights = model.stages(0).asInstanceOf[LinearRegressionModel].weights
    val bias = model.stages(0).asInstanceOf[LinearRegressionModel].bias

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val loaded_model = PipelineModel.load(tmpFolder.getAbsolutePath).stages(0).asInstanceOf[LinearRegressionModel]

    loaded_model.bias should be(bias +- delta)
    loaded_model.weights(0) should be(weights(0) +- delta)
    loaded_model.weights(1) should be(weights(1) +- delta)
    loaded_model.weights(2) should be(weights(2) +- delta)

    val pred = DenseVector(loaded_model.transform(df).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))
    sqrt(mean(pow(pred - y_true, 2))) should be(0.0 +- delta)
  }
}

