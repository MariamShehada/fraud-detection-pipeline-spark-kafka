import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// SQLTransformer + Imputer to put preprocessing inside Pipeline stages
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.feature.Imputer

object OfflineFraudTraining {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("OfflineFraudTraining")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    //===================== Read the file ===============================
    val dataPath = "data/financial_fraud_detection_dataset.CSV"

    // Define schema
    val schema = new StructType()
      .add("transaction_id", StringType)
      .add("timestamp", StringType)
      .add("sender_account", StringType)
      .add("receiver_account", StringType)
      .add("amount", DoubleType)
      .add("transaction_type", StringType)
      .add("merchant_category", StringType)
      .add("location", StringType)
      .add("device_used", StringType)
      .add("is_fraud", BooleanType)
      .add("fraud_type", StringType)
      .add("time_since_last_transaction", DoubleType)
      .add("spending_deviation_score", DoubleType)
      .add("velocity_score", IntegerType)
      .add("geo_anomaly_score", DoubleType)
      .add("payment_channel", StringType)
      .add("ip_address", StringType)
      .add("device_hash", StringType)

    val rawFull = spark.read
      .option("header", "true")
      .schema(schema)
      .csv(dataPath)

    //----------------------------------------------
    // ====================== Build Bloom filter for bad senders using all data ==============================
    //----------------------------------------------

    /*
    create a Bloom filter in each Spark partition, fill it with sender accounts from that partition, and then merge all
     those partition-level filters into one final Bloom filter (mergedBF) that knows about all bad senders.
     spark DataFrames do not let build one Bloom filter per partition, but RDDs do (Gives full control of each partition)
    */
    import org.apache.spark.util.sketch.BloomFilter
    import org.apache.spark.sql.functions._
    import spark.implicits._

    // =====================================
    // 1) Build badSenders from fraud rows
    // =====================================
    val totalTx = rawFull.count()

    val fraudDF = rawFull.filter($"is_fraud" === true)
    val totalFraudTx = fraudDF.count()

    val badSendersDF = fraudDF
      .select($"sender_account")
      .na.drop()
      .distinct()

    val distinctBadSenders = badSendersDF.count()

    println(s"Total transactions              = $totalTx")
    println(s"Total fraud transactions         = $totalFraudTx")
    println(s"Distinct bad senders (fraud)     = $distinctBadSenders")

    badSendersDF.show(20, false)

    // =====================================
    // 2) Build Bloom Filter (CORRECT)
    // =====================================
    val fpp = 0.01

    val mergedBF =
      badSendersDF.rdd.mapPartitions { rows =>
        val bf = BloomFilter.create(distinctBadSenders, fpp)
        rows.foreach { row =>
          val acc = row.getString(0)
          if (acc != null) bf.putString(acc)
        }
        Iterator(bf)
      }.treeReduce { (bf1, bf2) =>
        bf1.mergeInPlace(bf2)
      }

    println(s"Bad senders inserted into Bloom  = $distinctBadSenders")
  /*
    // =====================================
    // 3) Broadcast + UDF
    // =====================================
    val bfBc = spark.sparkContext.broadcast(mergedBF)
    val bloomHit = udf((s: String) => if (s == null) false else bfBc.value.mightContainString(s))

    // =====================================
    // 4) FN check (should be 0)
    // Fraud senders must always hit Bloom
    // =====================================
    val fnDF = badSendersDF
      .withColumn("bf_hit", bloomHit($"sender_account"))
      .filter(!$"bf_hit")

    val fn = fnDF.count()
    println(s"False negatives (should be 0) = $fn")
    fnDF.show(20, false)

    // =====================================
    // 5) FP check on CLEAN senders
    // "Clean senders" = senders that NEVER did fraud
    // =====================================
    val cleanSendersDF = rawFull
      .filter($"is_fraud" === false)
      .select($"sender_account")
      .na.drop()
      .distinct()
      .join(badSendersDF, Seq("sender_account"), "left_anti")

    val totalClean = cleanSendersDF.count()

    val fp = cleanSendersDF
      .withColumn("bf_hit", bloomHit($"sender_account"))
      .filter($"bf_hit")
      .count()

    val fpRate = if (totalClean == 0) 0.0 else fp.toDouble / totalClean.toDouble

    println(s"Clean senders count                 = $totalClean")
    println(s"False positives count               = $fp")
    println(f"False positive rate (expected ~0.01) = $fpRate%.4f")

    // =====================================
    // 6) How many transactions Bloom flags
    // =====================================
    val txFlagged = rawFull
      .withColumn("bf_hit", bloomHit($"sender_account"))
      .filter($"bf_hit")
      .count()

    val flaggedRatio = txFlagged.toDouble / totalTx.toDouble

    println(s"Transactions flagged by Bloom  = $txFlagged")
    println(f"Ratio flagged by Bloom         = $flaggedRatio%.4f")


   */
    // =====================================
    // 7) Save Bloom filter
    // =====================================
    val bfPath = "data/blacklist_senders.bf"
    val out = new java.io.FileOutputStream(bfPath)
    mergedBF.writeTo(out)
    out.close()

    println(s"Bloom filter saved to: $bfPath")








    // =========================== preprocessing and feature engineering======================================
    // handel imbalance using resampling, keep all fraud, only 20% of non-fraud
    val fractions = Map(false -> 0.2, true -> 1.0)
    val raw = rawFull.stat.sampleBy("is_fraud", fractions, seed = 42L)
    println(s"Loaded ${raw.count()} rows (sample)")

    // drop fraud_type
    val dropped = raw.drop("fraud_type")

    // cast numeric + label types
    // (training needs label as int; streaming inference doesn't require label)
    val typed = dropped.withColumn("is_fraud", $"is_fraud".cast("int"))

    // ================== Feature engineering ==================
    //
    // We put feature engineering inside a Spark ML Pipeline stage (SQLTransformer)
    // so that the SAME transformations are reused later in streaming.
    // When we save the PipelineModel, these SQL transformations are saved with it.
    //

    val fe = new SQLTransformer().setStatement(
      """
  SELECT *,
    -- Convert the raw string timestamp into a Spark Timestamp type
    -- This is required for time-based features and windowing
    to_timestamp(timestamp, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS") AS event_time,

    -- Extract hour of day (0–23) from the timestamp
    -- Fraud often has time-of-day patterns
    hour(to_timestamp(timestamp, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")) AS hour_of_day,

    -- Binary flag: transaction happened at night (midnight–5am)
    CASE
      WHEN hour(to_timestamp(timestamp, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")) BETWEEN 0 AND 5
      THEN 1 ELSE 0
    END AS is_night,

    -- Binary flag: transaction happened on weekend
    -- Spark: dayofweek() → 1 = Sunday, 7 = Saturday
    CASE
      WHEN dayofweek(to_timestamp(timestamp, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")) IN (1, 7)
      THEN 1 ELSE 0
    END AS is_weekend,

    -- Binary flag: transaction done via online channel
    CASE
      WHEN payment_channel = "online"
      THEN 1 ELSE 0
    END AS is_online_channel,

    -- Binary flag: merchant category considered high-risk
    CASE
      WHEN merchant_category IN ("online", "travel", "entertainment")
      THEN 1 ELSE 0
    END AS high_risk_merchant,

    -- Binary flag: large transaction amount
    CASE
      WHEN amount > 1000.0
      THEN 1 ELSE 0
    END AS is_large_amount,

    -- Binary flag: sender and receiver are the same account
    -- This can indicate suspicious self-transfer behavior
    CASE
      WHEN sender_account = receiver_account
      THEN 1 ELSE 0
    END AS sender_equals_receiver

  -- __THIS__ means: apply this SQL to the incoming DataFrame
  FROM __THIS__
  """
    )
    // ================== Missing-value handling ==================
    //
    // Imputer is also placed inside the Pipeline so that:
    // - it learns statistics (median) during training
    // - the SAME logic is applied later in streaming
    //
    // This prevents null values from breaking the ML model
    //

    val imputer = new Imputer()
      // Column that may contain null values
      .setInputCols(Array("time_since_last_transaction"))

      // Output column (same name = overwrite nulls)
      .setOutputCols(Array("time_since_last_transaction"))

      // Strategy: replace nulls with the MEDIAN value learned from training data
      .setStrategy("median")


    // ================== Feature columns ==================
    // Define feature columns
    val categoricalCols = Array(
      "transaction_type",
      "merchant_category",
      "location",
      "device_used",
      "payment_channel"
    )

    val numericCols = Array(
      "amount",
      "time_since_last_transaction",
      "spending_deviation_score",
      "velocity_score",
      "geo_anomaly_score",
      "hour_of_day",
      "is_night",
      "is_weekend",
      "is_online_channel",
      "high_risk_merchant",
      "is_large_amount",
      "sender_equals_receiver"
    )

    // index categoricals
    // ML models cannot work with text categories, so we convert categorical
    // columns into numeric category IDs using StringIndexer.
    // Example: "online" → 0.0, "travel" → 1.0, etc.
    val indexers = categoricalCols.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(s"${colName}_idx")
        .setHandleInvalid("keep")
    }





    // creates an array containing the names of these indexed columns
    val categoricalIdxCols = categoricalCols.map(c => s"${c}_idx")

    // Assemble all features into one vector
    val assembler = new VectorAssembler()
      .setInputCols(numericCols ++ categoricalIdxCols)
      .setOutputCol("features")
      .setHandleInvalid("skip")   // extra safety, will drop any bad rows (including null time features)

    // ============================== Random Forest model ========================================
    val rf = new RandomForestClassifier()
      .setLabelCol("is_fraud")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(10)
      .setSubsamplingRate(0.7)
      .setFeatureSubsetStrategy("sqrt")
      .setSeed(42L)

    // Build full Pipeline
    // The Pipeline now contains feature engineering + null handling + indexers + assembler + model.
    // After saving this model, streaming inference only needs raw columns (same schema) and can call model.transform(streamDF).
    /*
    first create engineered columns (fe)
    then handle nulls (imputer)
    hen index categories (indexers)
    then assemble features (assembler)
    then predict (rf)
     */

    val pipeline = new Pipeline()
      .setStages(Array(fe, imputer) ++ indexers ++ Array(assembler, rf))

    // Train / test split

    val prepared = typed.na.drop(Seq("is_fraud"))
    val Array(train, test) = prepared.randomSplit(Array(0.8, 0.2), seed = 42L)
    println(s"Train count = ${train.count()}, Test count = ${test.count()}")

    // Fit model
    val model = pipeline.fit(train)

    // Evaluate on test data
    import org.apache.spark.ml.functions.vector_to_array

    val predictions = model.transform(test)

    // =====================
    // Custom threshold
    // =====================
    val threshold = 0.15 // try: 0.05, 0.10, 0.15, 0.20

    val scored = predictions
      .withColumn(
        "fraud_probability",
        vector_to_array(col("probability"))(1)
      ) //extracts index 1 from the probability vector probability = [0.92, 0.08] then fraud_probability = 0.08
      .withColumn(
        "prediction_custom",
        when(col("fraud_probability") >= threshold, 1.0).otherwise(0.0)
      )

    scored
      .select("is_fraud", "fraud_probability", "prediction", "prediction_custom")
      .show(10, truncate = false)

    // =====================
    // AUC ROC / AUPRC (keep using rawPrediction)
    // =====================
    val evaluatorROC = new BinaryClassificationEvaluator()
      .setLabelCol("is_fraud")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val aucROC = evaluatorROC.evaluate(predictions)
    println(f"AUC ROC = $aucROC%1.4f")

    val evaluatorPR = new BinaryClassificationEvaluator()
      .setLabelCol("is_fraud")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderPR")

    val aucPR = evaluatorPR.evaluate(predictions)
    println(f"AUPRC   = $aucPR%1.4f")

    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    // ======================================================
    // Overall Evaluation Metrics
    // ======================================================

    // scored: DataFrame that already contains
    // - prediction_custom
    // - is_fraud (label)

    val evalDF = scored
      .select(
        col("prediction_custom").cast("double").alias("prediction"),
        col("is_fraud").cast("double").alias("label")
      )

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val accuracy = evaluator
      .setMetricName("accuracy")
      .evaluate(evalDF)

    val weightedPrecision = evaluator
      .setMetricName("weightedPrecision")
      .evaluate(evalDF)

    val weightedRecall = evaluator
      .setMetricName("weightedRecall")
      .evaluate(evalDF)

    val weightedF1 = evaluator
      .setMetricName("weightedFMeasure")
      .evaluate(evalDF)

    // =====================
    // Print results
    // =====================
    println(f"Threshold used = $threshold%1.2f")

    println("============== Overall metrics ================")
    println(f"Accuracy           : $accuracy%1.4f")
    println(f"Weighted Precision : $weightedPrecision%1.4f")
    println(f"Weighted Recall    : $weightedRecall%1.4f")
    println(f"Weighted F1        : $weightedF1%1.4f")

    // ================== Save the trained model ==================
    val modelPath = "data/fraud_lr_model"   // path name is historical; it's RF now
    model.write.overwrite().save(modelPath)

    println(s"Saved model to: $modelPath")

    spark.stop()
  }
}













/*
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics




object OfflineFraudTraining {

  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder()
      .appName("OfflineFraudTraining")
      .master("local[*]")
      .getOrCreate()


    spark.sparkContext.setLogLevel("WARN")

    //===================== Read the file ===============================
    val dataPath = "data/financial_fraud_detection_dataset.CSV"

    // Define schema
    val schema = new StructType()
      .add("transaction_id", StringType)
      .add("timestamp", StringType)
      .add("sender_account", StringType)
      .add("receiver_account", StringType)
      .add("amount", DoubleType)
      .add("transaction_type", StringType)
      .add("merchant_category", StringType)
      .add("location", StringType)
      .add("device_used", StringType)
      .add("is_fraud", BooleanType)
      .add("fraud_type", StringType)
      .add("time_since_last_transaction", DoubleType)
      .add("spending_deviation_score", DoubleType)
      .add("velocity_score", IntegerType)
      .add("geo_anomaly_score", DoubleType)
      .add("payment_channel", StringType)
      .add("ip_address", StringType)
      .add("device_hash", StringType)


    val rawFull = spark.read
      .option("header", "true")
      .schema(schema)
      .csv(dataPath)


    //----------------------------------------------
    // ====================== Build Bloom filter for bad senders using all data ==============================
    //----------------------------------------------

    import org.apache.spark.util.sketch.BloomFilter
    import spark.implicits._

    // All fraudulent rows from FULL dataset
    val fraudFull = rawFull.filter($"is_fraud" === true)

    // Distinct fraudulent sender accounts
    val badSenders = fraudFull
      .select("sender_account")
      .na.drop()
      .distinct()

    val approxCount = badSenders.count()
    println(s"Building Bloom filter with ~$approxCount fraudulent senders (full data)")

    val fpp = 0.01

    /*
    create a Bloom filter in each Spark partition, fill it with sender accounts from that partition, and then merge all
     those partition-level filters into one final Bloom filter (mergedBF) that knows about all bad senders.
     spark DataFrames do not let build one Bloom filter per partition, but RDDs do (Gives full control of each partition)
    */

    // Build Bloom filters per partition and merge them
    val mergedBF = badSenders.rdd.mapPartitions { rows =>
      val bf = BloomFilter.create(approxCount, fpp)
      rows.foreach { row =>
        val acc = row.getString(0)
        if (acc != null) bf.putString(acc)
      }
      Iterator(bf)
    }.treeReduce { (bf1, bf2) =>
      bf1.mergeInPlace(bf2)
    }

    // Save to file
    val bfPath = "data/blacklist_senders.bf"
    val out = new java.io.FileOutputStream(bfPath)
    mergedBF.writeTo(out)
    out.close()

    println(s"Bloom filter saved to: $bfPath")






    // =========================== preprocessing and feature engineering======================================
    // handel imbalance using resampling, keep all fraud, only 20% of non-fraud
    val fractions = Map(false -> 0.2, true -> 1.0)
    val raw = rawFull.stat.sampleBy("is_fraud", fractions, seed = 42L)
    println(s"Loaded ${raw.count()} rows (sample)")


    // drop fraud_type
    val dropped = raw.drop("fraud_type")
    // fill missing time_since_last_transaction with 0.0
    val filled = dropped.na.fill(Map(
      "time_since_last_transaction" -> 0.0
    ))
    // cast numeric + label types
    val typed = filled.withColumn("is_fraud", $"is_fraud".cast("int"))



    // ================== Feature engineering ==================
    val enriched1 = typed
      .withColumn(
        "event_time",
        to_timestamp($"timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")
      )
      .withColumn("hour_of_day", hour($"event_time"))
      .withColumn("is_night", $"hour_of_day".between(0, 5).cast("int"))
      .withColumn("is_weekend", dayofweek($"event_time").isin(1, 7).cast("int"))
      .withColumn("is_online_channel", ($"payment_channel" === "online").cast("int"))
      .withColumn(
        "high_risk_merchant",
        $"merchant_category".isin("online", "travel", "entertainment").cast("int")
      )


    // simple extra features
    val enriched2 = enriched1
      .withColumn("is_large_amount", ($"amount" > 1000.0).cast("int"))
      .withColumn(
        "sender_equals_receiver",
        ($"sender_account" === $"receiver_account").cast("int")
      )

    // Drop the few rows where derived time features are null + ensure label is not null
    val enriched = enriched2
      .filter($"hour_of_day".isNotNull && $"is_night".isNotNull && $"is_weekend".isNotNull)
      .na.drop(Seq("is_fraud"))





    // Define feature columns
    val categoricalCols = Array(
      "transaction_type",
      "merchant_category",
      "location",
      "device_used",
      "payment_channel"
    )

    val numericCols = Array(
      "amount",
      "time_since_last_transaction",
      "spending_deviation_score",
      "velocity_score",
      "geo_anomaly_score",
      "hour_of_day",
      "is_night",
      "is_weekend",
      "is_online_channel",
      "high_risk_merchant",
      "is_large_amount",
      "sender_equals_receiver"
    )



    // index categoricals
    // ML models cannot work with text categories, so we convert categorical
    // columns into numeric category IDs using StringIndexer.
    // Example: "online" → 0.0, "travel" → 1.0, etc.
    val indexers = categoricalCols.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(s"${colName}_idx")
        .setHandleInvalid("keep")
    }

    // creates an array containing the names of these indexed columns
    val categoricalIdxCols = categoricalCols.map(c => s"${c}_idx")

    // Assemble all features into one vector
    val assembler = new VectorAssembler()
      .setInputCols(numericCols ++ categoricalIdxCols)
      .setOutputCol("features")
      .setHandleInvalid("skip")   // extra safety, will drop any bad rows

    // ============================== Random Forest model ========================================
    val rf = new RandomForestClassifier()
      .setLabelCol("is_fraud")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(10)
      .setSubsamplingRate(0.7)
      .setFeatureSubsetStrategy("sqrt")
      .setSeed(42L)

    // Build full Pipeline
    val pipeline = new Pipeline()
      .setStages(indexers ++ Array(assembler, rf))

    // Train / test split
    val Array(train, test) = enriched.randomSplit(Array(0.8, 0.2), seed = 42L)
    println(s"Train count = ${train.count()}, Test count = ${test.count()}")

    // Fit model
    val model = pipeline.fit(train)

    // Evaluate on test data
    import org.apache.spark.ml.functions.vector_to_array

    val predictions = model.transform(test)

    // =====================
    // Custom threshold
    // =====================
    val threshold = 0.15 // try: 0.05, 0.10, 0.15, 0.20

    val scored = predictions
      .withColumn("fraud_probability", vector_to_array(col("probability"))(1))//extracts index 1 from the probability vector probability = [0.92, 0.08] then fraud_probability = 0.08
      .withColumn("prediction_custom",
        when(col("fraud_probability") >= threshold, 1.0).otherwise(0.0)
      )

    scored
      .select("is_fraud", "fraud_probability", "prediction", "prediction_custom")
      .show(10, truncate = false)


    // =====================
    // AUC ROC / AUPRC (keep using rawPrediction)
    // =====================
    val evaluatorROC = new BinaryClassificationEvaluator()
      .setLabelCol("is_fraud")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val aucROC = evaluatorROC.evaluate(predictions)
    println(f"AUC ROC = $aucROC%1.4f")

    val evaluatorPR = new BinaryClassificationEvaluator()
      .setLabelCol("is_fraud")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderPR")

    val aucPR = evaluatorPR.evaluate(predictions)
    println(f"AUPRC   = $aucPR%1.4f")


    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    // ======================================================
    // Overall Evaluation Metrics
    // ======================================================

    // scored: DataFrame that already contains
    // - prediction_custom
    // - is_fraud (label)

    val evalDF = scored
      .select(
        col("prediction_custom").cast("double").alias("prediction"),
        col("is_fraud").cast("double").alias("label")
      )

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val accuracy = evaluator
      .setMetricName("accuracy")
      .evaluate(evalDF)

    val weightedPrecision = evaluator
      .setMetricName("weightedPrecision")
      .evaluate(evalDF)

    val weightedRecall = evaluator
      .setMetricName("weightedRecall")
      .evaluate(evalDF)

    val weightedF1 = evaluator
      .setMetricName("weightedFMeasure")
      .evaluate(evalDF)

    // =====================
    // Print results
    // =====================
    println(f"Threshold used = $threshold%1.2f")

    println("============== Overall metrics ================")
    println(f"Accuracy           : $accuracy%1.4f")
    println(f"Weighted Precision : $weightedPrecision%1.4f")
    println(f"Weighted Recall    : $weightedRecall%1.4f")
    println(f"Weighted F1        : $weightedF1%1.4f")

    // ================== Save the trained model ==================
    val modelPath = "data/fraud_lr_model"   // path name is historical; it's RF now
    model.write.overwrite().save(modelPath)

    println(s"Saved model to: $modelPath")

    spark.stop()

  }
}


 */