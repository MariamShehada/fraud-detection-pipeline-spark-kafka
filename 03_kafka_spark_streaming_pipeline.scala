import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.util.sketch.BloomFilter

object test_final {

  def main(args: Array[String]): Unit = {
    // Create Spark session (smaller, more memory-friendly)
    val spark = SparkSession.builder()
      .appName("FraudStreamingApp-NewPipeline")
      .master("local[2]")    // use only 2 cores
      .config("spark.executor.memory", "2g") // limit executor memory
      .config("spark.driver.memory", "2g")   // limit driver memory
      .config("spark.sql.shuffle.partitions", "4") // fewer shuffle tasks
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._
    import org.apache.spark.util.sketch.BloomFilter

    // ---------------- Kafka topics ----------------
    val bootstrap = "localhost:9092" //tells Spark where Kafka lives
    val inputTopic = "transactions_raw"
    val otherTopic = "other_transactions"
    val fraudTopic = "fraud_transactions"

    // ---------------- Mongo ----------------
    val mongoUri = "mongodb://127.0.0.1:27017"
    val mongoDb = "frauddb"

    // ---------------- Load Bloom Filter ----------------
    /*
     .bf files contain binary-encoded Bloom Filter bits,
      need the Bloom Filter library to decode and restore the Bloom Filter object. .bf → use BloomFilter.readFrom
      senderBloom contains :  the Bloom Filter object that is a compressed, probabilistic data structure that can answer:
       Is this sender possibly in the blacklist? Yes / No
    */
    val bfPath = "data/blacklist_senders.bf"
    val in = new java.io.FileInputStream(bfPath)
    val senderBloom = BloomFilter.readFrom(in)
    in.close()
    // Broadcast it so all executors can use it
    val bfBroadcast = spark.sparkContext.broadcast(senderBloom)
    // UDF to check if sender is in the blacklist Bloom filter
    //mightContainString is NOT a built-in Spark function , UDF allows Spark to apply mightContainString on every row
    val isSenderBlacklisted = udf { sender: String =>
      if (sender == null) false else bfBroadcast.value.mightContainString(sender) //Before checking if a sender is blacklisted, make sure the sender is not null
    }

    // ---------------- Load ML PipelineModel   ----------------
    val modelPath = "data/fraud_lr_model"
    val model = PipelineModel.load(modelPath)

    // ---------------- Schema ----------------
    val transactionSchema = new StructType()
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
      .add("fraud_source", StringType)



    /*
    Flow is:
      1. Python producer writes messages to the Kafka topic.
      2. Kafka stores them on disk in the broker.
      3. Spark readStream pulls some messages each micro-batch (up to maxOffsetsPerTrigger).
      4. Spark stores its current offset in the checkpoint directory.
      Even if producer sends 1000 msg/sec and the consumer read 50 per batch →
       data stays in Kafka until Spark catches up or retention (retention time)is reached.
       NOTE:
       when use a checkpoint location IN writeStream , startingOffsets only matters the first time After that, Spark remembers
       where it stopped using the checkpoint, it continues from saved offsets, not from latest or earliest
    */

    def readKafka(topic: String): DataFrame =
      spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap) //tells Spark where Kafka lives
        .option("subscribe", topic)
        .option("startingOffsets", "latest") //controls where Spark begins reading,latest: only read new messages from now on
        .option("maxOffsetsPerTrigger", "50")// Spark can read at most 50 Kafka messages per micro-batch.
        .load()


    /*
    value is bytes -> cast to string then parse JSON
    Kafka does NOT store strings — it stores bytes ex:7B 22 61 6D 6F 75 6E 74 ...
    This code reads raw bytes from Kafka, converts them into a JSON string,
    parses that JSON using the schema, and expands it into proper Spark columns

   */

    def parseKafka(df: DataFrame): DataFrame =
      df.selectExpr("CAST(value AS STRING) as json")//Convert Kafka message bytes → JSON string
        .select(from_json($"json", transactionSchema).as("data"))//spark created one big column named "data" that contains all fields inside it
        .select("data.*")//expands them into separate columns


    def writeToKafka(df: DataFrame, topic: String, checkpoint: String): Unit = {
      df.selectExpr(
          "CAST(transaction_id AS STRING) as key",
          "to_json(struct(*)) as value"
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("topic", topic)
        .option("checkpointLocation", checkpoint)
        .outputMode("append")
        .start()
    }





    // ============================================================
    // STAGE 1: transactions_raw -> Bloom routing
    //   - non-blacklisted -> other_transactions
    //   - blacklisted -> fraud_transactions
    // ============================================================
    val rawStream = parseKafka(readKafka(inputTopic))

    val bloomTagged = rawStream.withColumn(
      "sender_blacklisted",
      isSenderBlacklisted($"sender_account")
    )

    val bloomFraud = bloomTagged
      .filter($"sender_blacklisted" === true)
      .withColumn("fraud_source", lit("bloom_filter"))


    val bloomOther = bloomTagged
      .filter($"sender_blacklisted" === false)

    writeToKafka(
      bloomOther,
      otherTopic,
      "file:///C:/tmp/spark_checkpoints/kafka_other_transactions"
    )

    writeToKafka(
      bloomFraud,
      fraudTopic,
      "file:///C:/tmp/spark_checkpoints/kafka_fraud_transactions_bloom"
    )

    // ============================================================
    // STAGE 2: other_transactions -> ML -> fraud_transactions
    // ============================================================

    // pipeline does preprocessing now
    val otherStream = parseKafka(readKafka(otherTopic))

    val scored = model.transform(otherStream)
      .withColumn("fraud_probability", vector_to_array($"probability")(1))
      .withColumn("is_fraud_prediction", $"prediction".cast("int"))

    val mlFraud = scored
       .filter($"fraud_probability" >= 0.5 || $"is_fraud_prediction" === 1)
      .withColumn("fraud_source", lit("ml_model")) //Fraud detected by: Bloom filter or ML model


    writeToKafka(
      mlFraud,
      fraudTopic,
      "file:///C:/tmp/spark_checkpoints/kafka_fraud_transactions_ml"
    )

    val mlNotFraud = scored
      .filter($"is_fraud_prediction" === 0 && $"fraud_probability" < 0.5)
      .select(
        $"transaction_id", $"sender_account", $"receiver_account", $"amount",
        $"transaction_type", $"merchant_category", $"payment_channel",
        $"fraud_probability", $"is_fraud_prediction"
      )


    // ============================================================
    // STAGE 3: fraud_transactions -> Insights -> Mongo
    // ============================================================

    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.functions.approx_count_distinct

    // ============================================================
    // Common streams
    // ============================================================

    // Fraud stream (alerts)
    val fraudStream = parseKafka(readKafka(fraudTopic))
      .withColumn("event_time", to_timestamp($"timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))
      .withWatermark("event_time", "2 minutes")

    // All transactions stream (needed for fraud rate)
    val allStream = parseKafka(readKafka(inputTopic))
      .withColumn("event_time", to_timestamp($"timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))
      .withWatermark("event_time", "2 minutes")

    import org.apache.spark.sql.DataFrame
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.functions.approx_count_distinct

    // ============================================================
    // 1) Real-time fraud detection rate (last 2 minutes)
    // ============================================================
    /*
    val total2m = allStream
      .groupBy(window($"event_time", "2 minutes"))
      .agg(count("*").as("total_tx_2m"))
      .withColumn("window_start", $"window.start")
      .withColumn("window_end", $"window.end")
      .drop("window")

    val fraud2m = fraudStream
      .groupBy(window($"event_time", "2 minutes"))
      .agg(count("*").as("fraud_tx_2m"))
      .withColumn("window_start", $"window.start")
      .withColumn("window_end", $"window.end")
      .drop("window")

    val fraudRate2m = total2m
      .join(fraud2m, Seq("window_start", "window_end"), "left")
      .na.fill(0, Seq("fraud_tx_2m"))
      .withColumn("fraud_rate_2m", $"fraud_tx_2m" / $"total_tx_2m")

    val qFraudRate2m = fraudRate2m.writeStream
      .outputMode("update")
      .option("checkpointLocation", "file:///C:/tmp/spark_checkpoints/fraud_rate_2m")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        val cnt = batchDF.count()
        println(s">>> Writing batch $batchId to Mongo (fraud_rate_2m), rows = $cnt")
        if (cnt > 0) {
          batchDF.write.format("mongo")
            .mode("append")
            .option("uri", mongoUri)
            .option("database", mongoDb)
            .option("collection", "fraud_rate_2m")
            .save()
        }
      }.start()
    */


    // ============================================================
    // 2) Fraud financial impact (loss per 2-minute window)
    // ============================================================

    val fraudLoss2m = fraudStream
      .groupBy(window($"event_time", "2 minutes"))
      .agg(
        count("*").as("fraud_count_2m"),
        sum($"amount").as("fraud_loss_2m")
      )
      .withColumn("window_start", $"window.start")
      .withColumn("window_end", $"window.end")
      .drop("window")
    // foreachBatch:
    // - Spark calls this ONCE PER MICRO-BATCH
    // - batchDF is a NORMAL (non-streaming) DataFrame
    // - batchId is just a sequential ID (0,1,2,...)
    val qFraudLoss2m = fraudLoss2m.writeStream
      .outputMode("append")
      .option("checkpointLocation", "file:///C:/tmp/spark_checkpoints/fraud_loss_2m")//Checkpoint stores :Kafka offset ,aggregation state , watermark progress
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        // Count how many rows Spark produced in THIS micro-batch
        val cnt = batchDF.count()
        println(s">>> Writing batch $batchId to Mongo (fraud_loss_2m), rows = $cnt")
        // Only write if Spark actually produced rows
        // (avoid empty writes to Mongo)
        // Write THIS micro-batch to MongoDB
        //
        // - This is a NORMAL batch write (not streaming)
        // - Each row becomes ONE Mongo document
        if (cnt > 0) {
          batchDF.write.format("mongo")
            .mode("append")
            .option("uri", mongoUri)
            .option("database", mongoDb)
            .option("collection", "fraud_loss_2m")
            .save()
        }
      }.start()


    // ============================================================
    // 3) Min / Max fraud transaction amount (time-based)
    // ============================================================

    val fraudMinMax2m = fraudStream
      .groupBy(window($"event_time", "2 minutes"))
      .agg(
        min($"amount").as("min_fraud_amount_2m"),
        max($"amount").as("max_fraud_amount_2m")
      )
      .withColumn("window_start", $"window.start")
      .withColumn("window_end", $"window.end")
      .drop("window")

    val qFraudMinMax2m = fraudMinMax2m.writeStream
      .outputMode("append")
      .option("checkpointLocation", "file:///C:/tmp/spark_checkpoints/fraud_minmax_2m")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        val cnt = batchDF.count()
        println(s">>> Writing batch $batchId to Mongo (fraud_minmax_2m), rows = $cnt")
        if (cnt > 0) {
          batchDF.write.format("mongo")
            .mode("append")
            .option("uri", mongoUri)
            .option("database", mongoDb)
            .option("collection", "fraud_minmax_2m")
            .save()
        }
      }.start()


    // ============================================================
    // 4) Suspicious receiver behavior
    //    (many senders → one receiver in 2 minutes)
    // ============================================================
    /*
    val receiverThreshold = 10

    val suspiciousReceiver2m = fraudStream
      .groupBy(window($"event_time", "2 minutes"), $"receiver_account")
      .agg(
        approx_count_distinct($"sender_account").as("approx_distinct_senders_2m")
      )
      .filter($"approx_distinct_senders_2m" >= receiverThreshold)
      .withColumn("window_start", $"window.start")
      .withColumn("window_end", $"window.end")
      .drop("window")

    val qSuspiciousReceiver2m = suspiciousReceiver2m.writeStream
      .outputMode("update")
      .option("checkpointLocation", "file:///C:/tmp/spark_checkpoints/suspicious_receivers_2m")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        val cnt = batchDF.count()
        println(s">>> Writing batch $batchId to Mongo (suspicious_receivers_2m), rows = $cnt")
        if (cnt > 0) {
          batchDF.write.format("mongo")
            .mode("append")
            .option("uri", mongoUri)
            .option("database", mongoDb)
            .option("collection", "suspicious_receivers_2m")
            .save()
        }
      }.start()
    */


    // ============================================================
    // 5) Real-time fraud alerts (no aggregation)
    // ============================================================

    val fraudAlerts = fraudStream
      .select(
        $"transaction_id",
        $"timestamp",
        $"sender_account",
        $"receiver_account",
        $"amount",
        $"transaction_type",
        $"merchant_category",
        $"payment_channel",
        $"fraud_source"
      )

    val qFraudAlerts = fraudAlerts.writeStream
      .outputMode("append")
      .option("checkpointLocation", "file:///C:/tmp/spark_checkpoints/fraud_alerts_realtime")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        val cnt = batchDF.count()
        println(s">>> Writing batch $batchId to Mongo (fraud_alerts_realtime), rows = $cnt")
        if (cnt > 0) {
          batchDF.write.format("mongo")
            .mode("append")
            .option("uri", mongoUri)
            .option("database", mongoDb)
            .option("collection", "fraud_alerts_realtime")
            .save()
        }
      }.start()



    // ============================================================
    // 6)  // Design (Option 2: Raw + Fraud-only topics):
    //// - Raw transactions DO NOT contain fraud labels (realistic assumption).
    //// - Fraud is detected upstream (Bloom Filter / ML model).
    //// - Two independent streaming aggregations are used:
    ////
    //// 1) summary_total_by_type_2m  (from RAW transactions topic)
    ////    - total_tx_2m        : total number of transactions per type
    ////    - total_amount_2m    : total transaction amount
    ////    - avg_amount_2m      : average transaction amount
    ////
    //// 2) summary_fraud_by_type_2m  (from FRAUD-ONLY topic)
    ////    - fraud_tx_2m        : number of fraudulent transactions per type
    ////
    //// Both aggregations use:
    //// - 2-minute tumbling event-time windows
    //// - watermark = 2 minutes
    //// - outputMode = append
    ////
    //// Each aggregation is written to a separate MongoDB collection.
    //# ============================================================
    //# Query 6 (Streamlit): Merge totals + fraud counts and compute fraud rate
    //# ============================================================
    //#
    //# Streamlit reads two MongoDB collections:
    //#
    //# 1) summary_total_by_type_2m
    //#    - Generated from the RAW transactions stream
    //#    - Contains total transaction counts and amounts per transaction type
    //#
    //# 2) summary_fraud_by_type_2m
    //#    - Generated from the FRAUD-ONLY stream
    //#    - Contains fraud transaction counts per transaction type
    //#
    //# Streamlit performs the following steps:
    //# - Incrementally loads new documents from both collections
    //# - Merges them on:
    //#     (window_start, window_end, transaction_type)
    //# - Fills missing fraud counts with zero
    //# - Computes:
    //#     fraud_rate_2m = fraud_tx_2m / total_tx_2m
    //#
    //# This design keeps Spark streaming logic simple and avoids
    //# multi-stream joins, while still producing accurate real-time
    //# fraud metrics for visualization.
    //#
    //# ============================================================


    // ============================================================


    //Totals from raw topic
    val rawBase = parseKafka(readKafka(inputTopic))
      .withColumn("event_time", to_timestamp($"timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))
      .withWatermark("event_time", "2 minutes")

    val totalByType2m = rawBase
      .groupBy(window($"event_time", "2 minutes"), $"transaction_type")
      .agg(
        count(lit(1)).as("total_tx_2m"),
        sum($"amount").as("total_amount_2m"),
        avg($"amount").as("avg_amount_2m")
      )
      .withColumn("window_start", $"window.start")
      .withColumn("window_end", $"window.end")
      .drop("window")

    val qTotalByType2m = totalByType2m.writeStream
      .outputMode("append")
      .trigger(org.apache.spark.sql.streaming.Trigger.ProcessingTime("2 minutes"))
      .option("checkpointLocation", "file:///C:/tmp/spark_checkpoints/summary_total_by_type_2m")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        val cnt = batchDF.count()
        println(s">>> Writing batch $batchId to Mongo (summary_total_by_type_2m), rows = $cnt")
        if (cnt > 0) {
          batchDF.write.format("mongo")
            .mode("append")
            .option("uri", mongoUri)
            .option("database", mongoDb)
            .option("collection", "summary_total_by_type_2m")
            .save()
        }
      }
      .start()
    //Fraud counts from fraud-only topic
    val fraudBase = parseKafka(readKafka(fraudTopic))
      .withColumn("event_time", to_timestamp($"timestamp", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"))
      .withWatermark("event_time", "2 minutes")

    val fraudByType2m = fraudBase
      .groupBy(window($"event_time", "2 minutes"), $"transaction_type")
      .agg(
        count(lit(1)).as("fraud_tx_2m")
      )
      .withColumn("window_start", $"window.start")
      .withColumn("window_end", $"window.end")
      .drop("window")

    val qFraudByType2m = fraudByType2m.writeStream
      .outputMode("append")
      .trigger(org.apache.spark.sql.streaming.Trigger.ProcessingTime("2 minutes"))
      .option("checkpointLocation", "file:///C:/tmp/spark_checkpoints/summary_fraud_by_type_2m")
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        val cnt = batchDF.count()
        println(s">>> Writing batch $batchId to Mongo (summary_fraud_by_type_2m), rows = $cnt")
        if (cnt > 0) {
          batchDF.write.format("mongo")
            .mode("append")
            .option("uri", mongoUri)
            .option("database", mongoDb)
            .option("collection", "summary_fraud_by_type_2m")
            .save()
        }
      }
      .start()


    spark.streams.awaitAnyTermination()
  }
}
