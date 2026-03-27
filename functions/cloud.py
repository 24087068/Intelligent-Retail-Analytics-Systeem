import logging
import os
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month, count as spark_count, mean, stddev, min as spark_min, max as spark_max

spark = SparkSession.builder.appName("CloudPipeline").getOrCreate()

def cloud_load_transform(path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("date", col("date").cast("date"))
    df = df.dropna()
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    df = df.withColumn("month", month(col("date")))

    print(f"Shape: ({df.count()}, {len(df.columns)})")
    print(f"Null: {df.select([spark_count(col(c).isNull().cast('int')).alias(c) for c in df.columns]).collect()[0].asDict()}")
    print(f"Negatives: {df.filter(col('sales') < 0).count()}")

    return df

def cloud_save_monitor(df, save_path, stats_path="../data/monitoring/cloud_stats.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    df.write.mode("overwrite").parquet(save_path)

    stats_data = df.select(
        spark_count("*").alias("row_count"),
        mean("sales").alias("sales_mean"),
        stddev("sales").alias("sales_std"),
        spark_min("sales").alias("sales_min"),
        spark_max("sales").alias("sales_max"),
        spark_min("date").alias("date_min"),
        spark_max("date").alias("date_max")
    ).collect()[0]

    store_count = df.select("store").distinct().count()
    item_count = df.select("item").distinct().count()

    stats = {
        "timestamp": datetime.now().isoformat(),
        "row_count": stats_data["row_count"],
        "sales_mean": float(stats_data["sales_mean"]),
        "sales_std": float(stats_data["sales_std"]),
        "sales_min": int(stats_data["sales_min"]),
        "sales_max": int(stats_data["sales_max"]),
        "date_range": [str(stats_data["date_min"]), str(stats_data["date_max"])],
        "store_count": store_count,
        "item_count": item_count
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Cloud stats: {stats}")
    return stats

def cloud_validate_output(path):
    df = spark.read.parquet(path)
    print(f"Validated shape: ({df.count()}, {len(df.columns)})")
    df.show(5)
    return df

def cloud_pipeline():
    logging.info("Running cloud pipeline")
    df = cloud_load_transform("../data/raw/text/train.csv")
    cloud_save_monitor(df, "../data/processed/sales.parquet")
    cloud_validate_output("../data/processed/sales.parquet")
    logging.info("Cloud pipeline finished")