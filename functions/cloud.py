import logging
import json
from datetime import datetime
from pyspark.sql.functions import col, dayofweek, month, count as spark_count, mean, stddev, min as spark_min, max as spark_max
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, dayofweek, month
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("IntelligentRetail") \
    .getOrCreate()

def cloud_load_transform(spark, path):
    """Load CSV and add date features for model training"""
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("date", col("date").cast("date"))
    df = df.dropna()
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    df = df.withColumn("month", month(col("date")))

    print(f"Shape: ({df.count()}, {len(df.columns)})")
    print(f"Null: {df.select([spark_count(col(c).isNull().cast('int')).alias(c) for c in df.columns]).collect()[0].asDict()}")
    print(f"Negatives: {df.filter(col('sales') < 0).count()}")
    return df

def cloud_save_monitor(df, save_path, stats_path=None):
    """Save parquet and generate monitoring stats"""
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
    stats = {
        "timestamp": datetime.now().isoformat(),
        "row_count": stats_data["row_count"],
        "sales_mean": float(stats_data["sales_mean"]),
        "sales_std": float(stats_data["sales_std"]),
        "sales_min": int(stats_data["sales_min"]),
        "sales_max": int(stats_data["sales_max"]),
        "date_range": [str(stats_data["date_min"]), str(stats_data["date_max"])],
        "store_count": df.select("store").distinct().count(),
        "item_count": df.select("item").distinct().count()
    }
    if stats_path:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
    print(f"Cloud stats: {stats}")
    return stats

def cloud_pipeline(spark, raw_input, processed_output):
    schema = StructType([
        StructField("date", DateType(), True),
        StructField("store", IntegerType(), True),
        StructField("item", IntegerType(), True),
        StructField("sales", IntegerType(), True)
    ])
    df = spark.read.csv(raw_input, header=True, schema=schema)
    df = df.dropna().filter(col("sales") >= 0)
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    df = df.withColumn("month", month(col("date")))
    # coalesce reduces number of output files
    df.coalesce(1).write.mode("overwrite").parquet(processed_output)
    print(f"Processed {df.count()} rows.")
    return df
