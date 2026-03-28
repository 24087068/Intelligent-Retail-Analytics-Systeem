import logging
import json
from datetime import datetime
from pyspark.sql.functions import col, dayofweek, month, count as spark_count, mean, stddev, min as spark_min, max as spark_max
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("IntelligentRetail") \
    .getOrCreate()

def cloud_load_transform(spark, path):
    """Load CSV and add date features - for Databricks"""
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = df.withColumn("date", col("date").cast("date"))
    df = df.dropna()
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    df = df.withColumn("month", month(col("date")))
    
    print(f"Shape: ({df.count()}, {len(df.columns)})")
    print(f"Null values: {df.select([spark_count(col(c).isNull().cast('int')).alias(c) for c in df.columns]).collect()[0].asDict()}")
    print(f"Negative sales: {df.filter(col('sales') < 0).count()}")
    return df

def cloud_save_monitor(df, save_path, stats_path=None):
    """Save parquet and generate stats - for Databricks"""
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


from pyspark.sql.functions import col, dayofweek, month
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType


def cloud_pipeline(spark, raw_input, processed_output):
    # 1. Define Schema (from your notes: Blueprint for validation)
    schema = StructType([
        StructField("date", DateType(), True),
        StructField("store", IntegerType(), True),
        StructField("item", IntegerType(), True),
        StructField("sales", IntegerType(), True)
    ])

    # 2. Load & Transform (Lazy Transformations)
    df = spark.read.csv(raw_input, header=True, schema=schema)
    df = df.dropna().filter(col("sales") >= 0)
    df = df.withColumn("day_of_week", dayofweek(col("date")))
    df = df.withColumn("month", month(col("date")))

    # 3. Action & Save (Optimization: coalesce reduces shuffling)
    df.coalesce(1).write.mode("overwrite").parquet(processed_output)

    # Simple console monitoring
    print(f"Processed {df.count()} rows.")
    return df
