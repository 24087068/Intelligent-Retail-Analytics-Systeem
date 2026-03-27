import logging
import os
import pandas as pd
from PIL import Image
import pickle
import json
from datetime import datetime
from pyspark.sql import SparkSession

def edge_load_process(spark, image_dir, label_path, size=(320, 240), batch_size=100):
    df_labels = spark.read.csv(label_path, header=True, inferSchema=True)
    labels = [row["count"] for row in df_labels.collect()]

    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")])
    processed = []

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        for file in batch_files:
            path = os.path.join(image_dir, file)
            img = Image.open(path).resize(size).convert("RGB")
            processed.append(img)

    print(f"Loaded images: {len(processed)}")
    print(f"Loaded labels: {len(labels)}")
    print(f"Number of images: {len(processed)}")
    print(f"Number of labels: {len(labels)}")

    return processed, labels

def edge_save_monitor(images, labels, save_path, stats_path="../data/monitoring/edge_stats.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(images, f)

    stats = {
        "timestamp": datetime.now().isoformat(),
        "image_count": len(images),
        "label_count": len(labels),
        "label_mean": sum(labels) / len(labels) if labels else 0,
        "label_min": min(labels) if labels else 0,
        "label_max": max(labels) if labels else 0
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Edge stats: {stats}")
    print(f"Processed images: {len(images)}")
    return stats

def edge_pipeline():
    logging.info("Running edge pipeline")
    processed, labels = edge_load_process("../data/raw/image/frames", "../data/raw/image/labels.csv")
    edge_save_monitor(processed, labels, "../data/processed/images.pkl")
    logging.info("Edge pipeline finished")