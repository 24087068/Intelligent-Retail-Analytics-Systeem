import logging
import os
import pandas as pd
from PIL import Image
import pickle
import hashlib
import json
from datetime import datetime

def edge_load_images(image_dir):
    images = []
    for file in sorted(os.listdir(image_dir)):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(image_dir, file)
            img = Image.open(path)
            images.append(img)
    print(f"Loaded images: {len(images)}")
    return images

def edge_load_labels(path):
    df = pd.read_csv(path)
    labels = df["count"].tolist()
    print(f"Loaded labels: {len(labels)}")
    return labels, df

def edge_validate(images, labels):
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

def edge_preprocess_images(images, size=(320, 240)):
    processed = []
    for img in images:
        img = img.resize(size)
        img = img.convert("RGB")
        processed.append(img)
    print(f"Processed images: {len(processed)}")
    return processed

def edge_save_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def edge_monitor_data(images, labels, stats_path="../data/monitoring/edge_stats.json"):
    stats = {
        "timestamp": datetime.now().isoformat(),
        "image_count": len(images),
        "label_count": len(labels),
        "label_mean": sum(labels) / len(labels) if labels else 0,
        "label_min": min(labels) if labels else 0,
        "label_max": max(labels) if labels else 0
    }
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Edge stats: {stats}")
    return stats


def edge_pipeline():
    logging.info("Running edge pipeline")

    images = edge_load_images("../data/raw/image/frames")
    labels, _ = edge_load_labels("../data/raw/image/labels.csv")
    edge_validate(images, labels)
    edge_monitor_data(images, labels)
    processed = edge_preprocess_images(images)
    edge_save_data(processed, "../data/processed/images.pkl")

    logging.info("Edge pipeline finished")