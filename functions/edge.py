import os
import pandas as pd
import pickle
import json
from datetime import datetime
from PIL import Image
import logging

def edge_load_process(image_dir, label_path, size=(320, 240)):
    """Load and resize images with labels for local execution"""
    df_labels = pd.read_csv(label_path)
    labels = df_labels["count"].tolist()
    files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])
    images = []
    for file in files:
        path = os.path.join(image_dir, file)
        img = Image.open(path).resize(size).convert("RGB")
        images.append(img)

    print(f"Loaded {len(images)} images and {len(labels)} labels")
    return images, labels

def edge_save_monitor(images, labels, save_path, stats_path=None):
    """Save processed images and generate monitoring stats"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    if stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
    print(f"Edge stats: {stats}")
    return stats

def edge_pipeline(image_dir, label_path, save_path):
    # Check if labels file exists before processing
    if not os.path.exists(label_path):
        print("No new data to process.")
        return None
    df_labels = pd.read_csv(label_path)
    images = []
    for file in sorted(os.listdir(image_dir)):
        if file.endswith((".jpg", ".png")):
            img = Image.open(os.path.join(image_dir, file)).resize((320, 240)).convert("RGB")
            images.append(img)
    pd.to_pickle(images, save_path)
    print(f"Edge Ingestion Complete: {len(images)} images ready.")
    return images
