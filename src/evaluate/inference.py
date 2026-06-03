import argparse
import os
import numpy as np
import geopandas as gpd
from datasets import load_dataset
from datasets import Image
from datasets import ClassLabel
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import CLIPForImageClassification
import evaluate
import torch
import shutil

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)

SEED = 42
accuracy_metric = evaluate.load("accuracy")

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def mean(arr):
  return sum(arr) / len(arr)

def median(arr):
  arr = sorted(arr)
  if len(arr)%2==1:
    return arr[len(arr)//2]
  else:
    return (arr[(len(arr)//2)] + arr[(len(arr)//2)-1]) / 2

def find_distance(grid_gdf, pred_cell_ids, true_cell_ids):
  distances = []
  for i in range(len(pred_cell_ids)):
    cell_id = pred_cell_ids[i]
    cell_df_pred = grid_gdf[grid_gdf['cell_id'] == cell_id]
    cell_df_true = grid_gdf[grid_gdf['cell_id'] == true_cell_ids[i]]
    lat_pred = cell_df_pred['centroid_lat'].values[0]
    lon_pred = cell_df_pred['centroid_lon'].values[0]
    lat_true = cell_df_true['centroid_lat'].values[0]
    lon_true = cell_df_true['centroid_lon'].values[0]
    distance = haversine_m(lat_pred, lon_pred, lat_true, lon_true)
    distances.append(distance)
  return distances

def calc_ratios(arr):
  counts = {100:0, 500:0, 1000:0, 5000:0, 10000:0}
  for el in arr:
    if el < 100:
      counts[100] += 1
    elif el < 500:
      counts[500] += 1
    elif el < 1000:
      counts[1000] += 1
    elif el < 5000:
      counts[5000] += 1
    else:
      counts[10000] += 1
  return counts

def per_district_performance(pred_cell_ids, true_cell_ids, grid_gdf):
    district_counts = {}
    for i in range(len(pred_cell_ids)):
        cell_id = pred_cell_ids[i]
        true_cell_id = true_cell_ids[i]
        cell_df_pred = grid_gdf[grid_gdf['cell_id'] == cell_id]
        cell_df_true = grid_gdf[grid_gdf['cell_id'] == true_cell_id]
        district_pred = cell_df_pred['district'].values[0]
        district_true = cell_df_true['district'].values[0]
        if district_true not in district_counts:
            district_counts[district_true] = {"correct": 0, "total": 0}
        if district_pred == district_true:
            district_counts[district_true]["correct"] += 1
        district_counts[district_true]["total"] += 1
    for district, counts in district_counts.items():
        accuracy = counts["correct"] / counts["total"]
        print(f"District: {district}, Accuracy: {accuracy:.4f} ({counts['correct']}/{counts['total']})")

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    preds = np.argmax(logits, axis=1)
    top1 = accuracy_metric.compute(
        predictions=preds,
        references=labels
    )["accuracy"]

    top5_preds = np.argsort(logits, axis=1)[:, -5:]
    top5 = np.mean([
        labels[i] in top5_preds[i]
        for i in range(len(labels))
    ])

    return {
        "accuracy": top1,
        "top5_accuracy": top5,
    }

def load_data(data_files, id2label, label2id):
    dataset = load_dataset("csv", data_files=data_files)
    dataset = dataset.cast_column("image", Image())

    dataset = dataset.map(lambda x: {"cell_id": str(x["cell_id"])})
    def keep_seen_cells(example):
        return str(example["cell_id"]) in label2id

    dataset["benchmark"] = dataset["benchmark"].filter(keep_seen_cells)
    def encode(example, label2id):
        label = str(example["cell_id"])
        example["label"] = label2id[label]
        return example
    dataset['benchmark'] = dataset['benchmark'].map(lambda x: encode(x, label2id))
    class_label = ClassLabel(
    num_classes=len(id2label),
    names=[id2label[i] for i in range(len(id2label))]
    )
    dataset = dataset.cast_column("label", class_label)
    return dataset

def preprocess_data(dataset, image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )
    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    dataset['benchmark'].set_transform(preprocess_val)
    return dataset

def predict(model, dataset, id2label):
    inference_args = TrainingArguments(
    output_dir="./temp",
    per_device_eval_batch_size=32,
    remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=inference_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    predictions = trainer.predict(dataset["benchmark"])

    logits = predictions.predictions
    predicted_class_ids = np.argmax(logits, axis=-1)
    print(predictions.metrics)
    true_labels = predictions.label_ids
    pred_cell_ids = [int(id2label[label]) for label in predicted_class_ids]
    true_cell_ids = [int(id2label[label]) for label in true_labels]

    shutil.rmtree("./temp", ignore_errors=True)
    return pred_cell_ids, true_cell_ids

def evaluate_predictions(pred_cell_ids, true_cell_ids, grid_gdf):
    distances = find_distance(grid_gdf, pred_cell_ids, true_cell_ids)
    mean_distance = mean(distances)
    print("Average distance between preds and labels in meters and kilometers: ", mean_distance, mean_distance/1000)
    median_distance = median(distances)
    print('Median distance between preds and labels in meters and kilometers: ', median_distance, median_distance/1000)
    ratios = calc_ratios(distances)
    print('Distance ratios: ', ratios)
    per_district_performance(pred_cell_ids, true_cell_ids, grid_gdf)

def main(args):
    is_clip = args.is_clip
    if is_clip:
        model = CLIPForImageClassification.from_pretrained(args.model)
    else:
        model = AutoModelForImageClassification.from_pretrained(args.model)
    image_processor = AutoImageProcessor.from_pretrained(args.model)
    data_path = args.data
    data_files = {
        "benchmark": os.path.join(data_path, "benchmark2/mapillary_images_metadata_benchmark.csv"),
    }
    id2label, label2id = model.config.id2label, model.config.label2id
    dataset = load_data(data_files, id2label, label2id)
    print("Data loaded. Number of samples in benchmark: ", len(dataset["benchmark"]))
    grid_path = os.path.join(data_path, args.grid)
    grid_gdf = gpd.read_file(grid_path)
    dataset = preprocess_data(dataset, image_processor)

    print("Starting inference...")
    pred_cell_ids, true_cell_ids = predict(model, dataset, id2label)
    evaluate_predictions(pred_cell_ids, true_cell_ids, grid_gdf)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the benchmark.")
    parser.add_argument("--model", type=str, help="Path to the trained model.")
    parser.add_argument("--data", type=str, help="Path to the data folder.")
    parser.add_argument("--grid", type=str, help="Grid file name.")
    parser.add_argument("--is_clip", action="store_true", help="Whether the model is a CLIP model.")
    args = parser.parse_args()
    main(args)