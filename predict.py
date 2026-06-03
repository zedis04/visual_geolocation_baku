import argparse
import os
import numpy as np
import geopandas as gpd
import json
from PIL import Image
from transformers import AutoImageProcessor
from transformers import AutoModelForImageClassification
from transformers import CLIPForImageClassification
import evaluate
import torch

SEED = 42
accuracy_metric = evaluate.load("accuracy")

def load_preprocess_image(image_path, image_processor):
    image = Image.open(image_path).convert("RGB")
    image_encoding = image_processor(image, return_tensors="pt")
    return image_encoding

def predict(model, image_encoding, id2label):
    with torch.no_grad():
        logits = model(**image_encoding).logits
    # predicted_label = logits.argmax(-1).item()
    # confidence_score = torch.softmax(logits, dim=-1).max().item()
    top3_predicted_labels = logits.argsort(-1, descending=True).squeeze()[:3].tolist()
    top3_predicted_labels_with_confidence = [(id2label[label], torch.softmax(logits, dim=-1)[0, label].item()) 
                                             for label in top3_predicted_labels]
    return top3_predicted_labels_with_confidence

def get_cell_centroid(cell_id, grid_gdf):
    cell_df = grid_gdf[grid_gdf['cell_id'] == cell_id]
    if cell_df.empty:
        raise ValueError(f"Cell ID {cell_id} not found in grid.")
    lat_pred = cell_df['centroid_lat'].values[0]
    lon_pred = cell_df['centroid_lon'].values[0]
    return lat_pred, lon_pred

def main(args):
    is_clip = args.is_clip
    if is_clip:
        model = CLIPForImageClassification.from_pretrained(args.model)
    else:
        model = AutoModelForImageClassification.from_pretrained(args.model)
    image_processor = AutoImageProcessor.from_pretrained(args.model)
    grid_path = os.path.join(args.grid)
    image_path = args.image
    id2label = model.config.id2label
    image_encoding = load_preprocess_image(image_path, image_processor)

    grid_gdf = gpd.read_file(grid_path)
    output = {}

    top3_pred_labels_with_confidence = predict(model, image_encoding, id2label)
    for i, (cell_id, confidence) in enumerate(top3_pred_labels_with_confidence):
        lat_pred, lon_pred = get_cell_centroid(int(cell_id), grid_gdf)
        if i==0:
            output["lat"] = float(lat_pred)
            output["lon"] = float(lon_pred)
            output["confidence"] = round(confidence, 2)
            output["top_k"]=[]
        else:
            output["top_k"].append({
                "lat": float(lat_pred),
                "lon": float(lon_pred),
                "confidence": round(confidence,2)
            })
    print(output)
    img_name = os.path.basename(image_path).split('.')[0]
    output_path = "output_" + img_name + ".json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the benchmark.")
    parser.add_argument("--model", type=str, help="Path to the trained model.")
    parser.add_argument("--image", type=str, help="Path to the image.")
    parser.add_argument("--grid", type=str, help="Path to the grid geojson.")
    parser.add_argument("--is_clip", action="store_true", help="Whether the model is a CLIP model.")
    args = parser.parse_args()
    main(args)
