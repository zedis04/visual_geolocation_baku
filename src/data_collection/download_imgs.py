import pandas as pd
import requests
import os

MLY_ACCESS_TOKEN = ""
metadata_path_input = "../../data/mapillary_images_metadata_filtered.csv"
output_train_path = "../../data/train/images/"
output_benchmark_path = "../../data/benchmark/images/"

def download_image(image_id, output_path):
    url = f"https://graph.mapillary.com/{image_id}?fields=thumb_1024_url&access_token={MLY_ACCESS_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        image_url = data['thumb_1024_url']
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            with open(os.path.join(output_path, f"{image_id}.jpg"), 'wb') as f:
                f.write(image_response.content)
            print(f"Downloaded image {image_id}")
            return 0
        else:
            print(f"Failed to download image from {image_url}")
            return 1
    else:
        print(f"Failed to get image metadata for {image_id}")
        return 2

def download_images():
    metadata_df = pd.read_csv(metadata_path_input)
    errors = []
    for index, row in metadata_df.iterrows():
        image_id = row['image_id']
        split = row['split']
        if split == "train":
            res = download_image(image_id, output_train_path)
        elif split == "benchmark":
            res = download_image(image_id, output_benchmark_path)
        if res != 0:
            errors.append(image_id)
    if errors:
        print(f"Failed to download {len(errors)} images: {errors}")
    print("Images downloaded.")

download_images()
