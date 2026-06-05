# Ground-Level Visual Geolocation in Baku

This repository contains a visual geolocation system for predicting the approximate location of outdoor ground-level photographs taken in Baku, Azerbaijan.

# Task Overview
The goal is to infer the location of a Baku street-level image. The problem is formulated as a grid-cell classification task: Baku is divided into spatial cells, and the model predicts the most likely cell for a given image and outputs its centroid coordinates with confidence score.

# Repo Structure Notes
Benchmark, train folders, grid files must be placed inside data folder. 

Format for either folder should include images folder and metadata csv name mapillary_images_metadata_train/benchmark.csv

# Environmental setup
Conda environment was used.
The project was developed using Python and Hugging Face Transformers. Training was performed on Google Colab Pro with a T4 GPU.

# Data 
Data was created using scripts in data_collection folder in following order: 

1. create_grids.py
2. collect_metadata.py
3. metadata_cleaning.py
4. benchmark_split.py
5. download_imgs.py
6. add_path.py

To use Mapillary API individual access token must be created.

# Traning 
The following models can be trained with script, the appropriate index of the model should be included in the command: 

model_checkpoints = ["facebook/convnext-tiny-224", "geolocal/StreetCLIP", "microsoft/swin-tiny-patch4-window7-224","microsoft/resnet-50"]

Example command:

python src/train/train.py \

--model  model_index_from_array \

 --epochs 15 \

--batch_size 32 \

--lr 5e-5 \

--data data_folder_path

# Evaluation
Example command:

python src/evaluate/inference.py \ 

--model  model_path \

--data data_folder_path \

--grid grid_file_name  \

--is_clip

# Inference
Example command:

python predict.py \ 

--model  model_path \ 

--image path/to/photo.jpg \

--grid grid_file_path  \ 

--is_clip

An output JSON file will be generated.

# Deliverables
The repo contains source code however benchmark and finetuned model weights are not provided here.