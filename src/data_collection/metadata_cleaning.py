import pandas as pd
import numpy as np
from geopy import distance
from sklearn.cluster import KMeans


metadata_path_input = "../../data/mapillary_images_metadata.csv"
metadata_path_clean = "../../data/mapillary_images_metadata_clean.csv"
mapillary_intermediate_path = "../../data/mapillary_images_metadata_intermediate.csv"
metadata_path_filtered = "../../data/mapillary_images_metadata_filtered.csv"

MAX_IMAGES_PER_CELL = 100
MIN_IMAGES_PER_CELL = 10
SEQUENCE_SAMPLING_DISTANCE = 30  # in meters
RANDOM_STATE = 42

def get_overall_stats(data_df):
    print("Total images: ", len(data_df))
    print("Unique imgs: ", data_df['image_id'].nunique())
    print("Unique sequences: ", data_df['sequence_id'].nunique())
    print("Unique cells(classes): ", data_df['cell_id'].nunique())
    print("District count: ", data_df['district'].nunique())
    print("Mean images per cells: ", data_df.groupby('cell_id').size().mean())
    print("Median images per cells: ", data_df.groupby('cell_id').size().median())
    print("Min images per cells: ", data_df.groupby('cell_id').size().min())
    print("Max images per cells: ", data_df.groupby('cell_id').size().max())
    print("Mean sequences per cell: ", data_df.groupby('cell_id')['sequence_id'].nunique().mean())
    print("Min sequences per cell: ", data_df.groupby('cell_id')['sequence_id'].nunique().min())
    print("Max sequences per cell: ", data_df.groupby('cell_id')['sequence_id'].nunique().max())
    print("Cells with less than 20 images: ", (data_df.groupby('cell_id').size() < 20).sum())
    print("Cells with less than 10 images: ", (data_df.groupby('cell_id').size() < 10).sum())
    print("Cells with more than 100 images: ", (data_df.groupby('cell_id').size() > 100).sum())
    print("Cells with more than 200 images: ", (data_df.groupby('cell_id').size() > 200).sum())
    print("Cells with more than 500 images: ", (data_df.groupby('cell_id').size() > 500).sum())
    print("Cells with more than 1000 images: ", (data_df.groupby('cell_id').size() > 1000).sum())
    print("Cells with more than 2000 images: ", (data_df.groupby('cell_id').size() > 2000).sum())
    print("Cells with more than 5000 images: ", (data_df.groupby('cell_id').size() > 5000).sum())
    print("Cells with more than 10000 images: ", (data_df.groupby('cell_id').size() > 10000).sum())
    print("Mean images per sequence: ", data_df.groupby('sequence_id').size().mean())
    print("Median images per sequence: ", data_df.groupby('sequence_id').size().median())
    print("Min images per sequence: ", data_df.groupby('sequence_id').size().min())
    print("Max images per sequence: ", data_df.groupby('sequence_id').size().max())
    print("Unique districts of cells with less than 20 images: ", data_df[data_df['cell_id'].isin(data_df.groupby('cell_id').size()[data_df.groupby('cell_id').size() < 20].index)]['district'].unique())
    print("Unique districts of cells with less than 10 images: ", data_df[data_df['cell_id'].isin(data_df.groupby('cell_id').size()[data_df.groupby('cell_id').size() < 10].index)]['district'].unique())
    print("Unique images per district:")
    print(data_df['district'].value_counts())
    print("Unique cells per district:")
    print(data_df.groupby('district')['cell_id'].nunique())
    print("Unique sequences per district:")
    print(data_df.groupby('district')['sequence_id'].nunique())

def clean_metadata(df, output_path):
    df.drop_duplicates(subset="image_id", inplace=True)
    df.dropna(subset=["lat", "lon", "sequence_id"], inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Metadata cleaned and saved to {output_path}.")
    return df

def filter_img_from_sequence_every_x_meter(seq_df, x_meter): 
    filtered_df = seq_df.copy().sort_values("captured_at")
    keep_indices =[]
    last_kept_coords = None
    for index,row in filtered_df.iterrows():
        current_coords = (row['lat'], row['lon'])
        if last_kept_coords is None:
            keep_indices.append(index)
            last_kept_coords = current_coords
            continue
        cur_distance =distance.distance(last_kept_coords, current_coords).meters
        if cur_distance>= x_meter:
            keep_indices.append(index)
            last_kept_coords = current_coords
    return filtered_df.loc[keep_indices].copy()

def filter_metadata_by_sequence_sampling(df, x_meter, min_images_per_sequence= None): 
    filtered_dfs = []
    for sequence_id, seq_df in df.groupby("sequence_id"):
        if min_images_per_sequence and len(seq_df) < min_images_per_sequence:
            filtered_dfs.append(seq_df) 
        else:
            filtered_dfs.append(filter_img_from_sequence_every_x_meter(seq_df, x_meter))
    filtered_df = pd.concat(filtered_dfs, ignore_index=True)
    print(f"Metadata filtered by sequence sampling.")
    return filtered_df

def sample_varied_spatial(df, n, random_state=None): 
    coords = df[["lat", "lon"]].to_numpy()
    kmeans = KMeans(
        n_clusters=n,
        random_state=random_state,
        n_init="auto"
    )
    cluster_labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    df_tmp = df.copy()
    df_tmp["_cluster"] = cluster_labels

    selected_indices = []
    for cluster_id in range(n):
        cluster_df = df_tmp[df_tmp["_cluster"] == cluster_id]
        if cluster_df.empty:
            continue
        center = centers[cluster_id]
        distances = np.sqrt(
            (cluster_df["lat"].to_numpy() - center[0]) ** 2 +
            (cluster_df["lon"].to_numpy() - center[1]) ** 2
        )
        selected_idx = cluster_df.iloc[np.argmin(distances)].name
        selected_indices.append(selected_idx)

    sampled = df.loc[selected_indices].copy()
    return sampled

def filter_cells_by_image_count(df, min_images, max_images, random_state): 
    cell_counts = df["cell_id"].value_counts()
    valid_cells = cell_counts[cell_counts >= min_images].index
    filtered_df = df[df["cell_id"].isin(valid_cells)].copy()
    output_parts = []

    for cell_id, cell_df in filtered_df.groupby("cell_id"):
        if len(cell_df) >max_images:
            sampled_cell_df = sample_varied_spatial(
                cell_df,
                n=max_images,
                random_state=random_state
            )
            output_parts.append(sampled_cell_df)
        else:
            output_parts.append(cell_df)

    filtered_df = pd.concat(output_parts, ignore_index=True)
    print(f"Metadata filtered by image count per cell.")
    return filtered_df

def filter_metadata(df, intermed_path, output_path): 
    df = filter_metadata_by_sequence_sampling(df, x_meter=SEQUENCE_SAMPLING_DISTANCE)
    mean_per_sequence = df.groupby('sequence_id').size().mean()
    df = filter_metadata_by_sequence_sampling(df, x_meter=2*SEQUENCE_SAMPLING_DISTANCE, min_images_per_sequence=mean_per_sequence)
    df.to_csv(intermed_path, index=False)
    print(f"Metadata filtered by sequence sampling and saved to {intermed_path}.")
    df = filter_cells_by_image_count(df, min_images=MIN_IMAGES_PER_CELL, max_images=MAX_IMAGES_PER_CELL, random_state=RANDOM_STATE)
    df.to_csv(output_path, index=False)
    print(f"Metadata filtered by image count per cell and saved to {output_path}.")
    return df

data_df = pd.read_csv(metadata_path_input)
data_df = clean_metadata(data_df, metadata_path_clean)
data_df = filter_metadata(data_df, mapillary_intermediate_path, metadata_path_filtered)
get_overall_stats(data_df)

