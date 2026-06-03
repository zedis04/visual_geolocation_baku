import pandas as pd
import numpy as np

metadata_path_input = "../../data/mapillary_images_metadata_filtered.csv"
metadata_path_benchmark = "../../data/benchmark/mapillary_images_metadata_benchmark.csv"
matedata_path_train = "../../data/train/mapillary_images_metadata_train.csv"
SPLIT = 0.2 
RANDOM_STATE = 42


def split_benchmark(data_df, input_path, output_path_benchmark, output_path_train, random_state=RANDOM_STATE, split=SPLIT):
    df = data_df.copy()
    sequence_ids = df["sequence_id"].unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(sequence_ids)
    n_benchmark_sequences = int(len(sequence_ids) * split)
    benchmark_sequences = sequence_ids[:n_benchmark_sequences]
    df["split"] = np.where(
        df["sequence_id"].isin(benchmark_sequences),
        "benchmark",
        "train",
    )
    benchmark_df = df[df["split"] == "benchmark"].drop(columns=["split"]).sort_values(by="cell_id")
    train_df = df[df["split"] == "train"].drop(columns=["split"]).sort_values(by="cell_id")
    benchmark_df.to_csv(output_path_benchmark, index=False)
    train_df.to_csv(output_path_train, index=False)
    df = df.sort_values(by="cell_id") 
    df.to_csv(input_path, index=False)  
    print(f"Benchmark set saved to {output_path_benchmark}.")
    print(f"Train set saved to {output_path_train}.")
    print(f"Original dataset updated with split information and saved to {input_path}.")

data_df = pd.read_csv(metadata_path_input)
split_benchmark(data_df, metadata_path_input, metadata_path_benchmark, matedata_path_train)