import pandas as pd

par_folder = "data/"
data_folder = "../../data/"
file_name = "mapillary_images_metadata"

def add_path_to_metadata(file_path, par_folder, is_train):
    df = pd.read_csv(file_path)
    if is_train:
        df["image"] = par_folder + "train/images/" + df["image_id"].astype(str) + ".jpg"
    else:
        df["image"] = par_folder + "benchmark/images/" + df["image_id"].astype(str) + ".jpg"
    df.to_csv(file_path, index=False)
    print(f"Path added to {file_path}")


train_file_path = data_folder + "train/" + file_name + "_train.csv"
benchmark_file_path = data_folder + "benchmark/" + file_name + "_benchmark.csv"
add_path_to_metadata(train_file_path, par_folder, is_train=True)
add_path_to_metadata(benchmark_file_path, par_folder, is_train=False)