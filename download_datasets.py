import os
import pandas as pd
import zipfile
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi

def unzip_all_in_directory(directory):
    """
    Recursively unzip all ZIP files in the directory.
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if zipfile.is_zipfile(item_path):
            print(f"Unzipping nested ZIP file: {item_path}")
            with zipfile.ZipFile(item_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            os.remove(item_path)  # Remove the nested ZIP file after extraction
            print(f"Unzipped and cleaned up: {item_path}")
            # Call recursively in case there are more nested ZIPs
            unzip_all_in_directory(directory)

def download_isic2020(dataset, target_dir):
    # Download and unzip the dataset
    api.dataset_download_files(dataset, path=target_dir, unzip=True)
    
    # Paths to train and test directories
    train_dir = os.path.join(target_dir, "train", "malignant")
    test_dir = os.path.join(target_dir, "test", "malignant")
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("The expected 'malignant' directories do not exist in the dataset structure.")

    # Collect paths and labels
    data = []
    
    for img_name in os.listdir(train_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Adjust extensions as needed
            data.append({"image": os.path.join(target_dir, "train", "malignant", img_name), "label": 1})
    
    for img_name in os.listdir(test_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            data.append({"image": os.path.join(target_dir, "test", "malignant", img_name), "label": 1})
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Save to train.csv
    csv_path = os.path.join(target_dir, "train.csv")
    df.to_csv(csv_path, index=False)
    print(f"'train.csv' created with {len(df)} malignant images at: {csv_path}")

def download_competition(competition, target_dir):
    zip_path = f"{target_dir}/{competition}.zip"
    
    # Check if the directory exists
    if os.listdir(target_dir):
        print(f"Checking contents of {target_dir}...")
        # If the directory contains only a ZIP file, unzip it
        contains_only_zip = all(zipfile.is_zipfile(os.path.join(target_dir, f)) for f in os.listdir(target_dir))
        if contains_only_zip:
            print(f"Directory contains ZIP file(s). Proceeding to unzip.")
            unzip_all_in_directory(target_dir)
            print(f"Unzipping completed for existing ZIP file(s) in {target_dir}.")
            return
        else:
            print(f"Files already exist in {target_dir}. Skipping download.")
            return

    print(f"Downloading competition: {competition} into {target_dir}")
    api.competition_download_files(competition, path=target_dir)
    print(f"Download completed for {competition}")

    # Unzip the dataset
    print(f"Unzipping files in {target_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(zip_path)  # Clean up the main ZIP file
    print(f"Unzipping completed and cleaned up main ZIP file.")

    # Handle nested ZIP files
    print(f"Checking for nested ZIP files in {target_dir}")
    unzip_all_in_directory(target_dir)
    print(f"All nested ZIP files unzipped in {target_dir}.")

def process_synthetic_dataset(target_dir):
    """
    Downloads the synthetic dataset, extracts all ZIP files, and updates paths in the CSV files.
    """
    print("Downloading synthetic dataset from Hugging Face...")
    ds = load_dataset("MAli-Farooq/Derm-T2IM-Dataset")

    # Create an extraction directory
    extraction_dir = os.path.join(target_dir, "images")
    os.makedirs(extraction_dir, exist_ok=True)

    label_mapping = {
        0: 0,  # Benign
        1: 1,  # Malignant
        2: 1,  # Malignant
        3: 0,  # Benign
        4: 0   # Benign
    }

    # Process each split (e.g., train, test)
    for split in ds:
        output_csv_path = os.path.join(target_dir, f"{split}.csv")
        df = ds[split].to_pandas()

        print(f"Processing {split} split...")
        print(f"Columns in the dataset: {df.columns.tolist()}")

        df["label"] = df["label"].map(label_mapping)

        # Extract ZIP paths from the 'image' column
        def get_zip_path(image_dict):
            if isinstance(image_dict, dict) and 'path' in image_dict:
                return image_dict['path'].split("::")[-1].replace("zip://", "").split("::")[0]
            else:
                raise ValueError(f"Unexpected format in 'image' column: {image_dict}")

        zip_paths = df['image'].apply(get_zip_path).unique()

        # Extract all ZIP files
        for zip_path in zip_paths:
            zip_path_resolved = os.path.expanduser(zip_path)  # Handle '~' for home directory
            print(f"Extracting ZIP file: {zip_path_resolved}")
            with zipfile.ZipFile(zip_path_resolved, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)

        # Update paths in the DataFrame
        def update_path(row):
            image_dict = row['image']
            if isinstance(image_dict, dict) and 'path' in image_dict:
                original_path = image_dict['path'].split("::")[0].replace("zip://", "")
                return os.path.join(extraction_dir, original_path)
            else:
                raise ValueError(f"Unexpected format in 'image' column: {image_dict}")

        df['image'] = df.apply(update_path, axis=1)

        # Save the updated CSV
        df.to_csv(output_csv_path, index=False)
        print(f"Updated and saved {split} split to {output_csv_path}.")

    print(f"Synthetic dataset processing completed.")

# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Set directories
download_dir = "./dataset"
dataset_isic_2024 = "isic-2024-challenge"
dataset_isic_2020 = "fanconic/skin-cancer-malignant-vs-benign"
dataset_isic_2024_dir = f"{download_dir}/isic2024"
dataset_isic_2020_dir = f"{download_dir}/isic2020"
dataset_synthetic_dir = f"{download_dir}/synthetic"

os.makedirs(download_dir, exist_ok=True)
os.makedirs(dataset_isic_2024_dir, exist_ok=True)
os.makedirs(dataset_isic_2020_dir, exist_ok=True)
os.makedirs(dataset_synthetic_dir, exist_ok=True)

# Download ISIC datasets (2020 & 2024)
download_competition(dataset_isic_2024, dataset_isic_2024_dir)
download_isic2020(dataset_isic_2020, dataset_isic_2020_dir)

# Process synthetic dataset
process_synthetic_dataset(dataset_synthetic_dir)

print(f"Dataset check and downloads (if needed) completed.")
