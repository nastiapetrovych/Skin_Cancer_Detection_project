import os
import cv2
import pydicom
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="Invalid value for VR UI:")

def remove_hair(image):
    """
    Remove hair artifacts from an image using the DullRazor approach.

    Args:
        image (numpy.ndarray): Input RGB image.

    Returns:
        numpy.ndarray: Image with hair artifacts removed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, thresh, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return inpainted


class HairRemovalTransform:
    """
    Custom transformation to apply hair removal on images.
    """
    def __call__(self, img):
        img_np = np.array(img)
        img_np = remove_hair(img_np)
        img = Image.fromarray(img_np)
        return img

class CassavaDataset(Dataset):
    """
    Helper class for DataLoader to handle loading images and their corresponding labels.
    """
    def __init__(self, df, dataset_paths, transforms=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            dataset_paths (dict): Mapping from dataset source to base image paths.
            transforms: Transformations to apply to images.
        """
        super().__init__()
        self.df_data = df
        self.dataset_paths = dataset_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        # Extract data
        row = self.df_data.iloc[index]
        img_path = row["image_path"]
        label = row.get("target", -1)  # Use -1 if no label is available (e.g., for test data)

        # Load image
        if img_path.endswith(".dcm"):
            dicom = pydicom.dcmread(img_path)
            img = dicom.pixel_array  # Extract pixel data
            img = (img / img.max() * 255).astype(np.uint8)  # Normalize to 0-255
            img = Image.fromarray(img).convert("RGB")  # Convert to RGB format
        else:
            img = Image.open(img_path).convert("RGB")  # Handle regular image files

        if self.transforms:
            img = self.transforms(img)

        return img, label

def combine_datasets(*datasets):
    """
    Combine multiple datasets into a single DataFrame.

    Args:
        *datasets: DataFrames of datasets to be combined.

    Returns:
        pd.DataFrame: Combined dataset.
    """
    return pd.concat(datasets, ignore_index=True)

def create_datasets(dataframes, dataset_paths, transforms_train, transforms_valid):
    """
    Create dataset objects for training, validation, and testing.

    Args:
        dataframes (dict): Dictionary containing train, validation, and test DataFrames.
        dataset_paths (dict): Dictionary with paths to the respective datasets.
        transforms_train: Transformations to apply to training data.
        transforms_valid: Transformations to apply to validation and test data.

    Returns:
        tuple: Train, validation, and test datasets.
    """
    train_dataset = CassavaDataset(dataframes["train"], dataset_paths, transforms=transforms_train)
    valid_dataset = CassavaDataset(dataframes["valid"], dataset_paths, transforms=transforms_valid)
    test_dataset = CassavaDataset(dataframes["test"], dataset_paths, transforms=transforms_valid)
    return train_dataset, valid_dataset, test_dataset

def initialize_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers):
    """
    Create DataLoader objects for training, validation, and testing.

    Args:
        train_dataset: Training dataset.
        valid_dataset: Validation dataset.
        test_dataset: Testing dataset.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader

def prepare_combined_dataset(dataset_paths, batch_size, num_workers):
    """
    Prepare datasets and loaders for individual or combined datasets with varying structures.

    Args:
        dataset_paths (dict): Dictionary containing paths for individual datasets (e.g., ISIC2020, ISIC2024, Synthetic).
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for data loading.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """

    def load_isic2020():
        train_csv = os.path.join(dataset_paths["ISIC2020"], "train.csv")
        train_df = pd.read_csv(train_csv)
        train_df = train_df.rename(columns={"image": "image_path", "label": "target"})[["image_path", "target"]]

        return train_df

    def load_isic2024():
        train_csv = os.path.join(dataset_paths["ISIC2024"], "train-metadata.csv")
        train_df = pd.read_csv(train_csv)
        train_df["image_path"] = train_df["isic_id"].apply(lambda x: os.path.join(dataset_paths["ISIC2024"], "train-image", "image", f"{x}.jpg"))
        train_df = train_df[["image_path", "target"]]
        malignant_df = train_df[train_df["target"] == 1]
        benign_df = train_df[train_df["target"] == 0].sample(n=7000, random_state=42)
        combined_df = pd.concat([malignant_df, benign_df], axis=0).reset_index(drop=True)

        return combined_df

    def load_synthetic():
        train_csv = os.path.join(dataset_paths["Synthetic"], "train.csv")
        train_df = pd.read_csv(train_csv)
        train_df = train_df.rename(columns={"image": "image_path", "label": "target"})[["image_path", "target"]]
        malignant_df = train_df[train_df["target"] == 1]

        return malignant_df

    def add_source_column(df, source):
        df["source"] = source
        return df

    # Load datasets
    isic2020_train = load_isic2020()
    isic2024_train = load_isic2024()
    synthetic_train = load_synthetic()

    # Add source identifiers
    isic2020_train = add_source_column(isic2020_train, "ISIC2020")
    isic2024_train = add_source_column(isic2024_train, "ISIC2024")
    synthetic_train = add_source_column(synthetic_train, "Synthetic")

    # Combine data
    merged_dataset = pd.concat([isic2020_train, isic2024_train, synthetic_train], ignore_index=True)

    print("\nSummary:")
    print(f"Total records with target == 0: {len(merged_dataset[merged_dataset['target'] == 0])}")
    print(f"Total records with target == 1: {len(merged_dataset[merged_dataset['target'] == 1])}")
    print("Record counts by source type:")
    print(merged_dataset['source'].value_counts())

    # Split combined training data into train and validation
    train_val_data, test_data = train_test_split(
        merged_dataset, test_size=0.2, stratify=merged_dataset["target"], random_state=42
    )

    train_data, valid_data = train_test_split(
        train_val_data, test_size=0.2, stratify=train_val_data['target'], random_state=42
    )

    # Create DataFrames
    dataframes = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
    }

    # Create datasets and DataLoaders
    train_dataset, valid_dataset, test_dataset = create_datasets(
        dataframes,
        dataset_paths,
        transforms_train,
        transforms_valid,
    )

    return initialize_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers)

transforms_train = transforms.Compose([
    HairRemovalTransform(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transforms_valid = transforms.Compose([
    HairRemovalTransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])