import os
import torch
from torch.utils.data import random_split, DataLoader, Dataset

from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm

from dataloader import TerrainDataset


data_directory = Path("data/rugd")
anno_directory = Path("data/RUGD_annotations")

dataset = TerrainDataset(data_directory, anno_directory)

train_size = int(0.7 * len(dataset))  # 70% for training
val_size = int(0.15 * len(dataset))   # 15% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 15% for testing

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Print dataset sizes to confirm
print(f"Train Dataset Size: {len(train_dataset)}")
print(f"Validation Dataset Size: {len(val_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")



# Create DataLoader objects for each subset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for image, blur_image in tqdm(train_loader):
    pass

# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

