import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from PIL import Image
from pathlib import Path
import random


def get_png_paths(directory):
    png_paths = []
    
    # Walk through the directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file ends with '.png' (case insensitive)
            if file.lower().endswith('.png'):
                # Construct the full path and add it to the list
                full_path = os.path.join(root, file)
                png_paths.append(full_path)
    
    return png_paths


data_directory = Path("data/rugd")
anno_directory = Path("data/RUGD_annotations")

data_paths = get_png_paths(data_directory)    
anno_paths = get_png_paths(anno_directory)


class TerrainDataset(Dataset):
    def __init__(self, image_paths, target_paths):
        self.image_paths = image_paths
        self.target_paths = target_paths

    def transform(self, image, mask):
        # Resize
        resize = v2.Resize(size=(520, 520))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = v2.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = v2.functional.crop(image, i, j, h, w)
        mask = v2.functional.crop(mask, i, j, h, w)
        
        # Transform to tensor
        image = v2.functional.pil_to_tensor(image)
        mask = v2.functional.pil_to_tensor(mask)
        
        if random.random() > 0.5:
            image = v2.functional.hflip(image)
            mask = v2.functional.hflip(mask)

        if random.random() > 0.5:
            image = v2.functional.vflip(image)
            mask = v2.functional.vflip(mask)

        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)
    
dataset = TerrainDataset(data_paths, anno_paths)
image1, image2 = next(iter(dataset))