import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from dataset import TerrainDataset

import imageio

from torchvision import transforms
from torch.utils.data import DataLoader

from models import DCGenerator, DCDiscriminator

import shutil

def remove_directory_if_exists(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Removed directory: {path}")
    else:
        print(f"Directory does not exist: {path}")


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


def create_image_grid(array, ncols=None):
    """Useful docstring (insert there)."""
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros(
        (cell_h * nrows, cell_w * ncols, channels),
        dtype=array.dtype
    )
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[
                i * cell_h:(i + 1) * cell_h,
                j * cell_w:(j + 1) * cell_w, :
            ] = array[i * ncols + j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result

def save_images(images, name):
    grid = create_image_grid(images.detach().numpy())
    grid = np.uint8(255 * (grid + 1) / 2)
    imageio.imwrite(name, grid)



### Setup Dataloader
normal_directory = Path("data/rugd")
blurry_directory = Path("blurry_images")

transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = TerrainDataset(normal_directory, blurry_directory, transform=transform)

dataloader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)



G = DCGenerator()
G.load_state_dict(torch.load("checkpoints_vanilla/G_iter16000.pkl"))

remove_directory_if_exists("unblurry_images")
os.mkdir("unblurry_images")

with torch.no_grad():
    i = 0
    for original, blurry in tqdm(dataloader):
        unhazy = G(blurry)
        combined = torch.cat([blurry, unhazy], dim=0)
        # print(unhazy.shape)
        save_images(combined, f"unblurry_images/{i}.png")
        # print("Saved")    
        i += 1