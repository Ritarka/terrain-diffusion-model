import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt

from haze_synthesis.main import get_encoder_decoder, test_image


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


class TerrainDataset(Dataset):
    def __init__(self, data_directory, anno_directory):
        self.image_paths = get_png_paths(data_directory)
        self.target_paths = get_png_paths(anno_directory)

        self.info = get_encoder_decoder()
        self.encoder, self.decoder, self.height, self.width = self.info
        
        self.random_kernels = [k for k in range(1, 6, 2)]

    def transform(self, image, mask):
        # Resize
        resize = v2.Resize(size=(520, 520))
        image = resize(image)
        mask = image

        # Random crop
        i, j, h, w = v2.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = v2.functional.crop(image, i, j, h, w)
        mask = v2.functional.crop(mask, i, j, h, w)
        
        if random.random() > 0.5:
            # recommended: beta = [1.0,3.0], airlight = [150,255]
            mask = test_image(self.info, mask, random.random() * 2 + 1, random.random() * 105 + 150)

        # Transform to tensor
        image = v2.functional.pil_to_tensor(image)
        mask = v2.functional.pil_to_tensor(mask)
        
        if random.random() > 0.5:
            image = v2.functional.hflip(image)
            mask = v2.functional.hflip(mask)

        if random.random() > 0.5:
            image = v2.functional.vflip(image)
            mask = v2.functional.vflip(mask)

        if random.random() > 0.5:            
            mask = v2.GaussianBlur(kernel_size=random.choice(self.random_kernels))(mask)

        image = v2.ToPILImage()(image)
        mask = v2.ToPILImage()(mask)
        plt.imshow(image)
        plt.savefig("image.png")
        plt.clf()
        
        plt.imshow(mask)
        plt.savefig("mask_image.png")
        exit()
            
            
        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.image_paths)
    