import os
from torch.utils.data import Dataset
from torchvision import transforms


from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

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
    def __init__(self, normal_path, blurry_path, transform=None):
        assert normal_path.exists(), f"Directory not found: {normal_path}"
        assert blurry_path.exists(), f"Directory not found: {blurry_path}"

        self.normal_paths = get_png_paths(normal_path)
        self.blurry_path = blurry_path
        self.transform = transform
        
        self.simple_transform = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
        ])



    def __getitem__(self, index):
        image_path = self.normal_paths[index]
        name = Path(image_path).name
        # print(name)
        blurry_path = self.blurry_path / name
        # assert blurry_path.exists(), f"File not found: {blurry_path}"
        original = Image.open(image_path)
        blurry = Image.open(blurry_path)
        
        # print(image_path)
        # print(blurry_path)
        
        # self.transform = None
        if self.transform:
            original = self.transform(original)
            blurry = self.transform(blurry)
        else:
            original = self.simple_transform(original)
            blurry = self.simple_transform(blurry)
        
        return original, blurry

    def __len__(self):
        return len(self.normal_paths)
    
