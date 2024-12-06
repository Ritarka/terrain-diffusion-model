import os
import random
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import v2

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


info = get_encoder_decoder()
random_kernels = [k for k in range(0, 7, 2)]
data_directory = Path("data/rugd")

os.mkdir("blurry_images")

for path in tqdm(get_png_paths(data_directory)):
    image = Image.open(path)
    hazy = test_image(info, image, random.random() * 2 + 1, random.random() * 105 + 150)
    hazy = hazy.filter(ImageFilter.GaussianBlur(radius = random.choice(random_kernels)))
    
    path_name = Path(path).name
    
    hazy.save(f"blurry_images/{path_name}")
