import os
import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset

from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm

from dataset import TerrainDataset


normal_directory = Path("data/rugd")
blurry_directory = Path("blurry_images")

dataset = TerrainDataset(normal_directory, blurry_directory)

mini_height, mini_width = 100000, 100000

for original, blurry in dataset:
    heighta, widtha = original.size
    heightb, widthb = blurry.size
    
    mini_height = min(mini_height, heighta, heightb)
    mini_width = min(mini_width, widtha, widthb)
    
print(mini_height, mini_width)

exit()

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


transform = transforms.Compose([
    transforms.RandomCrop((512, 512)),
    transforms.Resize((512, 512), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.40378464, 0.40370963, 0.40156843), (0.27393203, 0.27571777, 0.27818038)),
])


# Create DataLoader objects for each subset
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


for image, blur_image, aux in tqdm(train_loader):
    pass
