from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import get_png_paths

def compute_mean_std(image_paths):
    """
    Computes the mean and standard deviation of a dataset of images.
    
    Args:
        image_paths (list of str): List of file paths to the images.
        
    Returns:
        tuple: (mean, std) of the dataset across all channels.
    """
    pixel_sum = np.zeros(3)  # Assuming RGB images
    pixel_squared_sum = np.zeros(3)
    total_pixels = 0

    for path in tqdm(image_paths):
        # Open the image and convert it to RGB
        image = Image.open(path).convert('RGB')
        image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

        # Compute the sum and squared sum of pixels for each channel
        pixel_sum += image_array.sum(axis=(0, 1))
        pixel_squared_sum += (image_array ** 2).sum(axis=(0, 1))
        total_pixels += image_array.shape[0] * image_array.shape[1]  # Number of pixels

    # Calculate mean and std for each channel
    mean = pixel_sum / total_pixels
    variance = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    return mean, std

# Example usage
image_paths = get_png_paths("data/rugd")
mean, std = compute_mean_std(image_paths)
print("Mean:", mean)
print("Standard Deviation:", std)