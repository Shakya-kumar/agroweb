import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import numpy as np

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def prepare_dataset():
    # Create necessary directories
    os.makedirs('dataset/train/healthy', exist_ok=True)
    os.makedirs('dataset/train/diseased', exist_ok=True)
    os.makedirs('dataset/test/healthy', exist_ok=True)
    os.makedirs('dataset/test/diseased', exist_ok=True)
    
    # Download PlantVillage dataset
    print("Downloading PlantVillage dataset...")
    url = "https://storage.googleapis.com/plantvillage-dataset/plantvillage-dataset.zip"
    zip_path = "plantvillage-dataset.zip"
    download_file(url, zip_path)
    
    # Extract the dataset
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp")
    
    # Process and organize the images
    print("Organizing dataset...")
    # We'll focus on a few common crops for this example
    target_crops = ['Tomato', 'Potato', 'Corn']
    
    for crop in target_crops:
        crop_dir = os.path.join("temp", "plantvillage dataset", "color", crop)
        if not os.path.exists(crop_dir):
            continue
            
        for image_file in os.listdir(crop_dir):
            src_path = os.path.join(crop_dir, image_file)
            
            # Determine if the image is healthy or diseased
            is_healthy = "healthy" in image_file.lower()
            dest_dir = "dataset/train" if np.random.random() < 0.8 else "dataset/test"
            dest_dir = os.path.join(dest_dir, "healthy" if is_healthy else "diseased")
            
            # Copy the image to the appropriate directory
            shutil.copy2(src_path, os.path.join(dest_dir, image_file))
    
    # Clean up
    print("Cleaning up...")
    shutil.rmtree("temp")
    os.remove(zip_path)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    prepare_dataset() 