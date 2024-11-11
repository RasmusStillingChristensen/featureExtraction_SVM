import argparse
import os
import csv
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# Parse dataset path argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Path to the dataset for labeling')
opt = parser.parse_args()
print(opt)

# Preprocessing parameters for ViT (DINO)
input_size = (224, 224)  # ViT expects 224x224 images
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the DINO ViT-Base pre-trained model
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
model.eval()  # Set model to evaluation mode

# Preprocess an image to prepare it for the model
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

# Directory containing the images
images_dir = os.path.join(opt.dataset, "img")

# CSV file path for saving CLS token features
csv_path = os.path.join(opt.dataset, "DINO_vit_base_cls_features.csv")
with open(csv_path, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Process each image in the dataset
    for i, file in enumerate(sorted(os.listdir(images_dir))):
        img_path = os.path.join(images_dir, file)
        
        # Preprocess the image
        input_data = preprocess_image(img_path)
        
        # Extract CLS token features through model inference
        with torch.no_grad():
            features = model.forward_features(input_data)
            cls_token = features[:, 0, :]  # CLS token is the first token

        # Flatten CLS token feature vector for CSV
        flattened_cls_token = cls_token.flatten().tolist()
        
        # Write image filename and CLS token features to the CSV
        csv_writer.writerow([file] + flattened_cls_token)

        # Print progress for every 100 images
        if i % 100 == 0:
            print(f"Processed {i} images...")

print("CLS token extraction completed.")
