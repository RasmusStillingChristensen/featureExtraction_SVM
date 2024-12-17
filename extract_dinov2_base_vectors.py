import argparse
import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Parse dataset path argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Path to the dataset for feature extraction')
opt = parser.parse_args()
print(opt)

# Preprocessing parameters for DINOv2 (ViT-B/14 expects 224x224 images)
input_size = (224, 224)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the DINOv2 ViT-B/14 pre-trained model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.eval()  # Set model to evaluation mode

# Preprocess an image for the model
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    return img.unsqueeze(0)  # Add batch dimension

# Directory containing images
images_dir = os.path.join(opt.dataset, "img")

# CSV file path for saving CLS token features
csv_path = os.path.join(opt.dataset, "DINOv2_vit_b14_cls_features.csv")
with open(csv_path, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Process each image in the dataset
    for i, file in enumerate(sorted(os.listdir(images_dir))):
        img_path = os.path.join(images_dir, file)
        
        # Preprocess the image
        input_data = preprocess_image(img_path)
        
        # Perform inference and extract CLS token features
        with torch.no_grad():
            features = model(input_data)  # DINOv2 model inference
            
            # Extract CLS token depending on the output dimensions
            if features.dim() == 3:  # 3D output (batch, tokens, feature_dim)
                cls_token = features[:, 0, :]  # CLS token is the first token
            elif features.dim() == 2:  # 2D output (batch, feature_dim)
                cls_token = features  # Features directly represent the CLS token or pooled output
            else:
                raise ValueError(f"Unexpected feature dimensions: {features.shape}")
            
        # Flatten the CLS token feature vector to a 1D list
        flattened_cls_token = cls_token.flatten().tolist()
        
        # Write image filename and CLS token features to the CSV
        csv_writer.writerow([file] + flattened_cls_token)

        # Print progress every 100 images
        if i % 100 == 0:
            print(f"Processed {i} images...")

print("CLS token extraction completed.")
