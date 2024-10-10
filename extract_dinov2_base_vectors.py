import argparse
import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Parse the dataset path argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='../datasets/CelebA-HQ', help='path to the dataset we want to label')
opt = parser.parse_args()
print(opt)

# Define preprocessing parameters for DINOv2 (ViT-B/14 expects 224x224 images)
input_size = (224, 224)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the DINOv2 ViT-B/14 pre-trained model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.eval()  # Set the model to evaluation mode

# Function to preprocess an image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Directory containing the images
images_dir = os.path.join(opt.dataset, "img")

# Create a CSV file to write the output features
csv_path = os.path.join(opt.dataset, "DINOv2_vit_b14_cls_features.csv")
with open(csv_path, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Iterate over images in the directory
    for i, file in enumerate(sorted(os.listdir(images_dir))):
        img_path = os.path.join(images_dir, file)
        
        # Preprocess image
        input_data = preprocess_image(img_path)
        
        # Perform inference and extract CLS token features
        with torch.no_grad():
            features = model(input_data)  # Forward pass, DINOv2 outputs features

			# Check the dimensions of the output
            if features.dim() == 3:
				# If the output is 3D (batch, tokens, feature_dim), extract the CLS token
                cls_token = features[:, 0, :]  # CLS token is the first token
            elif features.dim() == 2:
				# If the output is 2D (batch, feature_dim), use the features directly
                cls_token = features  # Already represents the CLS token or pooled features
            else:
                raise ValueError(f"Unexpected feature dimensions: {features.shape}")
			
        # Flatten the CLS token feature vector to a 1D list
        flattened_cls_token = cls_token.flatten().tolist()
        
        # Write the image filename and CLS token features to the CSV
        csv_writer.writerow([file] + flattened_cls_token)

        if i % 100 == 0:
            print(f"Processed {i} images...")

print("CLS token extraction completed.")
