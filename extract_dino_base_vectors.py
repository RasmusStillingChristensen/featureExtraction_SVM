import argparse
import os
import csv
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms

# Parse the dataset path argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='path to the dataset we want to label')
opt = parser.parse_args()
print(opt)

# Define preprocessing parameters for ViT (DINO)
input_size = (224, 224)  # ViT expects 224x224 images
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the DINO ViT-Base pre-trained model
model = timm.create_model('vit_base_patch16_224_dino', pretrained=True)
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
csv_path = os.path.join(opt.dataset, "DINO_vit_base_cls_features.csv")
with open(csv_path, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Iterate over images in the directory
    for i, file in enumerate(sorted(os.listdir(images_dir))):
        img_path = os.path.join(images_dir, file)
        
        # Preprocess image
        input_data = preprocess_image(img_path)
        
        # Perform inference and extract CLS token features
        with torch.no_grad():
            features = model.forward_features(input_data)  # Extract features
            cls_token = features[:, 0, :]  # CLS token is the first token

        # Flatten the CLS token feature vector to a 1D list
        flattened_cls_token = cls_token.flatten().tolist()
        
        # Write the image filename and CLS token features to the CSV
        csv_writer.writerow([file] + flattened_cls_token)

        if i % 100 == 0:
            print(f"Processed {i} images...")

print("CLS token extraction completed.")
