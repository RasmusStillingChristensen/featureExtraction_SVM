from __future__ import print_function, division
import argparse
import os
import csv
from statistics import mean
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, models
from torch.autograd import Variable
import numpy as np
from PIL import Image

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to the trained VGG16 model')
parser.add_argument('--dataset', type=str, help='Path to the dataset for labeling')
opt = parser.parse_args()
print(opt)

# Configure device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Transform: resize, normalize, and prepare input images for VGG16
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load and modify VGG-16 model
vgg16 = models.vgg16_bn(pretrained=True)

# Freeze all layers in the feature extractor
for param in vgg16.features.parameters():
    param.requires_grad = False

# Modify the classifier to output a 512-dim vector followed by a 60-dim vector
num_features = vgg16.classifier[6].in_features
vgg16.classifier = nn.Sequential(
    *list(vgg16.classifier.children())[:-1],  # Remove the final layer
    nn.Linear(num_features, 512),
    nn.Linear(512, 60)
)

# Move the model to GPU if available
if torch.cuda.device_count() > 1:
    print("Running on", torch.cuda.device_count(), "GPUs.")
    vgg16 = nn.DataParallel(vgg16)
else:
    print("Running on CPU.")
vgg16.to(device)

# Load pretrained weights
vgg16.load_state_dict(torch.load(opt.model))

# Adjust classifier to extract the 512-dim feature vector
vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
vgg16.eval()

# Paths and dataset preparation
images_dir = os.path.join(opt.dataset, "img")
number_of_images = len(os.listdir(images_dir))
files, neurons = [], []

# Process each image in the dataset
for i, file in enumerate(sorted(os.listdir(images_dir))):
    # Load and transform image
    img = Image.open(os.path.join(images_dir, file)).convert('RGB')
    img = transform(img)
    img = torch.unsqueeze(img, 0).to(device)

    # Inference to obtain feature vector
    with torch.no_grad():
        output = vgg16(img)
    output_list = output.cpu().numpy().tolist()[0]
    output_list = [round(x, 10) for x in output_list]  # Round to optimize data size

    # Collect results
    files.append(file)
    neurons.append(output_list)

    # Progress update
    if i % 100 == 0:
        print(f'{i}/{number_of_images} images processed')

# Convert feature vectors to CSV format
csv_lines = []
for j in range(number_of_images):
    scores = ','.join([f'{neurons[j][i] * 1.0}' for i in range(512)])
    csv_lines.append(f'{files[j]},{scores}')

# Write the CSV file
csv_path = os.path.join(opt.dataset, "Vectors_512_removed_layer_cropped.csv")
with open(csv_path, "w") as csv_file:
    for line in csv_lines:
        csv_file.write(f'{line}\n')
