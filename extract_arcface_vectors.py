import argparse
import os
import numpy as np
import csv
import onnx
import onnxruntime
import cv2
from PIL import Image

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to the ONNX model')
parser.add_argument('--dataset', type=str, help='Path to the dataset for labeling')
opt = parser.parse_args()
print(opt)

# Preprocessing parameters for ONNX model input
input_mean = 127.5
input_std = 127.5
input_size = (112, 112)

def preprocess_image(image):
    """Preprocesses an image for ONNX model inference."""
    return cv2.dnn.blobFromImages(
        [image], 1.0 / input_std, input_size,
        (input_mean, input_mean, input_mean), swapRB=True
    )

# Load the ONNX model
onnx_model_path = opt.model
session = onnxruntime.InferenceSession(onnx_model_path)

# Extract input names for inference
onnx_model = onnx.load(onnx_model_path)
input_names = [input.name for input in onnx_model.graph.input]

# Directory for input images
images_dir = os.path.join(opt.dataset, "img")

# CSV file path for saving output features
csv_path = os.path.join(opt.dataset, "Arcface_vectors.csv")
with open(csv_path, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Process each image in the dataset
    for i, file in enumerate(sorted(os.listdir(images_dir))):
        img_path = os.path.join(images_dir, file)
        img = cv2.imread(img_path)
        
        # Preprocess the image
        input_data = preprocess_image(img)
        
        # Run model inference
        output = session.run(None, {input_names[0]: input_data})
        
        # Flatten the output data for CSV
        flattened_output = output[0].flatten().tolist() if isinstance(output, list) else output.flatten().tolist()
        
        # Write the image filename and output features to CSV
        csv_writer.writerow([file] + flattened_output)

        # Progress update
        if i % 100 == 0:
            print(f'{i} images processed')
