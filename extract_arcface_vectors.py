import argparse
import os
import numpy as np
from PIL import Image
import csv
import onnxruntime
import cv2
import onnx

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='path to the ONNX model')
parser.add_argument('--dataset', type=str, help='path to the dataset we want to label')
opt = parser.parse_args()
print(opt)

# Define preprocessing parameters
input_mean = 127.5
input_std = 127.5
input_size = (112, 112)

def preprocess_image(image):
    blob = cv2.dnn.blobFromImages([image], 1.0 / input_std, input_size, (input_mean, input_mean, input_mean), swapRB=True)
    return blob
	
# Load the ONNX model
onnx_model_path = opt.model
session = onnxruntime.InferenceSession(onnx_model_path)

onnx_model = onnx.load(onnx_model_path)
input_names = [input.name for input in onnx_model.graph.input]



images_dir = "{0}/img".format(opt.dataset)

# Create a CSV file to write the output features
csv_path = os.path.join(opt.dataset, "Arcface_vectors.csv")
with open(csv_path, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Iterate over images
    for i, file in enumerate(sorted(os.listdir(images_dir))):
        img_path = os.path.join(images_dir, file)
        img = cv2.imread(img_path)
        
        # Preprocess image
        input_data = preprocess_image(img)
        
        # Perform inference
        output = session.run(None, {input_names[0]: input_data})
        
        # Check if the output is a list
        if isinstance(output, list):
            # Flatten each element of the list
            flattened_output = [item.flatten().tolist() for item in output]
        else:
            # Flatten the output directly
            flattened_output = output.flatten().tolist()
        # Write features to CSV
        csv_writer.writerow([file] + flattened_output[0])


		
		
