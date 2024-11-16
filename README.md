# Image Feature Extraction and SVM Model Testing for Differential Morphing Attack Detection

This repository contains scripts for feature extraction from images, generation of difference vectors, and training/testing SVM models on the extracted features. The primary goal is to compare different feature extraction methods and evaluate SVM model performance using DET (Detection Error Tradeoff) curves.

## Directory Structure and Contents

### Feature Extraction Scripts
1. **`extract_512_vectors.py`** - Extracts "beauty" features from a modified VGG-16 model pre-trained on the SCUT-FBP 5500 dataset.
2. **`extract_arcface_vectors.py`** - Extracts feature vectors from an ONNX model trained on the LFW dataset with ArcFace loss.
3. **`extract_dino_base_vectors.py`** - Extracts CLS tokens from a ViT model (`vit_base_patch16_224_dino`) for feature extraction.
4. **`extract_dinov2_base_vectors.py`** - Extracts CLS tokens from the `dinov2_vitb14` model for feature extraction.

### Difference Vector Generation
- **`vectors_to_svm.py`** - Converts extracted feature vectors from paired images (D-MAD pairs) into difference vectors, suitable for training and testing SVM models. Output is in `.libsvm` format for direct compatibility with libsvm.

### DET Curve Generation
- **`DET.py`** - Generates DET curves, useful for visualizing SVM model performance across feature extraction methods.

### SVM Model Generation and Testing
These scripts compare the performance of ArcFace, VGG-16 beauty features, and DINO features across various datasets:

1. **`train_test_svm_cropped_combined_DET.py`** - SVM training and testing on FRGC and FERET datasets using ArcFace and VGG-16 beauty vectors.
2. **`train_test_svm_MSYNM.py`** - SVM model generation/testing on FRGC, FERET, and MSYNM datasets, comparing ArcFace with VGG-16 beauty vectors.
3. **`train_test_svm_FRLL.py`** - SVM model generation/testing on FRGC, FRLL, and FERET datasets, comparing ArcFace with VGG-16 beauty vectors.
4. **`train_test_svm_dinov2.py`** - SVM training/testing on FRGC and FERET datasets, comparing ArcFace and DINO.
5. **`train_test_svm_dino.py`** - SVM training/testing on FRGC and FERET datasets, comparing ArcFace and DINOv2.

## Usage Guide

### Step 1: Extract Feature Vectors
To begin, use one of the feature extraction scripts on an image dataset:
- Ensure the dataset is structured with images located in an `img/` folder.
- Run the desired script to generate feature vectors.

### Step 2: Generate Difference Vectors
Use **`vectors_to_svm.py`** on two sets of extracted feature vectors to create D-MAD pairs and difference vectors. These will be labeled for SVM training and testing.

### Step 3: Train and Test SVM Models
Use one of the SVM training/testing scripts:
- Each script performs SVM training and testing on specific datasets, generating DET curves to assess model performance.
- Models can be configured to evaluate different feature extraction methods and combinations (e.g., ArcFace vs. ArcFace + VGG-16).

## Requirements
To run the scripts in this repository, you will need the following Python packages:

```plaintext
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
Pillow>=9.0.0
opencv-python>=4.5.5
onnx>=1.10.0
onnxruntime>=1.10.0
scipy>=1.7.0
scikit-learn>=1.0.2
matplotlib>=3.5.0
pandas>=1.3.5
timm>=0.6.7
argparse>=1.4.0

## Example
The example folder contains images from the FRLL-Morphs dataset: https://paperswithcode.com/dataset/frll-morphs
- Trusted images from probe_smiling_front
- Bonafide images from bonafide_neutral_front
- Morphed images generated with FaceMorpher using images from bonafide_neutral_front

Example.bat will use DINOv2 to extract CLS tokens and generate labeled difference vectors for libsvm 