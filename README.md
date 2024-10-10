The directory contains files used for feature extraction from images, generation of difference vectors from D-MAD pairs, generation and testing of SVM models from difference vectors.


Files contained in the directory:

Generation of DET curves
DET.py

Extracts beauty features from modified VGG-16 model trained on SCUT-FBP 5500
extract_512_vectors.py

Extracts arcface feaure vectors from ONNX model trained on LFW with Arcface loss function
extract_arcface_vectors.py

Extracts CLS tokens from pretrained vit_base_patch16_224_dino
extract_dino_base_vectors.py

Extracts CLS tokens from pretrained dinov2_vitb14
extract_dinov2_base_vectors.py

Conversion of extracted CLS tokens or feature vectors to difference vectors from D-MAD pairs, in the form of a .libsvm file which can either be used for generating svm models in the other files or directly using libsvm executables
vectors_to_svm.py

SVM model generation and testing from FRGC and FERET datasets, comparing performance of ArcFace and ArcFace combined with VGG16 beauty vectors
train_test_svm_cropped_combined_DET.py

SVM model generation and testing from FRGC, FERET and MSYNM datasets, comparing performance of ArcFace and ArcFace combined with VGG16 beauty vectors
train_test_svm_MSYNM.py

SVM model generation and testing from FRGC, FRLL and FERET datasets, comparing performance of ArcFace and ArcFace combined with VGG16 beauty vectors
train_test_svm_FRLL.py

SVM model generation and testing from FRGC and FERET datasets, comparing performance of ArcFace and DINO
train_test_svm_dinov2.py

SVM model generation and testing from FRGC and FERET datasets, comparing performance of ArcFace and DINOv2
train_test_svm_dino.py

The folder trained_model_VGG_beauty_512 contains 2 files:

Image showing loss from training VGG16 model using SCUT-FBP 5500
beauty_rates_loss.png

VGG16 model to extract beauty vectors
VGG16_beauty_rates


How to use:
1. Extract feature vectors or CLS tokens from multiple image sets using one of the extraction scripts.
	-Folder containing the image set should contain a /img folder with all the images

2. vectors_to_svm.py can be used on 2 sets of extracted feature vectors, generating D-MAD pairs and generating difference vectors from the 2 sets labeling all image with either 0 or 1 to be used for training and testing SVM models.

3. The train_test files all contains sets of tests and generation of DET curves based on extracted features which can be used to evaluate the performance of different extraction methods.

