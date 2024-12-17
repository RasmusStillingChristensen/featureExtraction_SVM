@echo off
::Extract DINOv2 classification tokens
python extract_dinov2_base_vectors.py --dataset .\Example\FRLL_samples\bonafide_neutral_front
python extract_dinov2_base_vectors.py --dataset .\Example\FRLL_samples\morph_facemorpher
python extract_dinov2_base_vectors.py --dataset .\Example\FRLL_samples\probe_smiling_front

::Generate and label difference vectors
python vectors_to_svm.py --trusted .\Example\FRLL_samples\probe_smiling_front\DINOv2_vit_b14_cls_features.csv --suspected .\Example\FRLL_samples\bonafide_neutral_front\DINOv2_vit_b14_cls_features.csv --dataset FRLL --label 0 --savefile .\Example\FRLL_bonafide.libsvm
python vectors_to_svm.py --trusted .\Example\FRLL_samples\probe_smiling_front\DINOv2_vit_b14_cls_features.csv --suspected .\Example\FRLL_samples\morph_facemorpher\DINOv2_vit_b14_cls_features.csv --dataset FRLL --label 1 --savefile .\Example\FRLL_facemorpher.libsvm

::Generate SVM model for classifying vectors
python train_svm.py --config .\Example\ExampleConfig.txt --model .\Example\ExampleSVM.pkl
python test_svm.py --config .\Example\ExampleConfig.txt --model .\Example\ExampleSVM.pkl