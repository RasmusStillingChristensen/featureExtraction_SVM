import argparse
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack, csr_matrix
from DET import DET
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_svmlight_file(file_path):
    # Initialize empty lists to store features and labels
    features = []
    labels = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            label = float(parts[0])  # Convert label to float
            pairs = parts[1:]         # Extract feature-value pairs
            
            feature_vector = {}
            for pair in pairs:
                idx, value = pair.split(':')
                feature_vector[int(idx) - 1] = float(value)
            
            labels.append(label)
            features.append(feature_vector)

    # Convert to sparse matrix
    sparse_features = csr_matrix((len(features), max(max(f.keys()) for f in features) + 1))
    for i, feature_dict in enumerate(features):
        for idx, value in feature_dict.items():
            sparse_features[i, idx] = value

    return sparse_features, np.array(labels)
	
def parse_config(file_path):
    data_files = {"training data": [], "testing data": []}
    current_category = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.lower().endswith(":"):
                current_category = line[:-1].strip().lower()
                if current_category not in data_files:
                    data_files[current_category] = []
            elif current_category:
                data_files[current_category].append(line)
    return data_files

def adjust_scores_for_DET(scores_array, scores_type):
    scores_array = np.asarray(scores_array)
    if scores_type == "similarity":
        return scores_array
    elif scores_type == "dissimilarity":
        return -scores_array
    else:
        raise ValueError(f"Unknown type of comparison scores: {scores_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--model', required=True, help='Path to the SVM model used for testing')
    opt = parser.parse_args()

    data_files = parse_config(opt.config)

    # Load training data
    testing_files = data_files.get("testing data", [])
    all_features = []
    all_labels = []

    for f in testing_files:
        features, labels = load_svmlight_file(f)
        all_features.append(features)
        all_labels.append(labels)

    if not all_features:
        raise ValueError("No training data found in config file.")

    # Combine features and labels into single matrices
    combined_features = vstack(all_features)
    combined_labels = np.hstack(all_labels)
	# Split into bonafide and morphed based on labels
    bonafide_mask = combined_labels == 0  # Boolean mask for bonafide (label 0)
    morphed_mask = combined_labels == 1   # Boolean mask for morphed (label 1)
    bonafide_features = combined_features[bonafide_mask]
    morphed_features = combined_features[morphed_mask]


	# Load SVM model using pickle
    with open(opt.model, 'rb') as file:
        SVM_model = pickle.load(file)

    det = DET()
    det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Example of DET curve")
    det.x_limits = np.array([1e-4, 0.99])
    det.y_limits = np.array([1e-4, 0.99])
    det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
    det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
    det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
    det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
    det.create_figure()
    det.plot(tar=adjust_scores_for_DET(SVM_model.predict_proba(bonafide_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(SVM_model.predict_proba(morphed_features)[:, 1], "dissimilarity"), label='SVM model performance')
    det.legend_on(loc="upper right")
    det.show()

if __name__ == "__main__":
    main()
