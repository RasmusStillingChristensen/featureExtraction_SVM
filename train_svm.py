import argparse
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import vstack, csr_matrix

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

def train_svm_model(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize and train SVM model with RBF kernel
    svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return svm_model, accuracy

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--model', required=True, help='Name of the generated SVM model')
    opt = parser.parse_args()

    data_files = parse_config(opt.config)

    # Load training data
    training_files = data_files.get("training data", [])
    all_features = []
    all_labels = []

    for f in training_files:
        features, labels = load_svmlight_file(f)
        all_features.append(features)
        all_labels.append(labels)
    
    if not all_features:
        raise ValueError("No training data found in config file.")

    # Combine features and labels into single matrices
    combined_features = vstack(all_features)
    combined_labels = np.hstack(all_labels)

    # Train the SVM model
    svm_model, accuracy = train_svm_model(combined_features, combined_labels)

    print(f"Accuracy of model: {accuracy:.2f}")

    # Save the trained model
    with open(opt.model, 'wb') as file:
        pickle.dump(svm_model, file)
    print(f"Model saved to {opt.model}")

if __name__ == "__main__":
    main()
