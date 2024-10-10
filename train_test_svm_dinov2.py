from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy.sparse import vstack
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from DET import DET
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler

def fill_holes_in_line(line):
    # Split the line into label and feature-value pairs
    parts = line.strip().split(' ')
    label = parts[0]
    pairs = [pair.split(':') for pair in parts[1:]]

    # Extract feature indices
    indices = [int(pair[0]) for pair in pairs]

    # Find the maximum index
    max_index = max(indices)

    # Create a dictionary to store the existing feature values
    feature_dict = {int(pair[0]): float(pair[1]) for pair in pairs}

    # Fill in the holes with zeros
    filled_pairs = []
    for i in range(max_index + 1):
        value = feature_dict.get(i, 0.0)
        filled_pairs.append(f"{i}:{value}")

    # Reconstruct the line
    filled_line = f"{label} {' '.join(filled_pairs)}"

    return filled_line


def load_svmlight_file(file_path):
    # Initialize empty lists to store features and labels
    features = []
    labels = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into label and feature-value pairs
            line = fill_holes_in_line(line)
            parts = line.strip().split(' ')
            label = float(parts[0])  # Convert label to float
            pairs = parts[1:]         # Extract feature-value pairs
            # Initialize feature vector
            feature_vector = {}

            # Parse feature-value pairs
            for pair in pairs:
                idx, value = pair.split(':')
                # Append feature value to feature vector
                feature_vector[int(idx)] = float(value)
            
            # Append label and feature vector to lists
            labels.append(label)
            features.append(feature_vector)

    return features, labels

def load_svmlight_file_2(file_path):
    # Initialize empty lists to store features and labels
    features = []
    labels = []

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into label and feature-value pairs
            parts = line.strip().split(' ')
            label = float(parts[0])  # Convert label to float
            pairs = parts[1:]         # Extract feature-value pairs
            
            # Initialize feature vector
            feature_vector = []

            # Parse feature-value pairs
            for pair in pairs:
                idx, value = pair.split(':')
                # Append feature value to feature vector
                feature_vector.append(float(value))
            
            # Append label and feature vector to lists
            labels.append(label)
            features.append(feature_vector)

    return features, labels
	
	
def load_svmlight_file_512(file_path):
    num_features = 512  # Ensure each feature vector contains 512 features
    features = []
    labels = []

    def fill_holes_in_line(line):
        # Implement the function to fill any holes in the line if needed
        # This is a placeholder and should be replaced with actual implementation
        return line

    with open(file_path, 'r') as file:
        for line in file:
            line = fill_holes_in_line(line)
            parts = line.strip().split(' ')
            label = float(parts[0])  # Convert label to float
            pairs = parts[1:]        # Extract feature-value pairs

            # Initialize a fixed-size feature vector with zeros
            feature_vector = np.zeros(num_features)

            # Parse feature-value pairs and fill the feature vector
            for pair in pairs:
                idx, value = pair.split(':')
                feature_vector[int(idx) - 1] = float(value)  # Convert index to zero-based

            labels.append(label)
            features.append(feature_vector)

    return np.array(features), np.array(labels)

def load_svmlight_file_768(file_path):
    num_features = 768  # Ensure each feature vector contains 768 features
    features = []
    labels = []

    def fill_holes_in_line(line):
        # Implement the function to fill any holes in the line if needed
        # This is a placeholder and should be replaced with actual implementation
        return line

    with open(file_path, 'r') as file:
        for line in file:
            line = fill_holes_in_line(line)
            parts = line.strip().split(' ')
            label = float(parts[0])  # Convert label to float
            pairs = parts[1:]        # Extract feature-value pairs

            # Initialize a fixed-size feature vector with zeros
            feature_vector = np.zeros(num_features)

            # Parse feature-value pairs and fill the feature vector
            for pair in pairs:
                idx, value = pair.split(':')
                feature_vector[int(idx) - 1] = float(value)  # Convert index to zero-based

            labels.append(label)
            features.append(feature_vector)

    return np.array(features), np.array(labels)

	
def load_svmlight_file_1024(file_path):
    num_features = 1024  # Ensure each feature vector contains 512 features
    features = []
    labels = []

    def fill_holes_in_line(line):
        # Implement the function to fill any holes in the line if needed
        # This is a placeholder and should be replaced with actual implementation
        return line

    with open(file_path, 'r') as file:
        for line in file:
            line = fill_holes_in_line(line)
            parts = line.strip().split(' ')
            label = float(parts[0])  # Convert label to float
            pairs = parts[1:]        # Extract feature-value pairs

            # Initialize a fixed-size feature vector with zeros
            feature_vector = np.zeros(num_features)

            # Parse feature-value pairs and fill the feature vector
            for pair in pairs:
                idx, value = pair.split(':')
                feature_vector[int(idx) - 1] = float(value)  # Convert index to zero-based

            labels.append(label)
            features.append(feature_vector)

    return np.array(features), np.array(labels)



def train_svm_model(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Convert labels to integer type
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Calculate class weights to compensate for imbalance
    class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    
    # Initialize and train SVM model with RBF kernel and class weights
    svm_model = SVC(kernel='rbf', class_weight='balanced',probability=True)
    svm_model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return svm_model, accuracy

def train_svm_model_unweighted(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Convert labels to integer type
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # Initialize and train SVM model with RBF kernel
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return svm_model, accuracy

def get_BPCER_10(sorted_list):
    # Calculate the index for the 10th percentile
    index_10th_percentile = int(len(sorted_list) * 0.10)
    # Get the value at the calculated index
    value_at_10th_percentile = sorted_list[index_10th_percentile]
    return value_at_10th_percentile

def get_BPCER_1(sorted_list):
    # Calculate the index for the 1st percentile
    index_1st_percentile = int(len(sorted_list) * 0.01)
    # Get the value at the calculated index
    value_at_1st_percentile = sorted_list[index_1st_percentile]
    return value_at_1st_percentile

def ratio_values_below_threshold(sorted_list, threshold):
    count_below_threshold = sum(value < threshold for value in sorted_list)
    total_values = len(sorted_list)
    ratio = count_below_threshold / total_values
    return ratio

def calculate_eer(bonafide_probs, morphed_probs):
    fpr, tpr, thresholds = metrics.roc_curve(np.concatenate([np.zeros_like(bonafide_probs), np.ones_like(morphed_probs)]),
                                             np.concatenate([bonafide_probs, morphed_probs]), pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def test_svm(svm_model, bonafide_features, morphed_features):
    morph_probabilities = np.sort(svm_model.predict_proba(bonafide_features)[:, 1])[::-1]
    BPCER10 = get_BPCER_10(morph_probabilities)
    BPCER1 = get_BPCER_1(morph_probabilities)

    morph_probabilities_morphed = np.sort(svm_model.predict_proba(morphed_features)[:, 1])[::-1]
    performance10 = ratio_values_below_threshold(morph_probabilities_morphed, BPCER10)
    performance1 = ratio_values_below_threshold(morph_probabilities_morphed, BPCER1)
    
    # Calculate EER
    eer = calculate_eer(morph_probabilities, morph_probabilities_morphed)

    return performance10, performance1, eer

	
def adjust_scores_for_DET(scores_array, scores_type):
    scores_array = np.asarray(scores_array)
    if scores_type == "similarity":
        return scores_array
    elif scores_type == "dissimilarity":
        return -scores_array
    else:
        raise ValueError(f"Unknown type of comparison scores: {scores_type}")

print("loading files")

FERET_bonafide_file_path = r"./FERET_bonafide_cropped_differenceVectors.libsvm"
FERET_bonafide_features, FERET_bonafide_labels = load_svmlight_file_2(FERET_bonafide_file_path)
FERET_facefusion_file_path = r"./FERET_facefusion_cropped_differenceVectors.libsvm"
FERET_facefusion_features, FERET_facefusion_labels = load_svmlight_file_2(FERET_facefusion_file_path)
FERET_facemorpher_file_path = r"./FERET_facemorpher_cropped_differenceVectors.libsvm"
FERET_facemorpher_features, FERET_facemorpher_labels = load_svmlight_file_2(FERET_facemorpher_file_path)
FERET_opencv_file_path = r"./FERET_opencv_cropped_differenceVectors.libsvm"
FERET_opencv_features, FERET_opencv_labels = load_svmlight_file_2(FERET_opencv_file_path)
FERET_ubo_file_path = r"./FERET_ubo_cropped_differenceVectors.libsvm"
FERET_ubo_features, FERET_ubo_labels = load_svmlight_file_2(FERET_ubo_file_path)

FRGC_bonafide_file_path = r"./FRGC_bonafide_cropped_differenceVectors.libsvm"
FRGC_bonafide_features, FRGC_bonafide_labels = load_svmlight_file_2(FRGC_bonafide_file_path)
FRGC_facefusion_file_path = r"./FRGC_facefusion_cropped_differenceVectors.libsvm"
FRGC_facefusion_features, FRGC_facefusion_labels = load_svmlight_file_2(FRGC_facefusion_file_path)
FRGC_facemorpher_file_path = r"./FRGC_facemorpher_cropped_differenceVectors.libsvm"
FRGC_facemorpher_features, FRGC_facemorpher_labels = load_svmlight_file_2(FRGC_facemorpher_file_path)
FRGC_opencv_file_path = r"./FRGC_opencv_cropped_differenceVectors.libsvm"
FRGC_opencv_features, FRGC_opencv_labels = load_svmlight_file_2(FRGC_opencv_file_path)
FRGC_ubo_file_path = r"./FRGC_ubo_cropped_differenceVectors.libsvm"
FRGC_ubo_features, FRGC_ubo_labels = load_svmlight_file_2(FRGC_ubo_file_path)


FERET_bonafide_arcface_file_path = r"./FERET_bonafide_Arcface_cropped_differenceVectors.libsvm"
FERET_bonafide_arcface_features, FERET_bonafide_labels = load_svmlight_file_2(FERET_bonafide_arcface_file_path)
FERET_facefusion_arcface_file_path = r"./FERET_facefusion_Arcface_cropped_differenceVectors.libsvm"
FERET_facefusion_arcface_features, FERET_facefusion_labels = load_svmlight_file_2(FERET_facefusion_arcface_file_path)
FERET_facemorpher_arcface_file_path = r"./FERET_facemorpher_Arcface_cropped_differenceVectors.libsvm"
FERET_facemorpher_arcface_features, FERET_facemorpher_labels = load_svmlight_file_2(FERET_facemorpher_arcface_file_path)
FERET_opencv_arcface_file_path = r"./FERET_opencv_Arcface_cropped_differenceVectors.libsvm"
FERET_opencv_arcface_features, FERET_opencv_labels = load_svmlight_file_2(FERET_opencv_arcface_file_path)
FERET_ubo_arcface_file_path = r"./FERET_ubo_Arcface_cropped_differenceVectors.libsvm"
FERET_ubo_arcface_features, FERET_ubo_labels = load_svmlight_file_2(FERET_ubo_arcface_file_path)

FRGC_bonafide_arcface_file_path = r"./FRGC_bonafide_Arcface_cropped_differenceVectors.libsvm"
FRGC_bonafide_arcface_features, FRGC_bonafide_labels = load_svmlight_file_2(FRGC_bonafide_arcface_file_path)
FRGC_facefusion_arcface_file_path = r"./FRGC_facefusion_Arcface_cropped_differenceVectors.libsvm"
FRGC_facefusion_arcface_features, FRGC_facefusion_labels = load_svmlight_file_2(FRGC_facefusion_arcface_file_path)
FRGC_facemorpher_arcface_file_path = r"./FRGC_facemorpher_Arcface_cropped_differenceVectors.libsvm"
FRGC_facemorpher_arcface_features, FRGC_facemorpher_labels = load_svmlight_file_2(FRGC_facemorpher_arcface_file_path)
FRGC_opencv_arcface_file_path = r"./FRGC_opencv_Arcface_cropped_differenceVectors.libsvm"
FRGC_opencv_arcface_features, FRGC_opencv_labels = load_svmlight_file_2(FRGC_opencv_arcface_file_path)
FRGC_ubo_arcface_file_path = r"./FRGC_ubo_Arcface_cropped_differenceVectors.libsvm"
FRGC_ubo_arcface_features, FRGC_ubo_labels = load_svmlight_file_2(FRGC_ubo_arcface_file_path)


FERET_bonafide_combined_features = np.concatenate((np.array(FERET_bonafide_features).reshape(-1,512), np.array(FERET_bonafide_arcface_features).reshape(-1,512)), axis=1)
FERET_facefusion_combined_features = np.concatenate((np.array(FERET_facefusion_features).reshape(-1,512), np.array(FERET_facefusion_arcface_features).reshape(-1,512)), axis=1)
FERET_facemorpher_combined_features = np.concatenate((np.array(FERET_facemorpher_features).reshape(-1,512), np.array(FERET_facemorpher_arcface_features).reshape(-1,512)), axis=1)
FERET_opencv_combined_features = np.concatenate((np.array(FERET_opencv_features).reshape(-1,512), np.array(FERET_opencv_arcface_features).reshape(-1,512)), axis=1)
FERET_ubo_combined_features = np.concatenate((np.array(FERET_ubo_features).reshape(-1,512), np.array(FERET_ubo_arcface_features).reshape(-1,512)), axis=1)

FRGC_bonafide_combined_features = np.concatenate((np.array(FRGC_bonafide_features).reshape(-1,512), np.array(FRGC_bonafide_arcface_features).reshape(-1,512)), axis=1)
FRGC_facefusion_combined_features = np.concatenate((np.array(FRGC_facefusion_features).reshape(-1,512), np.array(FRGC_facefusion_arcface_features).reshape(-1,512)), axis=1)
FRGC_facemorpher_combined_features = np.concatenate((np.array(FRGC_facemorpher_features).reshape(-1,512), np.array(FRGC_facemorpher_arcface_features).reshape(-1,512)), axis=1)
FRGC_opencv_combined_features = np.concatenate((np.array(FRGC_opencv_features).reshape(-1,512), np.array(FRGC_opencv_arcface_features).reshape(-1,512)), axis=1)
FRGC_ubo_combined_features = np.concatenate((np.array(FRGC_ubo_features).reshape(-1,512), np.array(FRGC_ubo_arcface_features).reshape(-1,512)), axis=1)

FERET_all_features=(FERET_ubo_features +FERET_opencv_features + FERET_facemorpher_features +FERET_facefusion_features)
FERET_all_arcface_features=(FERET_ubo_arcface_features +FERET_opencv_arcface_features + FERET_facemorpher_arcface_features +FERET_facefusion_arcface_features)
FERET_all_combined_features=np.concatenate([FERET_ubo_combined_features,FERET_opencv_combined_features,FERET_facemorpher_combined_features,FERET_facefusion_combined_features])
FERET_all_labels=(FERET_ubo_labels +FERET_opencv_labels + FERET_facemorpher_labels +FERET_facefusion_labels)

FRGC_all_features=(FRGC_ubo_features +FRGC_opencv_features + FRGC_facemorpher_features +FRGC_facefusion_features)
FRGC_all_arcface_features=(FRGC_ubo_arcface_features +FRGC_opencv_arcface_features + FRGC_facemorpher_arcface_features +FRGC_facefusion_arcface_features)
FRGC_all_combined_features=np.concatenate([FRGC_ubo_combined_features,FRGC_opencv_combined_features,FRGC_facemorpher_combined_features,FRGC_facefusion_combined_features])
FRGC_all_labels=(FRGC_ubo_labels +FRGC_opencv_labels + FRGC_facemorpher_labels +FRGC_facefusion_labels)

### dino files
FERET_bonafide_dino_file_path = r"./FERET_bonafide_cropped_differenceVectors_dinov2.libsvm"
FERET_bonafide_dino_features, FERET_bonafide_dino_labels = load_svmlight_file_2(FERET_bonafide_dino_file_path)
FERET_facefusion_dino_file_path = r"./FERET_facefusion_cropped_differenceVectors_dinov2.libsvm"
FERET_facefusion_dino_features, FERET_facefusion_dino_labels = load_svmlight_file_2(FERET_facefusion_dino_file_path)
FERET_facemorpher_dino_file_path = r"./FERET_facemorpher_cropped_differenceVectors_dinov2.libsvm"
FERET_facemorpher_dino_features, FERET_facemorpher_dino_labels = load_svmlight_file_2(FERET_facemorpher_dino_file_path)
FERET_opencv_dino_file_path = r"./FERET_opencv_cropped_differenceVectors_dinov2.libsvm"
FERET_opencv_dino_features, FERET_opencv_dino_labels = load_svmlight_file_2(FERET_opencv_dino_file_path)
FERET_ubo_dino_file_path = r"./FERET_ubo_cropped_differenceVectors_dinov2.libsvm"
FERET_ubo_dino_features, FERET_ubo_dino_labels = load_svmlight_file_2(FERET_ubo_dino_file_path)

FRGC_bonafide_dino_file_path = r"./FRGC_bonafide_cropped_differenceVectors_dinov2.libsvm"
FRGC_bonafide_dino_features, FRGC_bonafide_dino_labels = load_svmlight_file_2(FRGC_bonafide_dino_file_path)
FRGC_facefusion_dino_file_path = r"./FRGC_facefusion_cropped_differenceVectors_dinov2.libsvm"
FRGC_facefusion_dino_features, FRGC_facefusion_dino_labels = load_svmlight_file_2(FRGC_facefusion_dino_file_path)
FRGC_facemorpher_dino_file_path = r"./FRGC_facemorpher_cropped_differenceVectors_dinov2.libsvm"
FRGC_facemorpher_dino_features, FRGC_facemorpher_dino_labels = load_svmlight_file_2(FRGC_facemorpher_dino_file_path)
FRGC_opencv_dino_file_path = r"./FRGC_opencv_cropped_differenceVectors_dinov2.libsvm"
FRGC_opencv_dino_features, FRGC_opencv_dino_labels = load_svmlight_file_2(FRGC_opencv_dino_file_path)
FRGC_ubo_dino_file_path = r"./FRGC_ubo_cropped_differenceVectors_dinov2.libsvm"
FRGC_ubo_dino_features, FRGC_ubo_dino_labels = load_svmlight_file_2(FRGC_ubo_dino_file_path)

FERET_all_dino_features=(FERET_ubo_dino_features +FERET_opencv_dino_features + FERET_facemorpher_dino_features +FERET_facefusion_dino_features)
FERET_all_dino_labels=(FERET_ubo_dino_labels +FERET_opencv_dino_labels + FERET_facemorpher_dino_labels +FERET_facefusion_dino_labels)

FRGC_all_dino_features=(FRGC_ubo_dino_features +FRGC_opencv_dino_features + FRGC_facemorpher_dino_features +FRGC_facefusion_dino_features)
FRGC_all_dino_labels=(FRGC_ubo_dino_labels +FRGC_opencv_dino_labels + FRGC_facemorpher_dino_labels +FRGC_facefusion_dino_labels)

###model training
print("training models")
trained_model_arcface_FERET, test_accuracy = train_svm_model(np.concatenate([FERET_bonafide_arcface_features,FERET_all_arcface_features]),np.concatenate([FERET_bonafide_labels, FERET_all_labels]))
trained_model_dino_FERET, test_accuracy = train_svm_model(np.concatenate([FERET_bonafide_dino_features,FERET_all_dino_features]),np.concatenate([FERET_bonafide_dino_labels, FERET_all_dino_labels]))

trained_model_arcface_FRGC, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_arcface_features,FRGC_all_arcface_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels]))


# debugging for FRGC model, for unknown reasons sets of features were read with wrong dimensions.
FRGC_bonafide_dino_features = np.array(FRGC_bonafide_dino_features)
FRGC_all_dino_features = np.array(FRGC_all_dino_features)

expected_length = 768
invalid_indices = []
filtered_features = []
for i, feature_vector in enumerate(FRGC_all_dino_features):
    if len(feature_vector) != expected_length:
        invalid_indices.append(i)
    else:
        filtered_features.append(feature_vector)
FRGC_all_dino_features = np.array(filtered_features)

FRGC_all_dino_labels = np.delete(FRGC_all_dino_labels, invalid_indices, axis=0)

# Now concatenate both features and labels
combined_features = np.concatenate([FRGC_bonafide_dino_features, FRGC_all_dino_features])
combined_labels = np.concatenate([FRGC_bonafide_dino_labels, FRGC_all_dino_labels])

# Train the SVM
trained_model_dino_FRGC, test_accuracy = train_svm_model(combined_features, combined_labels)


## DET curves
det = DET()
det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Trained on FERET tested on FRGC")
det.x_limits = np.array([1e-4, 0.99])
det.y_limits = np.array([1e-4, 0.99])
det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.create_figure()
det.plot(tar=adjust_scores_for_DET(trained_model_arcface_FERET.predict_proba(FRGC_bonafide_arcface_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_arcface_FERET.predict_proba(FRGC_all_arcface_features)[:, 1], "dissimilarity"), label='Arcface')
det.plot(tar=adjust_scores_for_DET(trained_model_dino_FERET.predict_proba(FRGC_bonafide_dino_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_dino_FERET.predict_proba(FRGC_all_dino_features)[:, 1], "dissimilarity"), label='DINOv2 Base')

det.legend_on(loc="upper right")
det.show()

det = DET()
det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Trained on FRGC tested on FERET")
det.x_limits = np.array([1e-4, 0.99])
det.y_limits = np.array([1e-4, 0.99])
det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.create_figure()
det.plot(tar=adjust_scores_for_DET(trained_model_arcface_FRGC.predict_proba(FERET_bonafide_arcface_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_arcface_FRGC.predict_proba(FERET_all_arcface_features)[:, 1], "dissimilarity"), label='Arcface')
det.plot(tar=adjust_scores_for_DET(trained_model_dino_FRGC.predict_proba(FERET_bonafide_dino_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_dino_FRGC.predict_proba(FERET_all_dino_features)[:, 1], "dissimilarity"), label='DINOv2 Base')

det.legend_on(loc="upper right")
det.show()

