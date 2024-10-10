from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
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
import pickle

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
	
def train_svm_model_optimized(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Convert labels to integer type
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Calculate class weights to compensate for imbalance
    class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000,10000],
        'gamma': [1],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    }
    
    # Initialize the SVM model
    svm_model = SVC(probability=True)
    
    # Initialize GridSearchCV with SVM and parameter grid
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best estimator
    best_svm = grid_search.best_estimator_
    
    # Evaluate the best model on the test set
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print the best parameters and the test set accuracy
    print("Best parameters found: ", grid_search.best_params_)
    print("Test set accuracy: ", accuracy)
    
    # Return the trained model
    return best_svm
	
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

		
		
## load data
#number of difference vectors from MSYNM
n=3000

## load MSYNM data
print("loading MSYNM data")
MSYNM_bonafide_file_path = r"./MSYNM_bonafide_beauty_differenceVectors.libsvm"
MSYNM_bonafide_features, MSYNM_bonafide_labels = load_svmlight_file_512(MSYNM_bonafide_file_path)
MSYNM_bonafide_features, MSYNM_bonafide_labels = MSYNM_bonafide_features[:n], MSYNM_bonafide_labels[:n]

MSYNM_ubo_file_path = r"./MSYNM_UBO_beauty_differenceVectors.libsvm"
MSYNM_ubo_features, MSYNM_ubo_labels = load_svmlight_file_512(MSYNM_ubo_file_path)
MSYNM_ubo_features, MSYNM_ubo_labels = MSYNM_ubo_features[:n], MSYNM_ubo_labels[:n]

MSYNM_mipgan_file_path = r"./MSYNM_MIPGAN_beauty_differenceVectors.libsvm"
MSYNM_mipgan_features, MSYNM_mipgan_labels = load_svmlight_file_512(MSYNM_mipgan_file_path)
MSYNM_mipgan_features, MSYNM_mipgan_labels = MSYNM_mipgan_features[:n], MSYNM_mipgan_labels[:n]

MSYNM_bonafide_arcface_file_path = r"./MSYNM_bonafide_arcface_differenceVectors.libsvm"
MSYNM_bonafide_arcface_features, MSYNM_bonafide_arcface_labels = load_svmlight_file_512(MSYNM_bonafide_arcface_file_path)
MSYNM_bonafide_arcface_features, MSYNM_bonafide_arcface_labels = MSYNM_bonafide_arcface_features[:n], MSYNM_bonafide_arcface_labels[:n]

MSYNM_ubo_arcface_file_path = r"./MSYNM_UBO_arcface_differenceVectors.libsvm"
MSYNM_ubo_arcface_features, MSYNM_ubo_arcface_labels = load_svmlight_file_512(MSYNM_ubo_arcface_file_path)
MSYNM_ubo_arcface_features, MSYNM_ubo_arcface_labels =MSYNM_ubo_arcface_features[:n], MSYNM_ubo_arcface_labels[:n]

MSYNM_mipgan_arcface_file_path = r"./MSYNM_MIPGAN_arcface_differenceVectors.libsvm"
MSYNM_mipgan_arcface_features, MSYNM_mipgan_arcface_labels = load_svmlight_file_512(MSYNM_mipgan_arcface_file_path)
MSYNM_mipgan_arcface_features, MSYNM_mipgan_arcface_labels=MSYNM_mipgan_arcface_features[:n], MSYNM_mipgan_arcface_labels[:n]

MSYNM_bonafide_combined_file_path = r"./MSYNM_bonafide_combined_differenceVectors.libsvm"
MSYNM_bonafide_combined_features, MSYNM_bonafide_combined_labels = load_svmlight_file_1024(MSYNM_bonafide_combined_file_path)
MSYNM_bonafide_combined_features, MSYNM_bonafide_combined_labels = MSYNM_bonafide_combined_features[:n], MSYNM_bonafide_combined_labels[:n]

MSYNM_ubo_combined_file_path = r"./MSYNM_UBO_combined_differenceVectors.libsvm"
MSYNM_ubo_combined_features, MSYNM_ubo_combined_labels = load_svmlight_file_1024(MSYNM_ubo_combined_file_path)
MSYNM_ubo_combined_features, MSYNM_ubo_combined_labels = MSYNM_ubo_combined_features[:n], MSYNM_ubo_combined_labels[:n]

MSYNM_mipgan_combined_file_path = r"./MSYNM_MIPGAN_combined_differenceVectors.libsvm"
MSYNM_mipgan_combined_features, MSYNM_mipgan_combined_labels = load_svmlight_file_1024(MSYNM_mipgan_combined_file_path)
MSYNM_mipgan_combined_features, MSYNM_mipgan_combined_labels = MSYNM_mipgan_combined_features[:n], MSYNM_mipgan_combined_labels[:n]

MSYNM_all_features=np.concatenate([MSYNM_ubo_features,MSYNM_mipgan_features])
MSYNM_all_arcface_features=np.concatenate([MSYNM_ubo_arcface_features,MSYNM_mipgan_arcface_features])
MSYNM_all_combined_features=np.concatenate([MSYNM_ubo_combined_features,MSYNM_mipgan_combined_features])
MSYNM_all_labels=np.concatenate([MSYNM_ubo_labels,MSYNM_mipgan_labels])

print("loading other data")
#FRLL
FRLL_bonafide_file_path = r"./FRLL_bonafide_differenceVectors.libsvm"
FRLL_bonafide_features, FRLL_bonafide_labels = load_svmlight_file_2(FRLL_bonafide_file_path)
FRLL_amsl_file_path = r"./FRLL_amsl_differenceVectors.libsvm"
FRLL_amsl_features, FRLL_amsl_labels = load_svmlight_file_2(FRLL_amsl_file_path)
FRLL_facemorpher_file_path = r"./FRLL_facemorpher_differenceVectors.libsvm"
FRLL_facemorpher_features, FRLL_facemorpher_labels = load_svmlight_file_2(FRLL_facemorpher_file_path)
FRLL_opencv_file_path = r"./FRLL_opencv_differenceVectors.libsvm"
FRLL_opencv_features, FRLL_opencv_labels = load_svmlight_file_2(FRLL_opencv_file_path)
FRLL_stylegan_file_path = r"./FRLL_stylegan_differenceVectors.libsvm"
FRLL_stylegan_features, FRLL_stylegan_labels = load_svmlight_file_2(FRLL_stylegan_file_path)
FRLL_webmorph_file_path = r"./FRLL_webmorph_differenceVectors.libsvm"
FRLL_webmorph_features, FRLL_webmorph_labels = load_svmlight_file_2(FRLL_webmorph_file_path)

FRLL_bonafide_arcface_file_path = r"./FRLL_bonafide_arcface_differenceVectors.libsvm"
FRLL_bonafide_arcface_features, FRLL_bonafide_arcface_labels = load_svmlight_file_2(FRLL_bonafide_arcface_file_path)
FRLL_amsl_arcface_file_path = r"./FRLL_amsl_arcface_differenceVectors.libsvm"
FRLL_amsl_arcface_features, FRLL_amsl_arcface_labels = load_svmlight_file_2(FRLL_amsl_arcface_file_path)
FRLL_facemorpher_arcface_file_path = r"./FRLL_facemorpher_arcface_differenceVectors.libsvm"
FRLL_facemorpher_arcface_features, FRLL_facemorpher_arcface_labels = load_svmlight_file_2(FRLL_facemorpher_arcface_file_path)
FRLL_opencv_arcface_file_path = r"./FRLL_opencv_arcface_differenceVectors.libsvm"
FRLL_opencv_arcface_features, FRLL_opencv_arcface_labels = load_svmlight_file_2(FRLL_opencv_arcface_file_path)
FRLL_stylegan_arcface_file_path = r"./FRLL_stylegan_arcface_differenceVectors.libsvm"
FRLL_stylegan_arcface_features, FRLL_stylegan_arcface_labels = load_svmlight_file_2(FRLL_stylegan_arcface_file_path)
FRLL_webmorph_arcface_file_path = r"./FRLL_webmorph_arcface_differenceVectors.libsvm"
FRLL_webmorph_arcface_features, FRLL_webmorph_arcface_labels = load_svmlight_file_2(FRLL_webmorph_arcface_file_path)

FRLL_bonafide_combined_features = np.concatenate((np.array(FRLL_bonafide_features).reshape(-1,512), np.array(FRLL_bonafide_arcface_features).reshape(-1,512)), axis=1)
FRLL_amsl_combined_features = np.concatenate((np.array(FRLL_amsl_features).reshape(-1,512), np.array(FRLL_amsl_arcface_features).reshape(-1,512)), axis=1)
FRLL_facemorpher_combined_features = np.concatenate((np.array(FRLL_facemorpher_features).reshape(-1,512), np.array(FRLL_facemorpher_arcface_features).reshape(-1,512)), axis=1)
FRLL_opencv_combined_features = np.concatenate((np.array(FRLL_opencv_features).reshape(-1,512), np.array(FRLL_opencv_arcface_features).reshape(-1,512)), axis=1)
FRLL_stylegan_combined_features = np.concatenate((np.array(FRLL_stylegan_features).reshape(-1,512), np.array(FRLL_stylegan_arcface_features).reshape(-1,512)), axis=1)
FRLL_webmorph_combined_features = np.concatenate((np.array(FRLL_webmorph_features).reshape(-1,512), np.array(FRLL_webmorph_arcface_features).reshape(-1,512)), axis=1)

FRLL_all_features=(FRLL_amsl_features +FRLL_facemorpher_features + FRLL_opencv_features +FRLL_stylegan_features + FRLL_webmorph_features)
FRLL_all_arcface_features=(FRLL_amsl_arcface_features +FRLL_facemorpher_arcface_features + FRLL_opencv_arcface_features +FRLL_stylegan_arcface_features +FRLL_webmorph_arcface_features)
FRLL_all_combined_features=np.concatenate([FRLL_amsl_combined_features,FRLL_facemorpher_combined_features,FRLL_opencv_combined_features,FRLL_stylegan_combined_features,FRLL_webmorph_combined_features])
FRLL_all_labels=(FRLL_amsl_labels +FRLL_facemorpher_labels + FRLL_opencv_labels +FRLL_stylegan_labels + FRLL_webmorph_labels)



#FERET and FRGC
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

FERET_all_features=(FRGC_ubo_features +FRGC_opencv_features + FRGC_facemorpher_features +FRGC_facefusion_features)
FERET_all_arcface_features=(FRGC_ubo_arcface_features +FRGC_opencv_arcface_features + FRGC_facemorpher_arcface_features +FRGC_facefusion_arcface_features)
FERET_all_combined_features=np.concatenate([FRGC_ubo_combined_features,FRGC_opencv_combined_features,FRGC_facemorpher_combined_features,FRGC_facefusion_combined_features])
FERET_all_labels=(FRGC_ubo_labels +FRGC_opencv_labels + FRGC_facemorpher_labels +FRGC_facefusion_labels)

FRGC_all_features=(FRGC_ubo_features +FRGC_opencv_features + FRGC_facemorpher_features +FRGC_facefusion_features)
FRGC_all_arcface_features=(FRGC_ubo_arcface_features +FRGC_opencv_arcface_features + FRGC_facemorpher_arcface_features +FRGC_facefusion_arcface_features)
FRGC_all_combined_features=np.concatenate([FRGC_ubo_combined_features,FRGC_opencv_combined_features,FRGC_facemorpher_combined_features,FRGC_facefusion_combined_features])
FRGC_all_labels=(FRGC_ubo_labels +FRGC_opencv_labels + FRGC_facemorpher_labels +FRGC_facefusion_labels)



## model training
print("training models")
trained_model_combined_FERET_FRGC_MSYNM, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_combined_features,FRGC_all_combined_features,FERET_bonafide_combined_features,FERET_all_combined_features,MSYNM_bonafide_combined_features,MSYNM_all_combined_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels,MSYNM_bonafide_combined_labels,MSYNM_all_labels]))
#svmutil.svm_save_model("SVM_FERET_FRGC_MSYNM.model", trained_model_combined_FERET_FRGC_MSYNM)
with open("SVM_FERET_FRGC_MSYNM.pkl", 'wb') as file:
    pickle.dump(trained_model_combined_FERET_FRGC_MSYNM,file)
print("saved combined model")

trained_model_arcface_FERET_FRGC, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_arcface_features,FRGC_all_arcface_features,FERET_bonafide_arcface_features,FERET_all_arcface_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels]))
trained_model_FERET_FRGC, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_features,FRGC_all_features,FERET_bonafide_features,FERET_all_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels]))
trained_model_combined_FERET_FRGC, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_combined_features,FRGC_all_combined_features,FERET_bonafide_combined_features,FERET_all_combined_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels]))



print("training optimized model")
trained_model_combined_FERET_FRGC_optimized= train_svm_model_optimized(np.concatenate([FRGC_bonafide_combined_features,FRGC_all_combined_features,FERET_bonafide_combined_features,FERET_all_combined_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels]))

#temp output
print("testing")
bpcer10,bpcer1,eer=test_svm(trained_model_FERET_FRGC,MSYNM_bonafide_features,MSYNM_all_features)
print("MSYNM beauty")
print("EER: " +str(eer))
print("BPCER 10: "+str(bpcer10))
print("BPCER 1"+str(bpcer1))

bpcer10,bpcer1,eer=test_svm(trained_model_arcface_FERET_FRGC,MSYNM_bonafide_arcface_features,MSYNM_all_arcface_features)
print("MSYNM Arcface")
print("EER: " +str(eer))
print("BPCER 10: "+str(bpcer10))
print("BPCER 1"+str(bpcer1))

bpcer10,bpcer1,eer=test_svm(trained_model_combined_FERET_FRGC,MSYNM_bonafide_combined_features,MSYNM_all_combined_features)
print("MSYNM combined")
print("EER: " +str(eer))
print("BPCER 10: "+str(bpcer10))
print("BPCER 1"+str(bpcer1))
##


trained_model_FRLL_all, test_accuracy = train_svm_model(np.concatenate([FRLL_bonafide_features,FRLL_all_features]),np.concatenate([FRLL_bonafide_labels, FRLL_all_labels]))
trained_model_arcface_FRLL_all, test_accuracy = train_svm_model(np.concatenate([FRLL_bonafide_arcface_features,FRLL_all_arcface_features]),np.concatenate([FRLL_bonafide_labels, FRLL_all_labels]))
trained_model_combined_FRLL_all, test_accuracy = train_svm_model(np.concatenate([FRLL_bonafide_combined_features,FRLL_all_combined_features]),np.concatenate([FRLL_bonafide_labels, FRLL_all_labels]))

trained_model_combined_FERET_all, test_accuracy = train_svm_model(np.concatenate([FERET_bonafide_combined_features,FERET_all_combined_features]),np.concatenate([FERET_bonafide_labels, FERET_all_labels]))
trained_model_combined_FRGC_all, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_combined_features,FRGC_all_combined_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels]))


trained_model_arcface_FERET_FRGC_FRLL, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_arcface_features,FRGC_all_arcface_features,FERET_bonafide_arcface_features,FERET_all_arcface_features,FRLL_bonafide_arcface_features,FRLL_all_arcface_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels, FRLL_bonafide_labels,FRLL_all_labels]))
trained_model_FERET_FRGC_FRLL, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_features,FRGC_all_features,FERET_bonafide_features,FERET_all_features,FRLL_bonafide_features,FRLL_all_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels, FRLL_bonafide_labels,FRLL_all_labels]))
trained_model_combined_FERET_FRGC_FRLL, test_accuracy = train_svm_model(np.concatenate([FRGC_bonafide_combined_features,FRGC_all_combined_features,FERET_bonafide_combined_features,FERET_all_combined_features,FRLL_bonafide_combined_features,FRLL_all_combined_features]),np.concatenate([FRGC_bonafide_labels, FRGC_all_labels,FERET_bonafide_labels, FERET_all_labels, FRLL_bonafide_labels,FRLL_all_labels]))







print("testing")
#testing optimized vs unoptimized
det = DET()
det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Performance of FERET and FRGC on MSYNM comparison between optimized and unoptimized SVM configuration, for combined")
det.x_limits = np.array([1e-4, 0.99])
det.y_limits = np.array([1e-4, 0.99])
det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.create_figure()
det.plot(tar=adjust_scores_for_DET(trained_model_combined_FERET_FRGC.predict_proba(MSYNM_bonafide_combined_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_combined_FERET_FRGC.predict_proba(MSYNM_all_combined_features)[:, 1], "dissimilarity"), label='Unoptimized')
det.plot(tar=adjust_scores_for_DET(trained_model_combined_FERET_FRGC_optimized.predict_proba(MSYNM_bonafide_combined_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_combined_FERET_FRGC.predict_proba(MSYNM_all_combined_features)[:, 1], "dissimilarity"), label='Optimized')
det.legend_on(loc="upper right")
det.show()






#testing on MSYNM
det = DET()
det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Performance of FERET and FRGC on MSYNM comparison between Beauty, Arcface and both together")
det.x_limits = np.array([1e-4, 0.99])
det.y_limits = np.array([1e-4, 0.99])
det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.create_figure()
det.plot(tar=adjust_scores_for_DET(trained_model_FERET_FRGC.predict_proba(MSYNM_bonafide_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_FERET_FRGC.predict_proba(MSYNM_all_features)[:, 1], "dissimilarity"), label='Beauty')
det.plot(tar=adjust_scores_for_DET(trained_model_arcface_FERET_FRGC.predict_proba(MSYNM_bonafide_arcface_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_arcface_FERET_FRGC.predict_proba(MSYNM_all_arcface_features)[:, 1], "dissimilarity"), label='Arcface')
det.plot(tar=adjust_scores_for_DET(trained_model_combined_FERET_FRGC.predict_proba(MSYNM_bonafide_combined_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_combined_FERET_FRGC.predict_proba(MSYNM_all_combined_features)[:, 1], "dissimilarity"), label='Combined')
det.legend_on(loc="upper right")
det.show()


det = DET()
det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Performance of FERET, FRGC and FRLL on MSYNM comparison between Beauty, Arcface and both together")
det.x_limits = np.array([1e-4, 0.99])
det.y_limits = np.array([1e-4, 0.99])
det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
det.create_figure()
det.plot(tar=adjust_scores_for_DET(trained_model_FERET_FRGC_FRLL.predict_proba(MSYNM_bonafide_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_FERET_FRGC_FRLL.predict_proba(MSYNM_all_features)[:, 1], "dissimilarity"), label='Beauty')
det.plot(tar=adjust_scores_for_DET(trained_model_arcface_FERET_FRGC_FRLL.predict_proba(MSYNM_bonafide_arcface_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_arcface_FERET_FRGC_FRLL.predict_proba(MSYNM_all_arcface_features)[:, 1], "dissimilarity"), label='Arcface')
det.plot(tar=adjust_scores_for_DET(trained_model_combined_FERET_FRGC_FRLL.predict_proba(MSYNM_bonafide_combined_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_combined_FERET_FRGC_FRLL.predict_proba(MSYNM_all_combined_features)[:, 1], "dissimilarity"), label='Combined')
det.legend_on(loc="upper right")
det.show()


#other DET curves if needed

#det = DET()
#det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Performance of FERET on MSYNM comparison between Beauty, Arcface and both together")
#det.x_limits = np.array([1e-4, 0.99])
#det.y_limits = np.array([1e-4, 0.99])
#det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
#det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
#det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
#det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
#det.create_figure()
#det.plot(tar=adjust_scores_for_DET(trained_model_FERET.predict_proba(MSYNM_bonafide_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_FERET.predict_proba(MSYNM_all_features)[:, 1], "dissimilarity"), label='Beauty')
#det.plot(tar=adjust_scores_for_DET(trained_model_arcface_FERET.predict_proba(MSYNM_bonafide_arcface_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_arcface_FERET.predict_proba(MSYNM_all_arcface_features)[:, 1], "dissimilarity"), label='Arcface')
#det.plot(tar=adjust_scores_for_DET(trained_model_combined_FERET.predict_proba(MSYNM_bonafide_combined_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_combined_FERET.predict_proba(MSYNM_all_combined_features)[:, 1], "dissimilarity"), label='Combined')
#det.legend_on(loc="upper right")
#det.show()


#det = DET()
#det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Performance of FRGC on MSYNM comparison between Beauty, Arcface and both together")
#det.x_limits = np.array([1e-4, 0.99])
#det.y_limits = np.array([1e-4, 0.99])
#det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
#det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
#det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
#det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
#det.create_figure()
#det.plot(tar=adjust_scores_for_DET(trained_model_FRGC.predict_proba(MSYNM_bonafide_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_FRGC.predict_proba(MSYNM_all_features)[:, 1], "dissimilarity"), label='Beauty')
#det.plot(tar=adjust_scores_for_DET(trained_model_arcface_FRGC.predict_proba(MSYNM_bonafide_arcface_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_arcface_FRGC.predict_proba(MSYNM_all_arcface_features)[:, 1], "dissimilarity"), label='Arcface')
#det.plot(tar=adjust_scores_for_DET(trained_model_combined_FRGC.predict_proba(MSYNM_bonafide_combined_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_combined_FRGC.predict_proba(MSYNM_all_combined_features)[:, 1], "dissimilarity"), label='Combined')
#det.legend_on(loc="upper right")
#det.show()

#det = DET()
#det = DET(biometric_evaluation_type='PAD', abbreviate_axes=True, plot_eer_line=True, plot_title="Performance of FRLL on MSYNM comparison between Beauty, Arcface and both together")
#det.x_limits = np.array([1e-4, 0.99])
#det.y_limits = np.array([1e-4, 0.99])
#det.x_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
#det.x_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
#det.y_ticks = np.array([1e-3, 1e-2, 10e-2, 20e-2, 40e-2,80e-2,99e-2])
#det.y_ticklabels = np.array(['0.1', '1', '10', '20', '40','80','99'])
#det.create_figure()
#det.plot(tar=adjust_scores_for_DET(trained_model_FRLL.predict_proba(MSYNM_bonafide_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_FRLL.predict_proba(MSYNM_all_features)[:, 1], "dissimilarity"), label='Beauty')
#det.plot(tar=adjust_scores_for_DET(trained_model_arcface_FRLL.predict_proba(MSYNM_bonafide_arcface_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_arcface_FRLL.predict_proba(MSYNM_all_arcface_features)[:, 1], "dissimilarity"), label='Arcface')
#det.plot(tar=adjust_scores_for_DET(trained_model_combined_FRLL.predict_proba(MSYNM_bonafide_combined_features)[:, 1], "dissimilarity"), non=adjust_scores_for_DET(trained_model_combined_FRLL.predict_proba(MSYNM_all_combined_features)[:, 1], "dissimilarity"), label='Combined')
#det.legend_on(loc="upper right")
#det.show()