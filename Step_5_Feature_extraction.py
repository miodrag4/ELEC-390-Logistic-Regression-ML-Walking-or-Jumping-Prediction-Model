import h5py
import pandas as pd
import numpy as np
from scipy.stats import skew
from Functions import apply_filter_and_extract_features

# Function to compute the features of the test and train data
def compute_features(sensor_data):
    max_val, min_val = np.max(sensor_data), np.min(sensor_data)
    range_val = max_val - min_val
    mean_val, median_val = np.mean(sensor_data), np.median(sensor_data)
    var_val, skew_val = np.var(sensor_data), skew(sensor_data)
    rms_val = np.sqrt(np.mean(sensor_data ** 2))
    kurt_val = np.mean((sensor_data - mean_val) ** 4) / (var_val ** 2)
    std_val = np.std(sensor_data)
    
    return max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val, kurt_val, std_val

# Function to extract the features from the train data
def extract_train_features(windows):
    # Create an empty array to hold the feature vectors and labels
    features = np.zeros((windows.shape[0], 10, 4))
    # Loop over each window and extract the features
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            # Extract the data from the window
            window_data = windows[j, :, i]
            # Compute the features
            features[j, :, i] = compute_features(window_data)

    return features
# Function to extract the features from the test data
def extract_test_features(windows):
    # Create an empty array to hold the feature vectors and labels
    features = np.zeros((windows.shape[0], 10, 4))
    # Loop over each window and extract the features
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            window_data = windows[j, :, i]
            # Compute the features
            features[j, :, i] = compute_features(window_data)
    # Extract the feature arrays
    x_features = features[:, :, 0]
    y_features = features[:, :, 1]
    z_features = features[:, :, 2]
    total_features = features[:, :, 3]

    # Concatenate the feature arrays
    all_features = np.concatenate((x_features, y_features, z_features, total_features), axis=0)

    # Create a column of all ones to hold the labels
    labels = np.concatenate((np.ones((x_features.shape[0], 1)),2 * np.ones((y_features.shape[0], 1)),
                             3 * np.ones((z_features.shape[0], 1)), 4 * np.ones((total_features.shape[0], 1))), axis=0)
    # Adding the labels to the feature array
    all_features = np.hstack((all_features, labels))

    return all_features

# Reading the data from the h5 file
with h5py.File('Data.h5', 'r') as file:
    train_walk_win = file['dataset/train/walking'][:, :, 1:]
    test_walk_win = file['dataset/test/walking'][:, :, 1:]
    train_jump_win = file['dataset/train/jumping'][:, :, 1:]
    test_jump_win = file['dataset/test/jumping'][:, :, 1:]

# Setting the window size
window_size = 5
# Extracting the features from the train and test data
train_walk_filt, train_jump_filt  = apply_filter_and_extract_features(train_walk_win, window_size), apply_filter_and_extract_features(train_jump_win, window_size)
# Concatenate the feature arrays
train_features = np.concatenate((train_walk_filt, train_jump_filt), axis=0)
# Create a column of all ones to hold the labels
train_labels = np.concatenate((np.zeros((train_walk_filt.shape[0], 1)), np.ones((train_jump_filt.shape[0], 1))), axis=0)
# Extracting the features from the test data, walking and jumping
test_walk_feats, test_jump_feats  = extract_test_features(test_walk_win), extract_test_features(test_jump_win)
test_features = np.concatenate((test_walk_feats, test_jump_feats), axis=0)
test_labels = np.concatenate((np.zeros((test_walk_feats.shape[0], 1)), np.ones((test_jump_feats.shape[0], 1))), axis=0)
# Add labels to the train and test feature arrays
cols = np.array(
['max_val', 'min_val', 'range_val', 'mean_val', 'median_val', 'var_val', 'skew_val', 'rms_val', 'kurt_val',
'std_val', 'measurement', 'activity'])
# Create a dataframe to hold the train and test data
train_data = pd.DataFrame(np.hstack((train_features, train_labels)), columns=cols)
test_data = pd.DataFrame(np.hstack((test_features, test_labels)), columns=cols)