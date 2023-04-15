import numpy as np
import pandas as pd
import sklearn as sk
from scipy.stats import skew
from scipy.stats import kurtosis

def compute_features(sensor_data):
    max_val, min_val = np.max(sensor_data), np.min(sensor_data)
    range_val = max_val - min_val
    mean_val, median_val = np.mean(sensor_data), np.median(sensor_data)
    var_val, skew_val = np.var(sensor_data), skew(sensor_data)
    rms_val = np.sqrt(np.mean(sensor_data ** 2))
    kurt_val = np.mean((sensor_data - mean_val) ** 4) / (var_val ** 2)
    std_val = np.std(sensor_data)
    
    return max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val, kurt_val, std_val

def extract_features(windows):
    # Create an empty array to hold the feature vectors
    features = np.zeros((windows.shape[0], 10, 4))
    # Iterate over each time window and extract the features
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            # Extract the data from the window
            window_data = windows[j, :, i]
            features[j, :, i] = compute_features(window_data)

    return features

def apply_filter_and_extract_features(data_windows, window_size):
    num_windows, num_samples, num_channels = data_windows.shape
    filtered_sample_count = num_samples - window_size + 1
    filtered_data = np.zeros((num_windows, filtered_sample_count, num_channels))

    for i, window in enumerate(data_windows):
        for channel in range(3):
            channel_data = window[:, channel]
            channel_df = pd.DataFrame(channel_data)
            channel_sma = channel_df.rolling(window_size).mean().values.ravel()
            filtered_data[i, :, channel] = channel_sma[window_size - 1:]

        filtered_data[i, :, 3] = window[window_size - 1:, 3]

    features = extract_features(filtered_data)

    feature_dfs = [pd.DataFrame(features[:, :, i]) for i in range(num_channels)]

    for df in feature_dfs:
        for col in range(df.shape[1]):
            column_data = df.iloc[:, col]
            z_scores = (column_data - column_data.mean()) / column_data.std()
            column_data = column_data.mask(abs(z_scores) > 3, other=np.nan)
            df.iloc[:, col] = column_data.fillna(column_data.mean())

    cleaned_features = pd.concat(feature_dfs, axis=0)
    labels = np.concatenate([i * np.ones((df.shape[0], 1)) for i, df in enumerate(feature_dfs, start=1)], axis=0)
    final_data = np.hstack((cleaned_features, labels))
    return final_data