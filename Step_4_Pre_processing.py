from Functions import extract_features
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Function to apply a moving average filter and extract features
def apply_filter_and_extract_features(data_windows, window_size):
    num_windows, num_samples, num_channels = data_windows.shape
    filtered_sample_count = num_samples - window_size + 1
    filtered_clean_data = np.zeros((num_windows, filtered_sample_count, num_channels))
    # Apply the moving average filter through a rolling window
    for i, window in enumerate(data_windows):
        for channel in range(num_channels):
            channel_data = window[:, channel] if channel < 2 else window[:, -1]
            channel_df = pd.DataFrame(channel_data)
            channel_sma = channel_df.rolling(window_size).mean().values.ravel()
            filtered_clean_data[i, :, channel] = channel_sma[window_size - 1:]

        filtered_clean_data[i, :, 3] = window[window_size - 1:, 3]
    # Extract features from the filtered data using function from Functions.py that was created in Step 5. 
    # To avoid circular imports, the function was copied into a new file called Functions.py
    features = extract_features(filtered_clean_data)
    # Create a list of dataframes, one for each channel
    feature_dfs = [pd.DataFrame(features[:, :, i]) for i in range(num_channels)]
    # Z-score normalization to remove outliers
    for df in feature_dfs:
        for col in range(df.shape[1]):
            column_data = df.iloc[:, col]
            z_score = (column_data - column_data.mean()) / column_data.std()
            column_data = column_data.mask(abs(z_score) > 3, other=np.nan)
            df.iloc[:, col] = column_data.fillna(column_data.mean())
    # Concatenate the features into one dataframe
    cleaned_features = pd.concat(feature_dfs, axis=0)
    labels = np.concatenate([i * np.ones((df.shape[0], 1)) for i, df in enumerate(feature_dfs, start=1)], axis=0)
    final_data = np.hstack((cleaned_features, labels))
    
    return final_data

# Load the hdf5 file creaited in Step 2 to plot the new processed data
with h5py.File('Data.h5', 'r') as f:
    train_walking_windows = f['dataset/train/walking'][:, :, 1:]
    test_walking_windows = f['dataset/test/walking'][:, :, 1:]
    train_jumping_windows = f['dataset/train/jumping'][:, :, 1:]
    test_jumping_windows = f['dataset/test/jumping'][:, :, 1:]

# Process data using a moving average filter and extract features
window_size = 5
walking_filtered = apply_filter_and_extract_features(train_walking_windows, window_size)
jumping_filtered = apply_filter_and_extract_features(train_jumping_windows, window_size)

# Combine walking and jumping data into one dataset for plotting purposes
final_data = np.concatenate((walking_filtered, jumping_filtered), axis=0)

# Separate the features and labels for plotting
preprocessed_features = final_data[:, :-1]

# Plot the pre-processed data, setting the number of features to plot based on the number of features in the data
num_features = preprocessed_features.shape[1]

# Plot the pre-processed data, setting the number of features to plot based on the number of features in the data
fig1, axes1 = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
fig2, axes2 = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

axes = np.concatenate((axes1, axes2))

for i, ax in enumerate(axes):
    ax.plot(preprocessed_features[:, i])
    ax.set_ylabel(f"Feature-Number {i + 1}")

plt.xlabel("Sample Index")
fig1.suptitle("Pre-processed 1-5")
fig2.suptitle("Pre-processed 6-10")
plt.show()