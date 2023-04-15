import h5py
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Segment the data
def segment_data(data, window_size):
    num_segments = data.shape[0] // window_size
    return data[:num_segments * window_size].reshape(num_segments, window_size, data.shape[1])

# Plot the walking and jumping data function
def plot_sample(sample, title):
    plt.figure(figsize=(10, 5))
    plt.plot(sample[:, 0], label='x-axis')
    plt.plot(sample[:, 1], label='y-axis')
    plt.plot(sample[:, 2], label='z-axis')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()
    
# Histrogram function to plot the data
def plot_histograms(data, title, axis_labels=['x-axis', 'y-axis', 'z-axis'], bins=50):
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(axis_labels):
        plt.hist(data[:, i], bins=bins, alpha=0.5, label=label)
    plt.title(title)
    plt.xlabel('Acceleration')
    plt.ylabel('Frequency (hz)')
    plt.legend()
    plt.show()
    
def plot_scatter(sample, title, axis_labels=['x-axis', 'y-axis', 'z-axis']):
    plt.figure(figsize=(10, 5))
    plt.scatter(sample[:, 0], sample[:, 1], label=f'{axis_labels[0]} vs {axis_labels[1]}')
    plt.scatter(sample[:, 0], sample[:, 2], label=f'{axis_labels[0]} vs {axis_labels[2]}')
    plt.scatter(sample[:, 1], sample[:, 2], label=f'{axis_labels[1]} vs {axis_labels[2]}')
    plt.title(title)
    plt.xlabel('Acceleration')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.show()

    
# Load the HDF5 file created earlier to collect the data to be plotted.
with h5py.File('Data.h5', 'r') as f:
    mile_jumping_data = f['Mile']['jumping'][:]
    mile_walking_data = f['Mile']['walking'][:]
    jeev_jumping_data = f['Jeev']['jumping'][:]
    jeev_walking_data = f['Jeev']['walking'][:]
    chris_jumping_data = f['Chris']['jumping'][:]
    chris_walking_data = f['Chris']['walking'][:]

# Segment the data into 5-second windows
window_size = 1000
# 500 * 20 = 10,000 ms = 10 seconds
mile_walking_data = segment_data(mile_walking_data, window_size)
mile_jumping_data = segment_data(mile_jumping_data, window_size)
jeev_walking_data = segment_data(jeev_walking_data, window_size)
jeev_jumping_data = segment_data(jeev_jumping_data, window_size)
chris_walking_data = segment_data(chris_walking_data, window_size)
chris_jumping_data = segment_data(chris_jumping_data, window_size)

# Plot the data once for each piece of data, smaple plot and histogram for each person across all activities.
for i in range(1):
    plot_scatter(mile_walking_data[i], f'Mile - Walking Sample {i+1}')
    plot_scatter(mile_jumping_data[i], f'Mile - Jumping Sample {i+1}')

    plot_scatter(jeev_walking_data[i], f'Jeev - Walking Sample {i+1}')
    plot_scatter(jeev_jumping_data[i], f'Jeev - Jumping Sample {i+1}')

    plot_scatter(chris_walking_data[i], f'Chris - Walking Sample {i+1}')
    plot_scatter(chris_jumping_data[i], f'Chris - Jumping Sample {i+1}')
    
    plot_sample(mile_walking_data[i], f'Mile - Walking Sample {i+1}')
    plot_sample(mile_jumping_data[i], f'Mile - Jumping Sample {i+1}')

    plot_sample(jeev_walking_data[i], f'Jeev - Walking Sample {i+1}')
    plot_sample(jeev_jumping_data[i], f'Jeev - Jumping Sample {i+1}')

    plot_sample(chris_walking_data[i], f'Chris - Walking Sample {i+1}')
    plot_sample(chris_jumping_data[i], f'Chris - Jumping Sample {i+1}')
    
    plot_histograms(mile_walking_data[i], f'Mile - Walking Sample {i+1}')
    plot_histograms(mile_jumping_data[i], f'Mile - Jumping Sample {i+1}')
    
    plot_histograms(jeev_walking_data[i], f'Jeev - Walking Sample {i+1}')
    plot_histograms(jeev_jumping_data[i], f'Jeev - Jumping Sample {i+1}')
    
    plot_histograms(chris_walking_data[i], f'Chris - Walking Sample {i+1}')
    plot_histograms(chris_jumping_data[i], f'Chris - Jumping Sample {i+1}')
    
f.close()

# Data dictionary to store the meta data for each person
data = {
    "Chris": {"jumping": {"device": "Chris-Jumping-Device.csv", "time": "Chris-Jumping-Time.csv"},
              "walking": {"device": "Chris-Walking-Device.csv", "time": "Chris-Walking-Time.csv"}},
    "Jeev": {"jumping": {"device": "Jeev-Jumping-Device.csv", "time": "Jeev-Jumping-Time.csv"},
             "walking": {"device": "Jeev-Walking-Device.csv", "time": "Jeev-Walking-Time.csv"}},
    "Mile": {"jumping": {"device": "Mile-Jumping-Device.csv", "time": "Mile-Jumping-Time.csv"},
             "walking": {"device": "Mile-Walking-Device.csv", "time": "Mile-Walking-Time.csv"}}
}

# Initialize an empty dictionary to store the duration data
durations = {}

# Loop through the data dictionary
for name, activities in data.items():
    durations[name] = {}
    for activity, files in activities.items():
        time_data = pd.read_csv(files["time"])

        if len(time_data.loc[time_data['event'] == 'START']) > 0 and len(time_data.loc[time_data['event'] == 'PAUSE']) > 0:
            # Extract the middle 12 characters of the "system time text"
            start_time_text = time_data.loc[time_data['event'] == 'START', 'system time text'].values[0][11:23]
            pause_time_text = time_data.loc[time_data['event'] == 'PAUSE', 'system time text'].values[0][11:23]

            # Convert the time strings to datetime objects, finds the specific group we want
            start_time = datetime.strptime(start_time_text, "%H:%M:%S.%f")
            pause_time = datetime.strptime(pause_time_text, "%H:%M:%S.%f")

            # Calculate the duration between start and pause events
            duration = (pause_time - start_time).total_seconds()

            durations[name][activity] = duration

# Create a DataFrame from the duration data
duration_df = pd.DataFrame(durations)

# Create a bar plot from the DataFrame
ax = duration_df.plot(kind='bar', figsize=(10, 6))

# Set plot attributes
ax.set_title("Walking and Jumping Durations")
ax.set_xlabel("Activity")
ax.set_ylabel("Duration (seconds)")
ax.legend(title="Person")

plt.show()