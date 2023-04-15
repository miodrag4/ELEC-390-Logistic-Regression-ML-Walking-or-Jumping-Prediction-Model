import pandas as pd
# Load the HDF5 file into a dictionary. The dictionary keys are the names of the people
data_files = {
    'mile_jumping': 'Jumping-Mile.csv',
    'mile_walking': 'Walking-Mile.csv',
    'jeev_jumping': 'Jumping-Jeev.csv',
    'jeev_walking': 'Walking-Jeev.csv',
    'chris_jumping': 'Jumping-Chris.csv',
    'chris_walking': 'Walking-Chris.csv'
}

# Assign variables to each person's data. The data is a 2D array with 3 columns (x, y, z)
mile_jumping_data_file = 'Jumping-Mile.csv'
mile_walking_data_file = 'Walking-Mile.csv'
jeev_jumping_data_file = 'Jumping-Jeev.csv'
jeev_walking_data_file = 'Walking-Jeev.csv'
chris_jumping_data_file = 'Jumping-Chris.csv'
chris_walking_data_file = 'Walking-Chris.csv'

# Read in the CSV files for each person, and assign the data to a variable
mile_jumping_data = pd.read_csv(mile_jumping_data_file)
mile_walking_data = pd.read_csv(mile_walking_data_file)
jeev_jumping_data = pd.read_csv(jeev_jumping_data_file)
jeev_walking_data = pd.read_csv(jeev_walking_data_file)
chris_jumping_data = pd.read_csv(chris_jumping_data_file)
chris_walking_data = pd.read_csv(chris_walking_data_file)

# Assign varible to each person' meta data.
Chris_jumping_Device = 'Chris_jumping_Device'
Chris_jumping_Time = 'Chris_jumping_Time'
Chris_walking_Device = 'Chris_walking_Device'
Chris_walking_Time = 'Chris_walking_Time'
jeev_jumping_Device = 'Jeev_jumping_Device'
jeev_jumping_Time = 'jeev_jumping_Time'
jeev_walking_Device = 'jeev_walking_Device'
jeev_walking_Time = 'jeev_walking_Time'
mile_jumping_Device = 'mile_jumping_Device'
mile_jumping_Time = 'mile_jumping_Time'
mile_walking_Device = 'mile_walking_Device'
mile_walking_Time = 'mile_walking_Time'

data = {
    "Chris": {"jumping": {"device": "Chris-Jumping-Device.csv", "time": "Chris-Jumping-Time.csv"},
              "walking": {"device": "Chris-Walking-Device.csv", "time": "Chris-Walking-Time.csv"}},
    "Jeev": {"jumping": {"device": "Jeev-Jumping-Device.csv", "time": "Jeev-Jumping-Time.csv"},
             "walking": {"device": "Jeev-Walking-Device.csv", "time": "Jeev-Walking-Time.csv"}},
    "Mile": {"jumping": {"device": "Mile-Jumping-Device.csv", "time": "Mile-Jumping-Time.csv"},
             "walking": {"device": "Mile-Walking-Device.csv", "time": "Mile-Walking-Time.csv"}}
}


device_frames = []
time_frames = []

for name, activities in data.items():
    for activity, files in activities.items():
        device_data = pd.read_csv(files["device"])
        time_data = pd.read_csv(files["time"])

        device_data["name"] = name
        device_data["activity"] = activity
        time_data["name"] = name
        time_data["activity"] = activity

        device_frames.append(device_data)
        time_frames.append(time_data)

# Concatenate all device and time data
all_device_data = pd.concat(device_frames)
all_time_data = pd.concat(time_frames)

# Merge device and time data
merged_data = pd.merge(all_device_data, all_time_data, on=["name", "activity"])

# Sort by name and file
sorted_data = merged_data.sort_values(["name", "activity"])

# Write sorted data to a single output CSV file
sorted_data.to_csv("merged_output.csv", index=False)