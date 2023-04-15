# Libraries
import h5py
import numpy as np
import pandas as pd

# Read Function to read in the CSV files from Step_1_Collecting_Data.py, each file is a 2D array with 3 columns (x, y, z). 
# Each team member will have 2 files, one for walking and one for jumping.
def read_csv_data(filename):
    return pd.read_csv(filename).values

# Segment function to create segments of data from the 2D array.
def create_segments_labels(member_name, activity, data, window_size, step_size):
    num_segments = (len(data) - window_size) // step_size + 1
    segments = [data[i * step_size:(i * step_size + window_size)] for i in range(num_segments)]
    labels = [f'{member_name}_{activity}' for _ in range(num_segments)]
    return list(zip(segments, labels))

# Function to create a dataset for walking and jumping for each person
def create_activity_datasets(group, segments, activity):
    activity_segments = [seg[0] for seg in segments if activity in seg[1]]
    group.create_dataset(activity, data=activity_segments)

# Assign variables to each person's data abd the activities they are performing.
members = ['Mile', 'Jeev', 'Chris']
activities = ['walking', 'jumping']
all_data = {}

# Loop through each person and activity, and read in the data from the CSV files
for member in members:
    all_data[member] = {}
    for activity in activities:
        all_data[member][activity] = read_csv_data(f'{activity}-{member}.csv')

all_segments = []
for member, member_data in all_data.items():
    for activity in activities:
        data = member_data[activity]
        segments_labels = create_segments_labels(member, activity, data, window_size =125, step_size=25)
        all_segments.extend(segments_labels)

# Randomize the segments and split into train and test sets, the train set will be 90% of the data, and the test set will be 10% of the data.
np.random.shuffle(all_segments)
num_train = int(0.9 * len(all_segments))
train_segments = all_segments[:num_train]
test_segments = all_segments[num_train:]

# Save the data to HDF5 called Data.h5
with h5py.File('Data.h5', 'w') as f:
    # Create a group for each person and save the data for each activity
    for member, member_data in all_data.items():
        member_group = f.create_group(member)
        # Creating the dataset for walking an jumping for each person
        for activity in activities:
            member_group.create_dataset(activity, data=member_data[activity])
    # Create a group for the dataset that has the train and test root groups
    dataset_group = f.create_group('dataset')
    # Create a group for the train and test datasets to go in the data group.
    train_group = dataset_group.create_group('train')
    test_group = dataset_group.create_group('test')

    # Creating the dataset for walking an jumping for each person
    for activity in activities:
        create_activity_datasets(train_group, train_segments, activity)
        create_activity_datasets(test_group, test_segments, activity)
f.close()