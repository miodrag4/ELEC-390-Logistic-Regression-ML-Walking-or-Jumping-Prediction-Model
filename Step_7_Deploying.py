import numpy as np
import pandas as pd
from scipy.stats import skew
import joblib
import tkinter as tk
from tkinter import PhotoImage, filedialog
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to compute the features of the test and train data
def compute_features(sensor_data):
    max_value, min_value = np.max(sensor_data), np.min(sensor_data)
    range_value = max_value - min_value
    mean_value, median_value = np.mean(sensor_data), np.median(sensor_data)
    var_value, skew_value = np.var(sensor_data), skew(sensor_data)
    rms_value = np.sqrt(np.mean(sensor_data ** 2))
    kurtosis_value = np.mean((sensor_data - mean_value) ** 4) / (var_value ** 2)
    std_value = np.std(sensor_data)
    
    return max_value, min_value, range_value, mean_value, median_value, var_value, skew_value, rms_value, kurtosis_value, std_value

# Function to extract the features from the test data
def extraction_of_data(windows):
    w_size = 5
    # Filter the data using a rolling mean
    filtered_data = np.zeros((windows.shape[0], windows.shape[1] - w_size + 1, windows.shape[2]))
    
    for j in range(windows.shape[0]):
        sma = pd.DataFrame(windows[j]).rolling(w_size).mean().values
        sma = sma[w_size - 1:, :]
        filtered_data[j] = sma

    num_windows, num_samples, num_axes = filtered_data.shape
    features = np.zeros((num_windows, 10, num_axes))

    # Iterate over each time window and extract the features
    for i in range(num_windows):
        for j in range(num_axes):
            w_data = filtered_data[i, :, j]
            features[i, :, j] = compute_features(w_data)

    # Concatenate the features
    all_features = np.vstack([features[:, :, i] for i in range(num_axes)])
    # Create labels for the data
    labels = np.repeat(np.arange(1, num_axes + 1), num_windows).reshape(-1, 1)
    # Concatenate the labels
    all_features = np.hstack((all_features, labels))

    return all_features

# Function to process the data using the trained model from previous steps
def processData():
    # Error handling when file is not selected, or name is not entered.
    if not file_path:
        tk.messagebox.showerror(title="Error", message="Please Select a File!")
        return
    if not file_name:
        tk.messagebox.showerror(title="Error", message="Please Enter a Name for the Output file!")
        return
    # Read the data from the file
    input_data = pd.read_csv(file_path)
    time_data = input_data['Time (s)']
    data = input_data.drop(columns=['Time (s)'])
    # Create a time window of 300ms
    window_time = 500
    num_rows = len(data)
    num_windows = num_rows // window_time
    num_rows = num_windows * window_time
    data = data.iloc[:num_rows]

    data_array = np.stack([data.iloc[i:i+window_time, :] for i in range(0, len(data), window_time)])
    # Extract the features from the data array
    data_features = extraction_of_data(data_array)
    # Create the output file rows and columns to be written
    column_labels = np.array( ['max_value', 'min_value', 'range_value', 'mean_value', 'median_value', 'var_value', 'skew_value', 'rms_value', 'kurt_value','std_value', 'measurement'])
    dataset = pd.DataFrame(data_features, columns=column_labels)
    
    X_combined = dataset.groupby('measurement').apply(lambda group: group.iloc[:, :-1].values).values
    X_combined = np.hstack(X_combined)
    # Load the trained model
    clfCombined = joblib.load('classifier.joblib')
    # Predict the output
    Y_predicted = clfCombined.predict(X_combined)
    Y_output = np.reshape(Y_predicted, (-1, 1))

    # Add the time information to the output file
    times = time_data[:len(Y_output) * window_time:window_time].values.reshape(-1, 1)
    output_data = pd.DataFrame(np.hstack((Y_output, times)), columns=['activity', 'start_time'])

    # Map activity numbers to names
    output_data['activity'] = output_data['activity'].replace({0: 'walking', 1: 'jumping'})

    # Save the output file
    output_data.to_csv(file_name, index=False)
    
    # Display a message box to show that the output file has been created
    tk.messagebox.showinfo(title="Done", message="Output file has been outputted and saved in your current directory!")

    return output_data

# Function to the plot the results from the output file
def plot_data():
    data = pd.read_csv(file_name)

    activity_data = data.loc[:, 'activity']

    # Create a DataFrame to store the walking and jumping probabilities
    activity_prob = pd.DataFrame(columns=['Walking', 'Jumping'])
    # Loop to determine walking or jumping
    for activity in activity_data:
        if activity == 'walking':
            activity_prob = activity_prob.append({'Walking': 1, 'Jumping': 0}, ignore_index=True)
        else:
            activity_prob = activity_prob.append({'Walking': 0, 'Jumping': 1}, ignore_index=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot the data as a stacked bar chart
    activity_prob.plot(kind='bar', stacked=True, ax=ax)

    # Set Plot attributes
    ax.set_xlabel('Window Number')
    ax.set_ylabel('Activity Classification')
    ax.set_title('Walking and Jumping Activities by Window Number')
    ax.legend()

    # Space out the x-axis ticks and labels
    xticks_spacing = 5
    ax.set_xticks(list(range(0, len(activity_prob), xticks_spacing)))
    ax.set_xticklabels(list(range(1, len(activity_prob) + 1, xticks_spacing)))

    # Create a new window to display the plot instead of the UI window
    plot_window = tk.Toplevel()
    plot_window.title("Activity Plot")
    plot_window.geometry("800x600")
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
# GUI functions for UI elements and button actions. The three main functions are get_file_path(), get_file_name(), and processData()
# which are called when the respective buttons are pressed. The other functions are helper functions for the main functions.
# The main window is created in the main_window() function. The main window is then called in the main() function.
# The main() function is called at the end of the file.

# Gets the file path from the file explorer to be inputted.
def get_file_path():
    global file_path
    global output_data

    file_path = filedialog.askopenfilename()
    selected_file_label.config(text=os.path.basename(file_path))

# Gets the file name from the text box the user entered
def get_file_name():
    global file_name
    file_name = inputtxt.get("1.0", "end-1c")

# Clear the entries
def clear_selection_and_file_name():
    inputtxt.delete("1.0", "end")
    selected_file_label.config(text="")
    fileName = ""

# Main function for the UI window, all elements are created and packed into the window.
def main_window():
    # Window size and name of the window
    window = tk.Tk()
    window.geometry("700x600")
    window.title("ELEC 390 Project")
    window.configure(bg='#B3E5FC')

    # Logo for the window and "game"
    logo_path = "C:/Users/miles/OneDrive - Queen's University/Eng Year 3 - 2022-2023/Sem 2/ELEC 390/Project-V2/Logo.png"
    logo = PhotoImage(file=logo_path).subsample(2, 2)
    logo_label = tk.Label(window, image=logo, bg='#B3E5FC')
    logo_label.pack(pady=10)

    # Ttile fo the window display at the top
    title_label = tk.Label(window, text="Welcome to are you Walking or Jumping?", font=("Arial", 16, "bold"), bg='#B3E5FC')
    title_label.pack(pady=10)

    # Button to select a file
    select_file_button = tk.Button(window, text="Select File", command=get_file_path, bg='#1976D2', fg='white', width=15)
    select_file_button.pack(pady=10)

    # Tell the user to select a file output name
    tk.Label(window, text="Enter Output File Name:", bg='#B3E5FC').pack()

    global inputtxt
    inputtxt = tk.Text(window, height=1, width=40)
    inputtxt.pack(pady=5)

    # Submit button to submit the file name
    submit = tk.Button(window, text="Submit", command=lambda: [get_file_name(), processData()], bg='#1976D2', fg='white', width=15)
    submit.pack(pady=10)
    
    # Option to show the plot
    show_plot_button = tk.Button(window, text="Show Plot", command=plot_data, bg='#1976D2', fg='white', width=15)
    show_plot_button.pack(pady=10)

    # Cleat button
    clear_selection_button = tk.Button(window, text="Clear Selection", command=clear_selection_and_file_name, bg='#1976D2', fg='white', width=15)
    clear_selection_button.pack(pady=10)

    # Show the user the selected file.
    global selected_file_label
    selected_file_label = tk.Label(window, text="", font=("Arial", 10), bg='#B3E5FC')
    selected_file_label.pack(pady=5)

    # Run the main window
    window.mainloop()

if __name__ == "__main__":
    main_window()