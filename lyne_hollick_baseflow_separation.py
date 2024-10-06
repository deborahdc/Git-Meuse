#%% Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing discharge data files
directory_path = r'D:\My Documents\LoFlowMaas\Discharge\FR\Donne\station_outputs'

# Function to apply Lyne-Hollick filter
def applyLHfilter(Qtotal, alpha=0.0, beta=2, npass=0, initopt=0):
    # create arrays for baseflow and quickflow
    Qquick= np.zeros(np.size(Qtotal), dtype=float)
    Qbase = Qtotal.copy()

    if initopt == 0:
        Qquick[0] = Qbase[0]

    # First forward pass
    for ii in np.arange(1, len(Qtotal)):
        Qquick[ii] = alpha * Qquick[ii - 1] + (1 + alpha) / beta * (Qtotal[ii] - Qtotal[ii - 1])
    
    # Calculate Qbase
    Qbase = np.where(Qquick > 0, Qtotal - Qquick, Qbase)
    
    # Sequence of backward and forward passes
    for nn in np.arange(0, npass):
        # Backward pass
        Qquick[-1] = 0
        if initopt == 0:
            Qquick[-1] = Qbase[-1]

        for ii in np.arange(len(Qtotal) - 1, 0, -1):
            Qquick[ii - 1] = alpha * Qquick[ii] + (1 + alpha) / beta * (Qbase[ii - 1] - Qbase[ii])

        Qbase = np.where(Qquick > 0, Qbase - Qquick, Qbase)

        # Forward pass
        Qquick[0] = 0
        if initopt == 0:
            Qquick[0] = Qbase[0]

        for ii in np.arange(1, len(Qtotal)):
            Qquick[ii] = alpha * Qquick[ii - 1] + (1 + alpha) / beta * (Qbase[ii] - Qbase[ii - 1])

        Qbase = np.where(Qquick > 0, Qbase - Qquick, Qbase)

    # Return separated baseflow and quickflow
    return Qbase, Qtotal - Qquick

# Function to process and plot data for each file
def process_file(file_path):
    # Extract station name from the filename 
    station_name = os.path.basename(file_path).replace('.txt', '')
    
    # Load discharge data (in cubic meters per second, m³/s)
    data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    
    # Set 'Date' as index for easier resampling
    data.set_index('Date', inplace=True)

    # Apply Lyne-Hollick filter
    Qtotal = data['Discharge'].values
    Qbase, Qquick = applyLHfilter(Qtotal, alpha=0.98, beta=2, npass=3, initopt=0)
    
    # Add the baseflow and quickflow to the DataFrame
    data['Baseflow'] = Qbase
    data['Quickflow'] = Qquick

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Discharge'], label='Total Discharge')
    plt.plot(data.index, data['Baseflow'], label='Baseflow', linestyle='--')
    plt.plot(data.index, data['Quickflow'], label='Direct runnof', linestyle=':')
    plt.title(f'Station: {station_name}')
    plt.xlabel('Date')
    plt.ylabel('Discharge (m³/s)')
    plt.legend()
    #plt.grid(True)
    plt.show()

    return data

# Process all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        processed_data = process_file(file_path)


#%%