#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing discharge data files

directory_path = r'D:\My Documents\LoFlowMaas\Discharge\BE-W\Wallonie\station_outputs'

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
    
    # Given catchment area (in km²)
    catchment_area_km2 = 100.0  # km²
    
    # Convert daily discharge (m³/s) to discharge in mm/day
    data['Discharge_mm_per_day'] = data['Discharge'] * 86400 / (catchment_area_km2 * 1e6) * 1000
    
    # Resample to monthly discharge (sum of daily discharge in mm/month)
    monthly_data = data['Discharge_mm_per_day'].resample('M').sum()
    
    # Define the reference period (before 2000)
    reference_period = monthly_data[monthly_data.index.year < 2000]
    
    # Calculate the 20% threshold for each month based on the reference period
    monthly_thresholds = reference_period.groupby(reference_period.index.month).quantile(0.2)
    
    # Now we will apply these thresholds to the data after 2000
    runs_period = monthly_data[monthly_data.index.year >= 2000]
    
    # Create a new column for threshold in the runs_period based on the month
    runs_period = runs_period.to_frame(name='Discharge_mm')
    runs_period['Threshold_20'] = runs_period.index.month.map(monthly_thresholds)
    
    # Identify low flow periods (below 20% threshold)
    low_flow_periods = runs_period[runs_period['Discharge_mm'] < runs_period['Threshold_20']]
    
    # Create a low flow marker
    plotLowflow = pd.DataFrame(index=runs_period.index, columns=['Lowflow'])
    for mm in plotLowflow.index:
        if runs_period.loc[mm, 'Discharge_mm'] <= runs_period.loc[mm, 'Threshold_20']:
            plotLowflow.at[mm, 'Lowflow'] = -3  # Set a negative value for visibility
    
    # Plotting
    plt.figure(figsize=(16, 6))
    
    # Plot the discharge data after 2000
    plt.plot(runs_period.index, runs_period['Discharge_mm'], label='Discharge', color='black', linewidth=1.3)
    
    # Plot the 20% threshold for each month
    plt.plot(runs_period.index, runs_period['Threshold_20'], label='20% flow threshold', color='black', linestyle='--', linewidth=1.3)
    
    # Shade the area below the threshold
    plt.fill_between(runs_period.index, runs_period['Discharge_mm'], runs_period['Threshold_20'], 
                     where=(runs_period['Discharge_mm'] < runs_period['Threshold_20']), 
                     color='orange', alpha=0.5, label='Low flow period')
    
    # Plot the low flow periods as markers
    plt.plot(plotLowflow.index, plotLowflow['Lowflow'], color='orange', linewidth=2, linestyle='-', marker='s', markersize=4)
    
    # Add title and labels
    plt.title(f'{station_name}')
    plt.xlabel('Date')
    plt.ylabel('Discharge (mm/month)')
    plt.legend()
    
    plt.show()

# Loop through all files in the directory that end with '.txt'
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        process_file(file_path)


#%%