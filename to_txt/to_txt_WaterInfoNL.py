#%% Developed by Deborah, 1st of October 2024
# Reads the CSV sent by RWS email based on the request from https://waterinfo.rws.nl/#/nav/thema
# Drops the outliers/strange values based on a threshold and replaces as NaN (), THRESHOLD NEEDS TO BE UPDATED after inspection

import pandas as pd
import matplotlib.pyplot as plt
import os

# Input and output directories
file_path = r'D:\My Documents\LoFlowMaas\Discharge\NL\20241011_035\20241011_035.csv' # One by one in this code

output_dir = r'D:\My Documents\LoFlowMaas\Discharge\NL\station_outputs'
input_dir = output_dir  # Keep output files for next steps

#%% Section 1: Process and save raw data

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
data = pd.read_csv(file_path, delimiter=';', encoding='ISO-8859-1', low_memory=False)

# Convert date and time columns to datetime
data['WAARNEMINGDATUM'] = pd.to_datetime(data['WAARNEMINGDATUM'], format='%d-%m-%Y', errors='coerce')
data['WAARNEMINGTIJD (MET/CET)'] = pd.to_datetime(data['WAARNEMINGTIJD (MET/CET)'], format='%H:%M:%S', errors='coerce')

# Convert discharge values to numeric (comma to dot conversion)
data['ALFANUMERIEKEWAARDE'] = pd.to_numeric(data['ALFANUMERIEKEWAARDE'].astype(str).str.replace(',', '.'), errors='coerce')

# Combine date and time into a single datetime column
data['Datetime'] = pd.to_datetime(data['WAARNEMINGDATUM'].astype(str) + ' ' + data['WAARNEMINGTIJD (MET/CET)'].astype(str), errors='coerce')

# Process each station's data
meetpunt_identificaties = data['MEETPUNT_IDENTIFICATIE'].unique()
for meetpunt in meetpunt_identificaties:
    meetpunt_data = data[data['MEETPUNT_IDENTIFICATIE'] == meetpunt]
    
    # Save raw data to file with consistent "YYYYMMDD" date format and rounded to 2 decimals
    meetpunt_data['Date'] = meetpunt_data['Datetime'].dt.strftime('%Y%m%d')
    meetpunt_data['ALFANUMERIEKEWAARDE'] = meetpunt_data['ALFANUMERIEKEWAARDE'].round(2)
    
    output_file_raw = os.path.join(output_dir, f"{meetpunt}_hourly_discharge_raw.txt")
    meetpunt_data[['Date', 'ALFANUMERIEKEWAARDE']].to_csv(output_file_raw, sep=' ', index=False, header=['Date', 'Discharge'])
    print(f"Raw data for station {meetpunt} saved to {output_file_raw}")

#%% Section 1.2 Inspecting Plot

file_path = r'D:\My Documents\LoFlowMaas\Discharge\NL\station_outputs\Eijsden grens_hourly_discharge_raw.txt' # Change to check other stations
data = pd.read_csv(file_path, sep=' ', header=0)  # Assuming space-separated file with 'Date' and 'Discharge' columns

# Convert the 'Date' column back to datetime for plotting
data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Discharge'], marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Discharge (m³/s)')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()

# Show the plot
plt.show()

#%% Section 2: Filter data based on threshold (Read from raw output)

# Threshold for filtering
threshold = 1000000  # Update this threshold after inspecting the raw data

# List all raw discharge files
files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]

for file in files:
    file_path = os.path.join(output_dir, file)
    
    # Read the raw discharge file
    meetpunt_data = pd.read_csv(file_path, delimiter=' ')
    
    # Filter out values exceeding the threshold and replace with NaN
    meetpunt_data['Filtered_Discharge'] = meetpunt_data['Discharge'].where(meetpunt_data['Discharge'] <= threshold, pd.NA)
    
    # Save the filtered data with the "YYYYMMDD" format and rounded to 2 decimals
    meetpunt_data['Filtered_Discharge'] = meetpunt_data['Filtered_Discharge'].round(2)
    output_file_filtered = os.path.join(output_dir, f"{file.split('_')[0]}_hourly_discharge_filtered.txt")
    meetpunt_data[['Date', 'Filtered_Discharge']].to_csv(output_file_filtered, sep=' ', index=False, header=['Date', 'Discharge'])
    print(f"Filtered data for station {file.split('_')[0]} saved to {output_file_filtered}")

#%% Section 3: Calculate and save daily averages from filtered data (excluding NaN values)
# Because raw data was hourly

# Create a directory for the daily average output
daily_avg_output_dir = os.path.join(output_dir, 'average')
os.makedirs(daily_avg_output_dir, exist_ok=True)

# List all filtered discharge files
files = [f for f in os.listdir(output_dir) if f.endswith('_hourly_discharge_filtered.txt')]

for file in files:
    # Read each filtered file
    file_path = os.path.join(output_dir, file)
    station_data = pd.read_csv(file_path, delimiter=' ')
    
    # Convert the 'Date' column back to datetime for resampling
    station_data['Date'] = pd.to_datetime(station_data['Date'], format='%Y%m%d')
    
    # Set date as index
    station_data.set_index('Date', inplace=True)
    
    # Group by day and calculate daily average (NaN values are automatically excluded)
    daily_avg = station_data['Discharge'].resample('D').mean()

    # Save daily average to a new file with the same format and rounded to 2 decimals
    daily_avg = daily_avg.round(2)
    output_file_avg = os.path.join(daily_avg_output_dir, f"{file.split('_')[0]}.txt")
    daily_avg.to_csv(output_file_avg, sep=' ', header=['Discharge'], date_format='%Y%m%d')
    
    print(f"Daily average discharge for station {file.split('_')[0]} saved to {output_file_avg}")


#%% 
