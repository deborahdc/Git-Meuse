#%% Developed by Deborah, 2nd of October 2024
# Reads csv from https://www.dov.vlaanderen.be/portaal/?module=verkenner

import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the path for the folder containing CSV files and the output folder for txt files
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BE-F"
output_folder = os.path.join(input_folder, 'station_outputs')

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process each CSV file
def process_csv(file_path):
    # Read the CSV file, skipping the first 8 rows
    df = pd.read_csv(file_path, skiprows=8, usecols=[0, 1], header=None, names=['Date', 'Discharge'], delimiter=';', encoding='ISO-8859-1')
    
    # Convert the 'Date' column to datetime, handling mixed timezones by using utc=True
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True).dt.strftime('%Y%m%d')

    # Convert discharge values to numeric (comma to dot conversion)
    df['Discharge'] = pd.to_numeric(df['Discharge'].astype(str).str.replace(',', '.'), errors='coerce')

    # Format the 'Discharge' column to two decimal places
    df['Discharge'] = df['Discharge'].round(2)

    # Filter out rows where the Date or Discharge is NaN
    df.dropna(subset=['Date', 'Discharge'], inplace=True)

    return df

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # Read the CSV file and extract station name from the second row, second column
        file_path = os.path.join(input_folder, filename)
        
        # Extract station name from the second row, second column
        station_info = pd.read_csv(file_path, nrows=2, header=None, delimiter=';', encoding='ISO-8859-1')
        station_name = station_info.iloc[1, 1]  # Second row, second column
        
        # Process the data
        df = process_csv(file_path)

        # Replace any invalid characters in station name for file naming (e.g., '/', '\' cannot be used in filenames)
        valid_station_name = station_name.replace("/", "_").replace("\\", "_")
        
        # Write the processed data to a txt file with the station name as the filename
        output_file = os.path.join(output_folder, f"{valid_station_name}.txt")
        df.to_csv(output_file, sep=' ', index=False, header=True, float_format='%.2f', na_rep='NaN')

        # # Plot the data for inspection (optional)
        # plt.figure(figsize=(10, 6))
        # plt.plot(pd.to_datetime(df['Date'], format='%Y%m%d'), df['Discharge'], marker='o', linestyle='-', label=valid_station_name)
        # plt.title(f'Discharge Data for {valid_station_name}')
        # plt.xlabel('Date')
        # plt.ylabel('Discharge (m³/s)')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        print(f"Processed {filename} into {output_file}")
    else:
        print(f"Skipped non-CSV file: {filename}")

#%%
