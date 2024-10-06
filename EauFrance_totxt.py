#%% Developed by Deborah, 2nd of October 2024
# Reads the CSV from https://www.hydro.eaufrance.fr/login 

import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the path for the folder containing CSV files and the output folder for txt files
input_folder = r"D:\My Documents\LoFlowMaas\Discharge\FR\Donne"
output_folder = os.path.join(input_folder, 'station_outputs')

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # Read the CSV file
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)
        
        # Ensure the 'Date (TU)' column is converted to datetime and sorted
        df['Date (TU)'] = pd.to_datetime(df['Date (TU)'])
        df.sort_values('Date (TU)', inplace=True)
        
        # Create a full date range between the minimum and maximum dates in the dataset
        full_date_range = pd.date_range(start=df['Date (TU)'].min(), end=df['Date (TU)'].max())
        
        # Reindex the dataframe to include missing dates, with NaN for missing values
        df_new = df.set_index('Date (TU)').reindex(full_date_range).reset_index()
        df_new.columns = ['Date', 'Discharge', 'Statut', 'Qualification', 'Méthode', 'Continuité']
        
        # Only keep 'Date' and 'Discharge', fill NaN for missing values
        df_final = df_new[['Date', 'Discharge']]
        df_final['Date'] = pd.to_datetime(df_final['Date'])  # Ensure dates are still datetime objects
        
        # Plot the data for inspection
        plt.figure(figsize=(10, 6))
        plt.plot(df_final['Date'], df_final['Discharge'], marker='o', linestyle='-', label=filename)
        plt.title(f'Discharge Data for {filename}')
        plt.xlabel('Date')
        plt.ylabel('Discharge (m³/s)')
        plt.grid(True)
        plt.legend()
        plt.show()


#%% End of Script
