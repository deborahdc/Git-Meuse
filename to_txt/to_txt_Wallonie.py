#%% Developed by Deborah, 2nd of October 2024
# Reads the excel from https://hydrometrie.wallonie.be/home/observations/debit.html?mode=map&station=DGH%2F7371 

import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the path for the folder containing Excel files and the output folder for txt files
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BE-W\Wallonie"
output_folder = os.path.join(input_folder, 'station_outputs')

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".xlsx"):
        # Read the Excel file, ensuring the correct header is used
        file_path = os.path.join(input_folder, filename)
        df = pd.read_excel(file_path, skiprows=10, header=None)  # Read without assuming column names

        # Inspect the DataFrame structure
        print(f"Inspecting file: {filename}")
        print(df.head())  # Check the first few rows of the data

        # Attempt to identify and name the columns based on file structure
        if df.shape[1] >= 2:  # Ensure there are at least two columns
            df.columns = ['Date', 'Discharge']  # Name the first two columns appropriately
            
            # Convert the 'Date' column to datetime and format as YYYYMMDD
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d')

            # Format the 'Discharge' column to two decimal places
            df['Discharge'] = df['Discharge'].round(2)

            # Extract station number from the file name (e.g., L6900 from DCENN_L6900_Q_DayMean)
            station_code = filename.split('_')[1]

            # Write the data to a txt file with the station code as the filename
            output_file = os.path.join(output_folder, f"{station_code}.txt")
            df.to_csv(output_file, sep=' ', index=False, header=True, float_format='%.2f', na_rep='NaN')

            # # Plot the data for inspection (optional)
            # plt.figure(figsize=(10, 6))
            # plt.plot(pd.to_datetime(df['Date']), df['Discharge'], marker='o', linestyle='-', label=station_code)
            # plt.title(f'Discharge Data for {station_code}')
            # plt.xlabel('Date')
            # plt.ylabel('Discharge (m³/s)')
            # plt.grid(True)
            # plt.legend()
            # plt.show()

            print(f"Processed {filename} into {output_file}")
        else:
            print(f"Unexpected format or insufficient data in {filename}")


#%%