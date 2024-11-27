#%% Developed by Deborah, 10th of October 2024
# Reads the excel from https://www.elwasweb.nrw.de/elwas-web/index.xhtml;jsessionid=CACC9E9457213FB7A115E552E3170738#

import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the path for the folder containing CSV files and the output folder for txt files
input_folder = r"D:\My Documents\LoFlowMaas\Discharge\DE\Data"
output_folder = os.path.join(input_folder, 'station_outputs')

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # Read the CSV file starting from the 11th row (header is not assumed)
        file_path = os.path.join(input_folder, filename)
        try:
            # Specify the encoding and delimiter (assuming ";" based on typical European CSV formatting)
            df = pd.read_csv(file_path, skiprows=10, header=None, encoding='ISO-8859-1', sep=';', on_bad_lines='skip') 
            
            # Inspect the DataFrame structure
            print(f"Inspecting file: {filename}")
            print(df.head())  # Check the first few rows of the data

            # Attempt to identify and name the columns based on file structure
            if df.shape[1] >= 2:  # Ensure there are at least two columns
                df.columns = ['Date', 'Discharge']  # Name the first two columns appropriately

                # Convert the 'Date' column to datetime and format as YYYYMMDD
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.strftime('%Y%m%d')

                # Replace commas in 'Discharge' column with dots and convert to float
                df['Discharge'] = df['Discharge'].astype(str).str.replace(',', '.').astype(float)

                # Replace negative discharge values with NaN
                df.loc[df['Discharge'] < 0, 'Discharge'] = pd.NA

                # Format the 'Discharge' column to two decimal places
                df['Discharge'] = df['Discharge'].round(2)

                # Extract the station number from the filename (after "_" and before ".csv")
                station_code = filename.split('_')[1].split('.')[0]  # Extract the part after _ and before .csv

                # Write the data to a txt file with the station code as the filename
                output_file = os.path.join(output_folder, f"{station_code}.txt")
                df.to_csv(output_file, sep=' ', index=False, header=True, float_format='%.2f', na_rep='NaN')

                # Plot the data for inspection (optional)
                plt.figure(figsize=(10, 6))
                plt.plot(pd.to_datetime(df['Date']), df['Discharge'], marker='o', linestyle='-', label=station_code)
                plt.title(f'Discharge Data for {station_code}')
                plt.xlabel('Date')
                plt.ylabel('Discharge (m³/s)')
                plt.grid(True)
                plt.legend()
                plt.show()

                print(f"Processed {filename} into {output_file}")
            else:
                print(f"Unexpected format or insufficient data in {filename}")

        except UnicodeDecodeError as e:
            print(f"Error reading {filename}: {e}")
        except pd.errors.ParserError as e:
            print(f"Error parsing {filename}: {e}")

#%%
