#%% Developed by Deborah, 2nd of October 2024
# Reads the CSV from https://www.hydro.eaufrance.fr/login 
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set the path for the folder containing CSV files and the output folder for txt files
input_folder = r"D:\My Documents\LoFlowMaas\Discharge\FR\Donne"
output_folder = os.path.join(input_folder, 'station_outputs')

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        # Read the CSV file, skip the first row (header)
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path, skiprows=1)  # Skip the first row

        # Rename the columns to 'Date' and 'Discharge'
        df.columns = ['Date', 'Discharge', 'Statut', 'Qualification', 'Méthode', 'Continuité']

        # Keep only the 'Date' and 'Discharge' columns
        df_final = df[['Date', 'Discharge']].copy()

        # Format 'Date' column to match the format YYYYMMDD
        df_final['Date'] = pd.to_datetime(df_final['Date']).dt.strftime('%Y%m%d')

        # Clean up the filename for output file: Keep everything before the first underscore
        base_filename = filename.split('_')[0]
        txt_filename = f"{base_filename}.txt"
        txt_file_path = os.path.join(output_folder, txt_filename)

        # Write the dataframe to a text file with exactly one space between columns
        with open(txt_file_path, 'w') as f:
            f.write("Date Discharge\n")  # Write the header
            for _, row in df_final.iterrows():
                # Ensure exactly one space between Date and Discharge
                f.write(f"{row['Date']} {row['Discharge']:.2f}\n")  # No extra alignment formatting

        print(f"Processed and saved file: {txt_filename}")

        # plt.figure(figsize=(8, 4))
        # df_final['Date'] = pd.to_datetime(df_final['Date'], format='%Y%m%d')  # Convert back to datetime for plotting

        # plt.plot(df_final['Date'], df_final['Discharge'], linestyle='-', color='blue', linewidth=1.5)
        # plt.title(f'Discharge Data for {base_filename}', fontsize=10)
        # plt.xlabel('Year', fontsize=9)
        # plt.ylabel('Discharge (m³/s)', fontsize=9)
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Major ticks on years
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format the x-axis to show only the year
        # plt.xticks(rotation=45)  # Rotate the year labels slightly for readability

        # plt.tight_layout()
        # plt.show()  # Display the plot without saving


#%%