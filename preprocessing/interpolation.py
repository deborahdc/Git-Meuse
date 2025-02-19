#%% Developed by Deborah, 14h of November 2024
# Interpolates gaps in the dataset up to x (15-20) days, recommended by Van Loon (2024)
import pandas as pd
import os

# Define paths
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\All"
output_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each .txt file in the folder
for file in os.listdir(input_folder):
    if file.endswith('.txt'):
        file_path = os.path.join(input_folder, file)
        
        try:
            # Read the file with proper column handling
            data = pd.read_csv(
                file_path,
                delim_whitespace=True, 
                names=["Date", "Discharge"],
                skiprows=1,  
                dtype={"Date": str, "Discharge": float}  
            )
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            continue

        # Convert 'Date' column to datetime
        try:
            data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d', errors='coerce')
        except Exception as e:
            print(f"Error parsing dates in file {file}: {e}")
            continue

        # Handle rows with invalid dates
        if data['Date'].isnull().all():
            print(f"Skipping {file}: all dates are invalid.")
            continue

        # Remove duplicate dates by keeping the first occurrence
        data = data.drop_duplicates(subset='Date', keep='first')

        # Set 'Date' as index
        data.set_index('Date', inplace=True)

        # Resample to daily frequency
        data = data.asfreq('D')

        # Interpolate gaps up x days   # Here is basically the line to interpolate
        data['Discharge'] = data['Discharge'].interpolate(
            method='linear',
            limit=20,
            limit_direction='both'
        )

        # Reset index to save the data back to the original format
        data.reset_index(inplace=True)

        # Format the 'Date' back to the original YYYYMMDD format
        data['Date'] = data['Date'].dt.strftime('%Y%m%d')

        # Save back in the same format as the input
        output_file = os.path.join(output_folder, file)
        try:
            data.to_csv(
                output_file,
                sep=' ',
                index=False,
                header=True,
                float_format='%.2f' 
            )
            print(f"Processed and saved: {output_file}")
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")

#%%
