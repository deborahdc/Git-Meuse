#%% Developed by Deborah, November 2024

# Apply 30-Day Moving Average to All Stations and save

import pandas as pd
import numpy as np
import os

def apply_moving_average(data, window_size=30):
    # Ensure the data is sorted by Date
    data = data.sort_index()

    # Apply the centered moving average
    data['Discharge_smoothed'] = data['Discharge'].rolling(
        window=window_size, center=True, min_periods=30
    ).mean()

    return data

# Filters out years with lower availability
def mask_low_data_availability(data, min_days=20):
    # Group by Year and Month, count valid days per month
    monthly_counts = data['Discharge_smoothed'].groupby([data.index.year, data.index.month]).count()

    # Identify years to discard (any year with a month < min_days valid entries)
    years_to_discard = monthly_counts.groupby(level=0).apply(
        lambda monthly: any(monthly < min_days)
    )

    # Mask all data for the identified years
    for year, discard in years_to_discard.items():
        if discard:
            data.loc[data.index.year == year, 'Discharge_smoothed'] = np.nan
            print(f'  [Info] Masked entire year {year} due to insufficient data in one or more months.')

    return data

def process_all_files(input_dir, output_dir, window_size=30, min_days_per_month=20):
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f'Error: Input directory "{input_dir}" does not exist.')
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')
    else:
        print(f'Output directory already exists: {output_dir}')

    # Iterate over all .txt files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            gauge_id = os.path.splitext(filename)[0] 
            print(f'\nProcessing Gauge ID: {gauge_id}')

            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)  

            # Read the discharge data
            try:
                data = pd.read_csv(input_file_path, delimiter=' ', header=0)
                print(f'  Read data from {filename}')
            except Exception as e:
                print(f'  [Error] Failed to read {filename}: {e}. Skipping this file.')
                continue

            # Check if 'Date' and 'Discharge' columns exist
            if 'Date' not in data.columns or 'Discharge' not in data.columns:
                print(f'  [Warning] "Date" or "Discharge" column missing in {filename}. Skipping this file.')
                continue

            # Convert 'Date' column to datetime format
            try:
                data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d', errors='coerce')
            except Exception as e:
                print(f'  [Error] Failed to parse dates in {filename}: {e}. Skipping this file.')
                continue

            # Drop rows with invalid dates
            initial_count = len(data)
            data = data.dropna(subset=['Date'])
            final_count = len(data)
            if final_count < initial_count:
                print(f'  [Info] Dropped {initial_count - final_count} rows due to invalid dates.')

            # Set 'Date' as the index
            data = data.set_index('Date').sort_index()

            # Apply the moving average
            data = apply_moving_average(data, window_size=window_size)
            print(f'  Applied {window_size}-day moving average to "Discharge"')

            # Mask months with fewer than min_days_per_month of data
            data = mask_low_data_availability(data, min_days=min_days_per_month)

            # Prepare the DataFrame for saving
            data_to_save = data[['Discharge_smoothed']].rename(columns={'Discharge_smoothed': 'Discharge'})

            # Reset index to have 'Date' as a column
            data_to_save = data_to_save.reset_index()

            # Save the smoothed data to the output directory
            try:
                data_to_save.to_csv(
                    output_file_path,
                    sep=' ',
                    index=False,
                    header=True,
                    date_format='%Y%m%d',  # Save Date in 'yyyymmdd' format
                    float_format='%.3f'
                )
                print(f'  Saved smoothed data to {filename}')
            except Exception as e:
                print(f'  [Error] Failed to save {filename}: {e}')

    print('\nAll files have been processed.')

if __name__ == '__main__':
    # Define paths
    input_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated'  
    output_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated\moving_average' 
    window_size = 30  # moving window
    min_days_per_month = 20  # Minimum number of valid days per month to filter year


    process_all_files(
        input_dir=input_dir,
        output_dir=output_dir,
        window_size=window_size,
        min_days_per_month=min_days_per_month
    )

#%%
