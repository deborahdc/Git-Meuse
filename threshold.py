
#%% Developed by Deborah, November 2024

# After this code, you have the 20th percentile thresholds in mm/day for each gauge_id based on one or multiple paired_ids that were weighted

# This step calculates the thresholds
import pandas as pd
import numpy as np
import os

# Function to calculate daily 20th percentile thresholds, possibility of change smoothing window if it was not enough
def calculate_daily_20th_percentile_threshold(paired_data, reference_start_year=1980, reference_end_year=2020, smoothing_window=10):
    # Filter the reference period
    reference = paired_data[(paired_data.index.year >= reference_start_year) & (paired_data.index.year <= reference_end_year)].copy()

    # Add month-day as a string column (mmdd format) to group by specific calendar days
    reference['mmdd'] = reference.index.strftime('%m%d')

    # Calculate the 20th percentile for each day across all years
    daily_thresholds = reference.groupby('mmdd')['Discharge_mm_per_day'].quantile(0.2)

    # Generate a full calendar of dates for the reference period
    full_date_range = pd.date_range(start=f'{reference_start_year}-01-01', end=f'{reference_end_year}-12-31', freq='D')

    # Map thresholds to the full date range using the mmdd grouping
    full_output = pd.DataFrame({'Date': full_date_range})
    full_output['mmdd'] = full_output['Date'].dt.strftime('%m%d')
    full_output['Threshold'] = full_output['mmdd'].map(daily_thresholds)

    # Identify leap years in the reference period
    leap_years = [year for year in range(reference_start_year, reference_end_year + 1) if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))]
    leap_dates = pd.date_range(start=f'{reference_start_year}-01-01', end=f'{reference_end_year}-12-31', freq='D').intersection(pd.to_datetime([f'{year}-02-29' for year in leap_years]))
    
    # Interpolate the threshold for leap years (to avoid getting noise from it because there are less days available)
    for leap_date in leap_dates:
        feb_28 = leap_date - pd.Timedelta(days=1)
        mar_1 = leap_date + pd.Timedelta(days=1)
        feb_28_mmdd = feb_28.strftime('%m%d')
        mar_1_mmdd = mar_1.strftime('%m%d')

        if feb_28_mmdd in daily_thresholds and mar_1_mmdd in daily_thresholds:
            interpolated_value = (daily_thresholds.loc[feb_28_mmdd] + daily_thresholds.loc[mar_1_mmdd]) / 2
            full_output.loc[full_output['Date'] == leap_date, 'Threshold'] = interpolated_value

    # Apply a 10-day moving average smoothing to the thresholds (or not)
    full_output['Smoothed_Threshold'] = full_output['Threshold'].rolling(window=smoothing_window, center=True, min_periods=1).mean()

    # Format the output to match the required format
    full_output['Formatted_Date'] = full_output['Date'].dt.strftime('%Y%m%d')
    return full_output[['Formatted_Date', 'Smoothed_Threshold']]

# Function to save thresholds to file
def save_thresholds(output_data, output_file_path):
    output_data.to_csv(output_file_path, index=False, header=False, sep=' ')

# Main function to process all paired data files
def process_all_thresholds(directory_path, excel_file_path, output_directory, reference_start_year=1980, reference_end_year=2020):

    # Load gauge information from Excel
    gauge_info = pd.read_excel(excel_file_path)
    gauge_info['gauge_id'] = gauge_info['gauge_id'].astype(str)
    gauge_info['paired_id'] = gauge_info['paired_id'].astype(str)

    for _, row in gauge_info.iterrows():
        gauge_id = row['gauge_id']  # For saving results
        paired_ids = str(row['paired_id']).split(';')  # Handle multiple paired_ids (case of Borgharen for ex)
        paired_areas = [float(area) for area in str(row['paired_area']).split(';')]  # Handle multiple paired areas
        combined_data = []

        for paired_id, area in zip(paired_ids, paired_areas):
            paired_file_path = os.path.join(directory_path, f"{paired_id}.txt")
            try:
                paired_data = pd.read_csv(paired_file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
                paired_data['Date'] = pd.to_datetime(paired_data['Date'], format='%Y%m%d', errors='coerce')
                paired_data = paired_data.dropna(subset=['Date']).drop_duplicates(subset=['Date']).set_index('Date').sort_index()
                
                # Convert discharge to mm/day using the paired catchment area
                if area > 0:
                    paired_data['Discharge_mm_per_day'] = (
                        paired_data['Discharge'] * 86400 / (area * 1e6) * 1000
                    )
                    paired_data['Weight'] = area  # Assign weight based on area
                    combined_data.append(paired_data)
                else:
                    print(f"Invalid area for paired ID {paired_id}. Skipping.")
                    continue
            except Exception as e:
                print(f"Error reading file {paired_file_path} for paired ID {paired_id}: {e}")
                continue

        if not combined_data:
            print(f"No valid data for gauge ID {gauge_id}. Skipping.")
            continue

        # Combine data from multiple paired files and calculate weighted discharge
        combined_data = pd.concat(combined_data, axis=0)
        combined_data['Weighted_Discharge'] = combined_data['Discharge_mm_per_day'] * combined_data['Weight']
        daily_weighted_discharge = combined_data.groupby(combined_data.index.date).apply(
            lambda x: x['Weighted_Discharge'].sum() / x['Weight'].sum()
        )
        weighted_data = pd.DataFrame(daily_weighted_discharge, columns=['Discharge_mm_per_day'])
        weighted_data.index = pd.to_datetime(weighted_data.index)  # Convert index back to datetime

        # Calculate thresholds with smoothing
        thresholds = calculate_daily_20th_percentile_threshold(weighted_data, reference_start_year, reference_end_year)

        # Save thresholds to the threshold folder using gauge_id
        output_file_path = os.path.join(output_directory, f'{gauge_id}.txt')
        save_thresholds(thresholds, output_file_path)
        print(f"Thresholds saved for gauge ID {gauge_id} to {output_file_path}")

# Define paths
directory_path = r'D:\My Documents\LoFlowMaas\Discharge\interpolated\moving_average' # If using already the moving averages, no need to add again, only if it is not smoothed enough
excel_file_path = r'D:\My Documents\LoFlowMaas\Discharge\Info_EStreams.xlsx'
output_directory = r'D:\My Documents\LoFlowMaas\Discharge\threshold\statistical_threshold'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Run the processing
process_all_thresholds(directory_path, excel_file_path, output_directory)




#%% Developed by Deborah, November 2024

# Calculates the p factors (based on one or more paired stations)

import pandas as pd
import numpy as np
import os

# Function to handle date correctly
def process_date_column(data):
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
    data.set_index('date', inplace=True)
    return data

# Function to calculate the observed/paired precipitation ratio with multiple paired basins
def calculate_precipitation_ratio(observed_met_file, paired_met_files, paired_areas, reference_period=(1980, 2020)):
    observed_data = pd.read_csv(observed_met_file)
    observed_data = process_date_column(observed_data)

    # Filter observed data for the reference period
    observed_data = observed_data.loc[
        (observed_data.index.year >= reference_period[0]) & (observed_data.index.year <= reference_period[1])
    ]

    # Remove months with more than x missing days in observed data and get moving average
    observed_data['missing'] = observed_data['p_mean'].isna()
    observed_data = observed_data.groupby([observed_data.index.year, observed_data.index.month]).filter(
        lambda x: x['missing'].sum() <= 10
    )
    observed_data['p_mean_30day'] = observed_data['p_mean'].rolling(window=30, center=True, min_periods=15).mean()

    # Process paired meteorology files
    paired_data_combined = []
    for paired_file, area in zip(paired_met_files, paired_areas):
        paired_data = pd.read_csv(paired_file)
        paired_data = process_date_column(paired_data)

        # Filter paired data for the reference period
        paired_data = paired_data.loc[
            (paired_data.index.year >= reference_period[0]) & (paired_data.index.year <= reference_period[1])
        ]

        paired_data['missing'] = paired_data['p_mean'].isna()
        paired_data = paired_data.groupby([paired_data.index.year, paired_data.index.month]).filter(
            lambda x: x['missing'].sum() <= 10
        )

        # Calculate 30-day moving average
        paired_data['p_mean_30day'] = paired_data['p_mean'].rolling(window=30, center=True, min_periods=15).mean()

        # Add weighted contribution based on the area
        paired_data['weighted_p_mean'] = paired_data['p_mean_30day'] * area
        paired_data_combined.append(paired_data)

    # Combine all paired data and calculate the weighted average
    paired_data_combined = pd.concat(paired_data_combined)
    paired_data_weighted = paired_data_combined.groupby(paired_data_combined.index).apply(
        lambda x: x['weighted_p_mean'].sum() / sum(x['weighted_p_mean'] / x['p_mean_30day'])
    )
    paired_data_weighted = pd.DataFrame(paired_data_weighted, columns=['p_mean_30day'])

    # Merge observed and paired data
    merged_data = observed_data[['p_mean_30day']].merge(
        paired_data_weighted, left_index=True, right_index=True, suffixes=('_obs', '_paired')
    )

    # Calculate precipitation ratio (obs / paired)
    merged_data['precipitation_ratio'] = merged_data['p_mean_30day_obs'] / merged_data['p_mean_30day_paired']

    # Group by day of year (mmdd) and calculate the average precipitation ratio for each day
    merged_data['mmdd'] = merged_data.index.strftime('%m%d')
    daily_avg_ratio = merged_data.groupby('mmdd')['precipitation_ratio'].mean()

    return daily_avg_ratio

# Function to interpolate February 29th in leap years only
def interpolate_feb_29(daily_avg_ratio, year):
    if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):  
        feb_27 = daily_avg_ratio.get('0227', None)
        feb_28 = daily_avg_ratio.get('0228', None)
        if feb_27 is not None and feb_28 is not None:
            feb_29 = (feb_27 + feb_28) / 2
            daily_avg_ratio['0229'] = feb_29
    return daily_avg_ratio

def handle_inf_values(ratio_data):
    ratio_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    ratio_data.fillna(method='ffill', inplace=True) 
    return ratio_data

# Function to save the precipitation ratio to a text file
def save_precipitation_ratio_to_txt(ratio_data, gauge_id, output_dir):
    output_file = os.path.join(output_dir, f"{gauge_id}.txt")
    output_data = []
    for year in range(1980, 2021):
        ratio_data_for_year = interpolate_feb_29(ratio_data.copy(), year)
        for mmdd, pfactor in ratio_data_for_year.items():
            yyyymmdd = f"{year}{mmdd}"
            pfactor_rounded = round(pfactor, 3)
            output_data.append([yyyymmdd, pfactor_rounded])
    output_df = pd.DataFrame(output_data, columns=["yyyymmdd", "pfactor"])
    output_df.to_csv(output_file, sep=' ', index=False, header=False)
    print(f"Precipitation ratio saved to {output_file}")

# Main function to process gauges with multiple paired basins
def process_gauges(excel_file_path, meteorology_dir, output_dir, reference_period=(1980, 2020)):
    basin_info = pd.read_excel(excel_file_path)
    for _, row in basin_info.iterrows():
        basin_id = str(row['basin_id'])
        paired_basin_ids = str(row['paired_basin_id']).split(';') if not pd.isna(row['paired_basin_id']) else []
        paired_areas = [float(area) for area in str(row['paired_area']).split(';')] if not pd.isna(row['paired_area']) else []
        gauge_id = str(row['gauge_id']) if not pd.isna(row['gauge_id']) else None

        # Check if gauge_id or paired_basin_ids are missing
        if gauge_id is None or not paired_basin_ids:
            print(f"Skipping basin {basin_id} due to missing paired basin or gauge ID.")
            continue

        # Check if the number of paired_basin_ids matches the number of paired_areas
        if len(paired_basin_ids) != len(paired_areas):
            print(f"Mismatch between paired basins and areas for basin {basin_id}, skipping.")
            continue

        observed_met_file = os.path.join(meteorology_dir, f"estreams_meteorology_{basin_id}.csv")
        paired_met_files = [
            os.path.join(meteorology_dir, f"estreams_meteorology_{paired_basin_id}.csv") for paired_basin_id in paired_basin_ids
        ]

        # Check file existence for observed and paired data
        if not os.path.exists(observed_met_file):
            print(f"Missing observed meteorology file for basin: {basin_id}")
            continue
        if not all(os.path.exists(pf) for pf in paired_met_files):
            print(f"Missing one or more paired meteorology files for basin: {basin_id}")
            continue

        try:
            # Calculate the precipitation ratio
            ratio_data = calculate_precipitation_ratio(observed_met_file, paired_met_files, paired_areas, reference_period)

            # Handle invalid values (e.g., NaN or infinite ratios)
            ratio_data = handle_inf_values(ratio_data)

            # Save the ratio data using the gauge_id
            save_precipitation_ratio_to_txt(ratio_data, gauge_id, output_dir)

        except Exception as e:
            print(f"Error processing basin {basin_id}: {e}")

# Example usage
excel_file_path = r"D:\My Documents\LoFlowMaas\Discharge\Info_EStreams.xlsx"
meteorology_dir = r"D:\My Documents\LoFlowMaas\EStreams_data\EStreams\meteorology"
output_dir = r"D:\My Documents\LoFlowMaas\Discharge\threshold\p_factor"
process_gauges(excel_file_path, meteorology_dir, output_dir, reference_period=(1980, 2020))

#%% Multiplies the thresholds by the p-factor

import pandas as pd
import os

# Function to multiply the thresholds by the precipitation factor (p_factor)
def multiply_thresholds_by_pfactor(thresholds_file, p_factors_file, output_dir, gauge_id):
    # Read the threshold data and p_factor data
    thresholds_data = pd.read_csv(thresholds_file, sep=' ', header=None, names=["yyyymmdd", "threshold"])
    p_factors_data = pd.read_csv(p_factors_file, sep=' ', header=None, names=["yyyymmdd", "pfactor"])

    # Merge the two datasets on 'yyyymmdd'
    merged_data = pd.merge(thresholds_data, p_factors_data, on="yyyymmdd", how="left")

    # Multiply the threshold by the precipitation factor (p_factor)
    merged_data["threshold_p"] = merged_data["threshold"] * merged_data["pfactor"]

    # Save the result to a new file named after the gauge_id
    output_file = os.path.join(output_dir, f"{gauge_id}.txt")
    merged_data[["yyyymmdd", "threshold_p"]].to_csv(output_file, sep=' ', index=False, header=False)

    print(f"Thresholds multiplied by p_factor saved to {output_file}")

# Main function to process thresholds and p_factors
def process_thresholds_and_p_factors(thresholds_dir, p_factors_dir, output_dir):
    # Get all threshold and p_factor files
    threshold_files = [f for f in os.listdir(thresholds_dir) if f.endswith('.txt')]
    p_factor_files = [f for f in os.listdir(p_factors_dir) if f.endswith('.txt')]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each pair of threshold and p_factor files
    for threshold_file in threshold_files:
        gauge_id = threshold_file.split('.')[0]  # Extract gauge_id from the file name
        
        # Match the corresponding p_factor file
        p_factor_file = f"{gauge_id}.txt"
        if p_factor_file in p_factor_files:
            threshold_file_path = os.path.join(thresholds_dir, threshold_file)
            p_factor_file_path = os.path.join(p_factors_dir, p_factor_file)

            # Multiply thresholds by p_factor and save the result
            multiply_thresholds_by_pfactor(threshold_file_path, p_factor_file_path, output_dir, gauge_id)
        else:
            print(f"Missing p_factor file for {gauge_id}, skipping.")

thresholds_dir = r"D:\My Documents\LoFlowMaas\Discharge\threshold\statistical_threshold"
p_factors_dir = r"D:\My Documents\LoFlowMaas\Discharge\threshold\p_factor"
output_dir = r"D:\My Documents\LoFlowMaas\Discharge\threshold\threshold_p"

process_thresholds_and_p_factors(thresholds_dir, p_factors_dir, output_dir)
#%%


