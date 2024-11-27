#%% Developed by Deborah, November 2024

import pandas as pd
import numpy as np
import os
import warnings


# Paths
discharge_dir = r"D:\My Documents\LoFlowMaas\Discharge\interpolated\moving_average"  # Path to the pre-calculated moving average data
excel_file_path = r"D:\My Documents\LoFlowMaas\Discharge\Info_EStreams.xlsx" # Path to the excel containing data information
thresholds_folder = r"D:\My Documents\LoFlowMaas\Discharge\threshold\threshold_p"  # Path to thresholds folder (already corrected with p factor)
deficit_output_dir = r"D:\My Documents\LoFlowMaas\Discharge\deficit"  # Output path
meteorology_dir = r"D:\My Documents\LoFlowMaas\EStreams_data\EStreams\meteorology"  # Meteorology data directory
p_factor_dir = r"D:\My Documents\LoFlowMaas\Discharge\threshold\p_factor"  # Path to p_factor folder

# Ensure the deficit directory exists
os.makedirs(deficit_output_dir, exist_ok=True)

# Loads basin information from an Excel file.
def load_basin_info(excel_file_path):
    try:
        basin_info = pd.read_excel(excel_file_path)
        return basin_info
    except Exception as e:
        return pd.DataFrame()

# Convert discharge from cubic meters per second (m³/s) to millimeters per day (mm/day)
def discharge_m3s_to_mmday(discharge_m3_per_s, catchment_area_km2):
    if catchment_area_km2 is None or catchment_area_km2 <= 0:
        return np.nan
    discharge_mm_per_day = (discharge_m3_per_s * 86400) / (catchment_area_km2 * 1e6) * 1000
    return discharge_mm_per_day

# Load the threshold file for a specific gauge ID
def load_threshold_file(thresholds_folder, gauge_id):
    threshold_file_path = os.path.join(thresholds_folder, f"{gauge_id}.txt")
    if not os.path.exists(threshold_file_path):
        raise FileNotFoundError(f"Threshold file not found for gauge {gauge_id}: {threshold_file_path}")
    threshold_data = pd.read_csv(
        threshold_file_path,
        delimiter=' ',
        header=None,
        names=['Date', 'Threshold']
    )
    threshold_data['Date'] = pd.to_datetime(threshold_data['Date'], format='%Y%m%d', errors='coerce')
    threshold_data = threshold_data.dropna(subset=['Date']).set_index('Date')
    
    return threshold_data['Threshold']

# Load the p_factor file for a specific gauge ID
def load_p_factor_file(p_factor_dir, gauge_id):
    p_factor_file_path = os.path.join(p_factor_dir, f"{gauge_id}.txt")
    if not os.path.exists(p_factor_file_path):
        raise FileNotFoundError(f"P-factor file not found for gauge {gauge_id}: {p_factor_file_path}") 
    p_factor_data = pd.read_csv(
        p_factor_file_path,
        delimiter=' ',
        header=None,
        names=['Date', 'p_factor']
    )
    p_factor_data['Date'] = pd.to_datetime(p_factor_data['Date'], format='%Y%m%d', errors='coerce')
    p_factor_data = p_factor_data.dropna(subset=['Date']).set_index('Date')
    
    return p_factor_data['p_factor']

# Assign hydrological year to each date. Hydrological year starts in October
# For example, October 1980 - September 1981 is assigned to 1981
def assign_year(dates):
    return dates.year + (dates.month >= 10).astype(int)

# Apply the precipitation factor to the paired discharge data
def apply_precipitation_factor(paired_data, p_factor_data):
    paired_data = paired_data.sort_index()

    # Join precipitation factor data with paired discharge data
    paired_data = paired_data.join(p_factor_data, how='left')

    # Fill missing p_factor values with 1 (no correction)
    paired_data['p_factor'].fillna(1.0, inplace=True)

    # Apply precipitation factor to the discharge
    paired_data['Adjusted_Discharge_mm_per_day'] = paired_data['Discharge_mm_per_day'] * paired_data['p_factor']

    return paired_data['Adjusted_Discharge_mm_per_day']

# Calculate annual deficit volumes and total days below threshold, and excludes minor events where the duration is less than min_duration_days
# Sensitivity analysis recommended to the definition of min_duration_days (Van Loon. 2024)
def calculate_annual_deficits_and_durations(discharge, threshold, area_km2, start_year=1980, end_year=2020, min_duration_days=10):

    # Ensure discharge and threshold are aligned
    discharge, threshold = discharge.align(threshold, join='inner')

    # Define analysis period
    analysis_mask = (discharge.index >= pd.Timestamp(f'{start_year}-10-01')) & \
                    (discharge.index <= pd.Timestamp(f'{end_year}-09-30'))

    # Apply analysis period
    discharge = discharge[analysis_mask]
    threshold = threshold[analysis_mask]

    # Assign hydrological years
    year = assign_year(discharge.index)

    # Calculate deficits where discharge < threshold
    deficit = (threshold - discharge).clip(lower=0)  # if there are values above the threshold, it is set to 0 (no deficit)

    # Calculate days below threshold
    days_below = (discharge < threshold).astype(int)

    # Handle missing data by setting deficits and days_below to NaN where either is NaN
    valid_mask = (~discharge.isna()) & (~threshold.isna())
    deficit = deficit.where(valid_mask, np.nan)
    days_below = days_below.where(valid_mask, np.nan)

    # Create a DataFrame 
    df = pd.DataFrame({
        'Deficit_mm': deficit,
        'Days_Below': days_below,
        'Year': year
    })

    # Filter out minor events (those with duration < min_duration_days)
    event_duration = df.groupby('Year')['Days_Below'].sum()
    valid_years = event_duration[event_duration >= min_duration_days].index
    df_filtered = df[df['Year'].isin(valid_years)]

    # Group by hydrological year, aggregate and rename
    annual_summary = df_filtered.groupby('Year').agg({
        'Deficit_mm': 'sum',
        'Days_Below': 'sum'
    }).reset_index()
    annual_summary.rename(columns={
        'Deficit_mm': 'Deficit_mm',
        'Days_Below': 'Days_Below_Threshold'
    }, inplace=True)

    return annual_summary

# Calculate additional metrics for annual low-flow data.

def calculate_additional_metrics(combined_annual):
    h_total_dur = combined_annual['Days_Below_Threshold_Human'].sum()
    n_total_dur = combined_annual['Days_Below_Threshold_Natural'].sum()
    h_total_def = combined_annual['Deficit_mm_Human'].sum() 
    n_total_def = combined_annual['Deficit_mm_Natural'].sum() 
    h_av_dur = combined_annual['Days_Below_Threshold_Human'].mean()
    n_av_dur = combined_annual['Days_Below_Threshold_Natural'].mean()
    h_av_def = combined_annual['Deficit_mm_Human'].mean() 
    n_av_def = combined_annual['Deficit_mm_Natural'].mean() 
    total_duration_diff = ((h_total_dur - n_total_dur) / n_total_dur) * 100 if n_total_dur > 0 else np.nan
    deficit_index_diff = ((h_total_def - n_total_def) / n_total_def) * 100 if n_total_def > 0 else np.nan

    return {
        "H_total_dur (days)": h_total_dur,
        "N_total_dur (days)": n_total_dur,
        "H_total_def (mm)": h_total_def,
        "N_total_def (mm)": n_total_def,
        "H_av_dur (days)": h_av_dur,
        "N_av_dur (days)": n_av_dur,
        "H_av_def (mm)": h_av_def,
        "N_av_def (mm)": n_av_def,
        "Total Duration Index Difference (%)": total_duration_diff,
        "Deficit Index Difference (%)": deficit_index_diff,
    }

def save_annual_data_to_txt(annual_data, gauge_id, data_type, output_dir):
    file_path = os.path.join(output_dir, f"{gauge_id}_{data_type}.txt")
    if data_type == 'volume':
        header = "Year\tDeficit_mm_Human\tDeficit_mm_Natural\n"
    elif data_type == 'duration':
        header = "Year\tDays_Below_Threshold_Human\tDays_Below_Threshold_Natural\n"
    else:
        raise ValueError("data_type must be either 'volume' or 'duration'")

    with open(file_path, 'w') as file:
        file.write(header)
        for _, row in annual_data.iterrows():
            if data_type == 'volume':
                line = f"{int(row['Year'])}\t{round(row['Deficit_mm_Human'], 2)}\t{round(row['Deficit_mm_Natural'], 2)}\n"
            elif data_type == 'duration':
                line = f"{int(row['Year'])}\t{int(row['Days_Below_Threshold_Human'])}\t{int(row['Days_Below_Threshold_Natural'])}\n"
            file.write(line)

    print(f"Saved {data_type} data for Gauge ID: {gauge_id} at {file_path}")

def save_metrics_to_txt(metrics, gauge_id, output_dir):
    file_path = os.path.join(output_dir, f"{gauge_id}_metrics.txt")
    with open(file_path, 'w') as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value:.2f}\n")
    print(f"Saved metrics for Gauge ID: {gauge_id} at {file_path}")


# Now looping through files and applying functions

# Load basin info
basin_info_df = load_basin_info(excel_file_path)

# Process each basin's data, for paired basins it can read more than one entry
for index, row in basin_info_df.iterrows():
    gauge_id = str(row['gauge_id'])
    paired_ids = str(row['paired_id']).split(';')
    paired_areas = [float(area) for area in str(row['paired_area']).split(';')]
    human_area_km2 = row['area'] if not pd.isna(row['area']) else row['area_calc']

    # Load discharge data
    file_path = os.path.join(discharge_dir, f'{gauge_id}.txt')
    try:
        data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    except Exception as e:
        continue

    # Prepare discharge data (Moving Average already applied)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d', errors='coerce')
    data = data.dropna(subset=['Date']).drop_duplicates(subset=['Date']).set_index('Date').sort_index()

    # Convert to mm/day
    data['Discharge_mm_per_day'] = discharge_m3s_to_mmday(data['Discharge'], human_area_km2)

    # Load threshold data
    try:
        thresholds = load_threshold_file(thresholds_folder, gauge_id)
    except FileNotFoundError as e:
        print(e)
        continue

    # Load p-factor data
    try:
        p_factor_data = load_p_factor_file(p_factor_dir, gauge_id)
    except FileNotFoundError as e:
        print(e)
        continue

    # Apply precipitation factor
    for paired_id, paired_area in zip(paired_ids, paired_areas):
        paired_file_path = os.path.join(discharge_dir, f'{paired_id}.txt')
        try:
            paired_data = pd.read_csv(paired_file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
        except Exception as e:
            print(f"Error loading paired data for {paired_id}: {e}")
            continue
        
        # Prepare paired discharge data
        paired_data['Date'] = pd.to_datetime(paired_data['Date'], format='%Y%m%d', errors='coerce')
        paired_data = paired_data.dropna(subset=['Date']).drop_duplicates(subset=['Date']).set_index('Date').sort_index()
        paired_data['Discharge_mm_per_day'] = discharge_m3s_to_mmday(paired_data['Discharge'], paired_area)

        # Apply p-factor to paired data
        adjusted_paired_discharge = apply_precipitation_factor(paired_data, p_factor_data)

        # Filter data for analysis period (1980-10-01 to 2020-09-30)
        start_year = 1980
        end_year = 2020
        analysis_start = pd.Timestamp(f'{start_year}-10-01')
        analysis_end = pd.Timestamp(f'{end_year}-09-30')

        filtered_data = data.loc[
            (data.index >= analysis_start) & 
            (data.index <= analysis_end), 
            'Discharge_mm_per_day'
        ]
        filtered_paired_data = adjusted_paired_discharge.loc[
            (adjusted_paired_discharge.index >= analysis_start) & 
            (adjusted_paired_discharge.index <= analysis_end)
        ]
        filtered_threshold = thresholds.loc[
            (thresholds.index >= analysis_start) & 
            (thresholds.index <= analysis_end)
        ]

        # Calculate Annual Deficits and Durations
        human_annual = calculate_annual_deficits_and_durations(
            filtered_data, filtered_threshold, human_area_km2, start_year, end_year
        )
        natural_annual = calculate_annual_deficits_and_durations(
            filtered_paired_data, filtered_threshold, paired_area, start_year, end_year
        )

        # Merge human and natural annual data 
        combined_annual = pd.merge(
            human_annual,
            natural_annual,
            on='Year',
            how='outer',
            suffixes=('_Human', '_Natural')).fillna(0)

        # Save Volume Data
        volume_data = combined_annual[['Year', 'Deficit_mm_Human', 'Deficit_mm_Natural']].copy()
        save_annual_data_to_txt(volume_data, gauge_id, 'volume', deficit_output_dir)

        # Save Duration Data
        duration_data = combined_annual[['Year', 'Days_Below_Threshold_Human', 'Days_Below_Threshold_Natural']].copy()
        save_annual_data_to_txt(duration_data, gauge_id, 'duration', deficit_output_dir)

        # Calculate Additional Metrics
        metrics = calculate_additional_metrics(combined_annual)
        save_metrics_to_txt(metrics, gauge_id, deficit_output_dir)

        print(f"Processed Gauge ID: {gauge_id}")

print("All gauges have been processed and the respective volume, duration, and metrics files have been saved.")
#%%

