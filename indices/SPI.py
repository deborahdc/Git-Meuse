#%% Developed by Deborah, October 2024
# For a given gauge_id, gets the basin_id and gets the met data and calculates the SPI

from scipy.stats import norm, gamma
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to retrieve basin and gauge information from Excel based on gauge_id
def get_basin_info_from_gauge(gauge_id, excel_file_path):
    try:
        print("Loading gauge information from Excel...")
        gauge_info_df = pd.read_excel(excel_file_path, dtype={'gauge_id': str})
        gauge_id = str(gauge_id)
        row = gauge_info_df[gauge_info_df['gauge_id'] == gauge_id]
        if not row.empty:
            basin_id = row['basin_id'].values[0]
            gauge_name = row['gauge_name'].values[0]
            print(f"Found gauge information for ID {gauge_id}: Basin ID = {basin_id}, Gauge Name = {gauge_name}")
            return basin_id, gauge_name
    except Exception as e:
        print(f"Error reading gauge information: {e}")
    return None, None

# Function to retrieve meteorological file based on basin_id
def get_meteorological_file(basin_id, meteorology_directory):
    print("Searching for meteorological file...")
    for file in os.listdir(meteorology_directory):
        if basin_id in file:
            print(f"Found meteorological file: {file}")
            return os.path.join(meteorology_directory, file)
    print(f"No meteorology file found for basin ID: {basin_id}")
    return None

# SPI Calculation Function 
def calculate_spi(precip_data, spi_scale):
    print("Calculating SPI...")
    precip_data = precip_data.resample('MS').sum().replace(0, 1e-6)
    precip_data = precip_data.dropna() 
    precip_rolling = precip_data.rolling(window=spi_scale).sum().dropna()
    spi_series = pd.Series(index=precip_rolling.index, dtype=float)

    for month in range(1, 13):
        month_data = precip_rolling[precip_rolling.index.month == month].dropna()
        if not month_data.empty:
            try:
                shape, loc, scale = gamma.fit(month_data, floc=0)
                cdf = gamma.cdf(month_data, shape, loc, scale)
                spi_series[month_data.index] = norm.ppf(cdf)
            except ValueError:
                print(f"Failed to fit distribution for month {month}. Skipping.")
                continue
    print("SPI calculation complete.")
    return spi_series

# Color mapping function
def get_fixed_color(value):
    if value >= 2:
        return "#08306b"
    elif 1 <= value < 2:
        return "#2171b5"
    elif 0 < value < 1:
        return "#6baed6"
    elif -1 <= value < 0:
        return "#fd8d3c"
    elif -2 <= value < -1:
        return "#e6550d"
    elif value < -2:
        return "#a63603"
    else:
        return "gray"

# Main function to generate the SPI plot
def generate_spi_for_gauge(gauge_id, excel_file_path, meteorology_directory, start_date, end_date, spi_scale):
    print("Starting SPI plot generation...")
    basin_id, gauge_name = get_basin_info_from_gauge(gauge_id, excel_file_path)
    if not basin_id or not gauge_name:
        print("Basin or gauge information not found. Exiting.")
        return

    meteorology_file = get_meteorological_file(basin_id, meteorology_directory)
    if not meteorology_file:
        print("Meteorology file not found. Exiting.")
        return

    try:
        meteorology_data = pd.read_csv(meteorology_file)
        meteorology_data['date'] = pd.to_datetime(meteorology_data['date'], format='%Y-%m-%d')
        meteorology_data.set_index('date', inplace=True)
        precip_data = meteorology_data['p_mean'][start_date:end_date]

        # Calculate SPI
        spi_series = calculate_spi(precip_data, spi_scale)
        spi_series_filtered = spi_series[start_date:end_date]

        # Apply fixed color mapping
        colors = [get_fixed_color(spi) for spi in spi_series_filtered]

        # Plotting SPI values
        plt.figure(figsize=(14, 6))
        plt.bar(spi_series_filtered.index, spi_series_filtered, color=colors, width=20)
        plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
        plt.axhline(1, color="#6baed6", linestyle='-', linewidth=0.8)
        plt.axhline(2, color="#2171b5", linestyle='-', linewidth=0.8)
        plt.axhline(3, color="#08306b", linestyle='-', linewidth=0.8)
        plt.axhline(-1, color="#fd8d3c", linestyle='-', linewidth=0.8)
        plt.axhline(-2, color="#e6550d", linestyle='-', linewidth=0.8)
        plt.axhline(-3, color="#a63603", linestyle='-', linewidth=0.8)

        # Set x-axis limits based on the actual data range
        plt.xlim(spi_series_filtered.index.min(), spi_series_filtered.index.max())

        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
        plt.gca().xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.title(f'{spi_scale}-Month SPI for {gauge_name}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('SPI', fontsize=12)
        plt.ylim(-4, 4)
        plt.tight_layout()
        plt.show()
        print("SPI plot complete.")
    except Exception as e:
        print(f"Error during SPI generation: {e}")

# Example usage
gauge_id = 'B720000002'
excel_file_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\Info_EStreams.xlsx'
meteorology_directory = r'C:\Users\ddo001\Documents\LoFloMeuse\EStreams_data\EStreams\meteorology'

start_date = '1980-01-01'
end_date = '2020-12-31'
spi_scale = 12 #can change to other SPI 

generate_spi_for_gauge(gauge_id, excel_file_path, meteorology_directory, start_date, end_date, spi_scale)
#%%
