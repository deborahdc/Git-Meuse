#%% Developed by Deborah, October 2024
from scipy.stats import norm, gamma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Function to load discharge data
def load_discharge_data(gauge_id, directory_path):
    file_path = os.path.join(directory_path, f'{gauge_id}.txt')
    data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d', errors='coerce')
    data = data.dropna(subset=['Date'])
    data.set_index('Date', inplace=True)
    return data['Discharge']

# Function to get gauge information from the Excel file
def get_gauge_info(gauge_id, excel_file_path):
    gauge_info = pd.read_excel(excel_file_path)
    gauge_info['gauge_id'] = gauge_info['gauge_id'].astype(str)
    
    row = gauge_info[gauge_info['gauge_id'] == gauge_id]
    if not row.empty:
        station_name = row['gauge_name'].values[0]
        return station_name
    else:
        print(f"Gauge ID {gauge_id} not found in the Excel file.")
        return None

# Function to filter out incomplete years based on daily data
def filter_complete_years(discharge_data):
    # Resample to daily frequency and count available data for each day
    daily_counts = discharge_data.resample('D').count()
    yearly_counts = daily_counts.resample('Y').sum()

    # Define expected days per year, adjusting for leap years
    expected_days_per_year = yearly_counts.index.to_series().apply(
        lambda x: 366 if x.is_leap_year else 365
    )

    # Identify complete years where daily counts match expected days
    complete_years = yearly_counts == expected_days_per_year
    years_with_complete_data = complete_years[complete_years].index.year.tolist()
    discharge_data = discharge_data[discharge_data.index.year.isin(years_with_complete_data)]
    return discharge_data

# Function to calculate SSI with handling for missing data
def calculate_ssi(discharge_data, ssi_scale):
    # Resample to monthly sums - calculates the totals for each month
    discharge_data = discharge_data.resample('MS').sum()
    
    # Replace zero discharge with a very small value to avoid issues with gamma fitting
    discharge_data = discharge_data.replace(0, 1e-6)
    
    # Drop missing values before calculating the rolling sum
    discharge_data = discharge_data.dropna()
    
    # Calculate the rolling sum based on the specified SSI scale
    discharge_rolling = discharge_data.rolling(window=ssi_scale).sum().dropna()

    # Initialize an empty series to store SSI values
    ssi_series = pd.Series(index=discharge_rolling.index, dtype=float)

    for month in range(1, 13):
        # Filter data for the specific month
        month_data = discharge_rolling[discharge_rolling.index.month == month]
        
        # Check if there is enough data to fit the distribution
        if len(month_data) > 0:
            try:
                # Fit the gamma distribution to the monthly data
                shape, loc, scale = gamma.fit(month_data, floc=0)
                
                # Compute the cumulative distribution function (CDF)
                cdf = gamma.cdf(month_data, shape, loc, scale)
                
                # Convert the CDF to a standard normal distribution
                ssi_series[month_data.index] = norm.ppf(cdf)
            except ValueError:
                continue

    return ssi_series

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

# Main function to generate the SSI plot with the gauge name in the title
def generate_ssi_plot(gauge_id, discharge_data, start_date, end_date, ssi_scale, excel_file_path):
    # Get gauge name
    station_name = get_gauge_info(gauge_id, excel_file_path)
    
    # Filter out incomplete years from the data
    discharge_data = filter_complete_years(discharge_data)

    # Calculate SSI based on the provided scale
    ssi_series = calculate_ssi(discharge_data, ssi_scale)

    # Filter SSI data for the selected display date range
    ssi_series_filtered = ssi_series[start_date:end_date]
    colors = [get_fixed_color(ssi) for ssi in ssi_series_filtered]

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.bar(ssi_series_filtered.index, ssi_series_filtered, color=colors, width=20)
    plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
    plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    plt.axhline(1, color="#6baed6", linestyle='-', linewidth=0.8)
    plt.axhline(2, color="#2171b5", linestyle='-', linewidth=0.8)
    plt.axhline(3, color="#08306b", linestyle='-', linewidth=0.8)
    plt.axhline(-1, color="#fd8d3c", linestyle='-', linewidth=0.8)
    plt.axhline(-2, color="#e6550d", linestyle='-', linewidth=0.8)
    plt.axhline(-3, color="#a63603", linestyle='-', linewidth=0.8)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gca().xaxis.set_minor_locator(mdates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Title with gauge name
    plt.title(f'{ssi_scale}-Month SSI for {station_name}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('SSI', fontsize=12)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.show()

# Example usage
gauge_id = 'B720000002'
directory_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\All'
excel_file_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\Info_EStreams.xlsx'
start_date = '1980-01-01'
end_date = '2020-12-31'
ssi_scale = 12

# Load discharge data and generate the SSI plot
discharge_data = load_discharge_data(gauge_id, directory_path)
generate_ssi_plot(gauge_id, discharge_data, start_date, end_date, ssi_scale, excel_file_path)

#%%
