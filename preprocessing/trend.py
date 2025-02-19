#%% Developed by Deborah, 15h of November 2024

# This code performs trend analysis, visualization and computes trend indices

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pymannkendall as mk

# Function to perform trend analysis for all files in the directory
def trend_analysis_all_files(directory_path, output_dir, summary_file, start_year=1980, end_year=2020, min_data_ratio=0.98):
    # Get all files in the directory
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare summary DataFrame
    summary_data = []

    for file_name in all_files:
        gauge_id = file_name.split('.')[0]  # Extract gauge ID from file name
        file_path = os.path.join(directory_path, file_name)

        # Load discharge data
        try:
            data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Convert 'Date' column to datetime format, set as index, sort, and drop duplicates
        data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d', errors='coerce')
        data = data.dropna(subset=['Date']).drop_duplicates(subset=['Date']).set_index('Date').sort_index()

        # Filter data between the specified years
        data = data[(data.index.year >= start_year) & (data.index.year <= end_year)]
        if data.empty:
            print(f"No data available for {gauge_id} in the specified period ({start_year}-{end_year}).")
            continue

        # Aggregate data annually (mean discharge per year), ensuring at least 98% data availability per year
        annual_data = data['Discharge'].resample('Y').apply(
            lambda x: x.mean() if len(x.dropna()) / x.size >= min_data_ratio else np.nan
        )
        annual_data = annual_data.dropna()  # Remove years with insufficient data

        # Perform linear regression on the annual data
        x = np.arange(len(annual_data))  # Time in years as an index
        y = annual_data.values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        # Calculate trend line
        trend_line = slope * x + intercept

        # Perform Mann-Kendall test
        mk_result = mk.original_test(annual_data)

        # Determine significance and format title
        is_significant = "Significant" if getattr(mk_result, 'h', False) else "Not Significant"
        title = (
            f'Annual Trend Analysis for Gauge {gauge_id} ({start_year}-{end_year})\n'
            f'{is_significant} (MK p-value={mk_result.p:.4f})'
        )

        # Save plot
        plt.figure(figsize=(12, 6))
        plt.plot(annual_data.index.year, annual_data, label='Annual Mean Discharge', color='blue', marker='o')
        plt.plot(annual_data.index.year, trend_line, label=f'Trend (Slope={slope:.4f} m³/s/year)', color='red', linestyle='--')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Discharge (m³/s)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{gauge_id}_annual_trend.png')
        plt.savefig(plot_path)
        plt.close()

        # Append summary data
        summary_data.append({
            'Gauge ID': gauge_id,
            'Slope (m³/s/year)': slope,
            'Intercept (m³/s)': intercept,
            'R-squared': r_value**2,
            'P-value': p_value,
            'Standard Error': std_err,
            'Mann-Kendall Trend': 'Increasing' if mk_result.trend == 'increasing' else 'Decreasing' if mk_result.trend == 'decreasing' else 'No trend',
            'Mann-Kendall S': getattr(mk_result, 'S', 'Not Available'),
            'Mann-Kendall Tau': getattr(mk_result, 'Tau', 'Not Available'),
            'Mann-Kendall P-value': getattr(mk_result, 'p', 'Not Available'),
            'Significant': is_significant
        })

    # Save summary to Excel
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(summary_file, index=False)

# Example usage
directory_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated' 
output_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\trend'
summary_file = os.path.join(output_dir, 'trend_summary.xlsx')

# Perform trend analysis for all files
trend_analysis_all_files(directory_path, output_dir, summary_file, start_year=1980, end_year=2020, min_data_ratio=0.98)

#%%
