

#%% Option 1: Calculates baseflow separation based in given alpha, beta and n passes - one by one
# Adapted from Micha's code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from scipy.stats import linregress

# Lyne-Hollick filter implementation
def applyLHfilter(Qtotal, alpha=0.0, beta=2, npass=0, initopt=0):
    # Create array for baseflow
    Qbase = Qtotal.copy()
    Qquick = np.zeros(np.size(Qtotal), dtype=float)

    if initopt == 0:
        Qquick[0] = Qtotal[0]  # Initialize Qquick with the first value of Qtotal

    # First forward pass
    for ii in np.arange(1, len(Qtotal)):
        Qquick[ii] = alpha * Qquick[ii - 1] + (1 + alpha) / beta * (Qtotal[ii] - Qtotal[ii - 1])
    
    # Calculate Qbase (baseflow)
    Qbase = np.where(Qquick > 0, Qtotal - Qquick, Qbase)

    # Sequence of backwards and forward passes
    for nn in np.arange(npass):
        Qquick[-1] = Qtotal[-1]  # Initialize the last value of Qquick for backward pass
        if initopt == 0:
            Qquick[-1] = Qbase[-1]

        # Loop backwards
        for ii in np.arange(len(Qtotal) - 1, 0, -1):
            Qquick[ii - 1] = alpha * Qquick[ii] + (1 + alpha) / beta * (Qbase[ii - 1] - Qbase[ii])
        Qbase = np.where(Qquick > 0, Qbase - Qquick, Qbase)

        # Do a forward pass again
        Qquick[0] = Qtotal[0]  # Set initial value again for consistency
        if initopt == 0:
            Qquick[0] = Qbase[0]

        for ii in np.arange(1, len(Qtotal)):
            Qquick[ii] = alpha * Qquick[ii - 1] + (1 + alpha) / beta * (Qbase[ii] - Qbase[ii - 1])
        Qbase = np.where(Qquick > 0, Qbase - Qquick, Qbase)

    return Qbase, Qtotal - Qbase  # Quickflow = Total Discharge - Baseflow

# Function to calculate recession constant from baseflow
def calculate_recession_constant(baseflow, dates):
    dates = pd.Series(dates)
    recession_periods = []
    start = None

    for i in range(1, len(baseflow)):
        if baseflow[i] < baseflow[i - 1]:  # Detect a decrease
            if start is None:
                start = i - 1  # Start of recession period
        else:
            if start is not None:  # End of a recession period
                if i - start > 1:  # Only consider if length > 1
                    recession_periods.append((start, i - 1))
                start = None

    # Calculate k for each recession period
    k_values = []
    for start, end in recession_periods:
        # Get the time (in days) and baseflow values for this period
        times = (dates[start:end+1] - dates[start]).dt.days
        log_baseflow = np.log(baseflow[start:end+1])

        # Linear regression to find slope, which is -k
        slope, _, _, _, _ = linregress(times, log_baseflow)
        k = -slope  # k is the negative of the slope
        k_values.append(k)

    # Return the mean recession constant k, or handle empty k_values case
    mean_k = np.mean(k_values) if k_values else None
    print(f"Mean recession constant (k): {mean_k:.4f}" if mean_k is not None else "No recession periods found.")

    return mean_k, k_values

# Function to process and plot data for a specific file based on user input
def process_file(gauge_id, start_date, end_date, directory_path, alpha=0.92, beta=2, npass=2):
    # Construct the file path from the input gauge ID
    file_path = os.path.join(directory_path, gauge_id + '.txt')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{gauge_id}.txt' not found in the directory.")
        return

    # Load the full dataset to calculate the BFI for the entire period
    data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    data = data[data['Discharge'].notna()].copy()  # Keep rows with zero discharge but remove rows with missing values

    # Apply the Lyne-Hollick filter to the entire dataset to calculate BFI
    Qtotal_full = data['Discharge'].values
    Qbase_full, Qquick_full = applyLHfilter(Qtotal_full, alpha=alpha, beta=beta, npass=npass, initopt=0)

    # Calculate the BFI for the entire period
    BFI = np.sum(Qbase_full) / np.sum(Qtotal_full)
    print(f"Baseflow Index (BFI) for the entire period: {BFI:.2f}")

    # Add baseflow and quickflow to the DataFrame for the full period
    data['Baseflow'] = Qbase_full
    data['Quickflow'] = Qquick_full

    # Calculate the recession constant for the entire period
    mean_k, k_values = calculate_recession_constant(data['Baseflow'], data['Date'])

    # Filter data based on the specified date range for plotting
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(filtered_data['Date'], filtered_data['Discharge'], label='Total Discharge', color='black')
    plt.plot(filtered_data['Date'], filtered_data['Baseflow'], label='Baseflow', linestyle='--', color='green')
    plt.title(f'Station: {gauge_id} (BFI: {BFI:.2f}, Mean k: {mean_k:.4f})' if mean_k is not None else f'Station: {gauge_id} (BFI: {BFI:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Discharge (m³/s)')
    plt.legend()
    plt.xlim(filtered_data['Date'].min(), filtered_data['Date'].max())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)) 
    plt.tight_layout()
    plt.show()

    return filtered_data

directory_path = r'D:\My Documents\LoFlowMaas\Discharge\All'
gauge_id = '6421500'  # Replace with desired gauge_id
start_date = '2010-01-01'
end_date = '2010-12-31' # Just to inspect closely

# Process the file and plot with recession curve calculation
processed_data = process_file(gauge_id, start_date, end_date, directory_path, alpha=0.95, beta=2, npass=2)






#%% Option 2: Calculates baseflow separation given BFI (EStreams) for all basins and also volumes (baseflow and total)

# It also generates a plot for a zoomed in period. If necessary to change the BFI target, change in the excel file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from scipy.stats import linregress

def apply_LH_filter(Qtotal, alpha, beta=2, npass=2, initopt=0):
    Qquick = np.zeros_like(Qtotal, dtype=float)
    Qbase = Qtotal.copy()

    if initopt == 0:
        Qquick[0] = Qtotal[0]

    # First forward pass
    for ii in range(1, len(Qtotal)):
        Qquick[ii] = alpha * Qquick[ii - 1] + ((1 + alpha) / beta) * (Qtotal[ii] - Qtotal[ii - 1])

    Qbase = np.where(Qquick > 0, Qtotal - Qquick, Qbase)

    # Sequence of backward and forward passes
    for _ in range(npass):
        Qquick[-1] = Qtotal[-1] if initopt != 0 else Qbase[-1]

        # Backward pass
        for ii in range(len(Qtotal) - 1, 0, -1):
            Qquick[ii - 1] = alpha * Qquick[ii] + ((1 + alpha) / beta) * (Qbase[ii - 1] - Qbase[ii])
        Qbase = np.where(Qquick > 0, Qbase - Qquick, Qbase)

        # Forward pass
        Qquick[0] = Qtotal[0] if initopt != 0 else Qbase[0]

        for ii in range(1, len(Qtotal)):
            Qquick[ii] = alpha * Qquick[ii - 1] + ((1 + alpha) / beta) * (Qbase[ii] - Qbase[ii - 1])
        Qbase = np.where(Qquick > 0, Qbase - Qquick, Qbase)

    return Qbase, Qtotal - Qbase  # Quickflow = Total Discharge - Baseflow

def calculate_BFI(Qtotal, Qbase):
    return np.sum(Qbase) / np.sum(Qtotal)

def optimize_alpha(target_BFI, Qtotal, beta=2, npass=2):
    def objective(alpha):
        alpha = alpha.item()
        Qbase, _ = apply_LH_filter(Qtotal, alpha=alpha, beta=beta, npass=npass)
        calculated_BFI = calculate_BFI(Qtotal, Qbase)
        return (calculated_BFI - target_BFI) ** 2

    initial_alpha = 0.92  # Initial guess for alpha
    result = minimize(objective, initial_alpha, bounds=[(0, 1)], method='L-BFGS-B')
    return result.x[0] if result.success else initial_alpha

def calculate_recession_constant(baseflow, dates):
    baseflow = np.array(baseflow)
    dates = pd.to_datetime(dates)

    recession_periods = []
    start = None

    for i in range(1, len(baseflow)):
        if baseflow[i] < baseflow[i - 1]:
            if start is None:
                start = i - 1
        else:
            if start is not None:
                if i - start > 10:  # Minimum time to consider a recession
                    recession_periods.append((start, i - 1))
                start = None

    # Check if last period is a recession period
    if start is not None and (len(baseflow) - start) > 10: # Minimum time to consider a recession (used to calculate recession time - it is very sensitive to this value)
        recession_periods.append((start, len(baseflow) - 1))

    k_values = []
    for start, end in recession_periods:
        try:
            times = (dates.iloc[start:end+1] - dates.iloc[start]).dt.days.values
            log_baseflow = np.log(baseflow[start:end+1])

            # Ensure log_baseflow has no invalid values
            valid_indices = np.isfinite(log_baseflow) & (log_baseflow > -np.inf)
            if not valid_indices.any():
                continue

            times = times[valid_indices]
            log_baseflow = log_baseflow[valid_indices]

            if len(times) < 2:
                continue  # Need at least two points to perform linear regression

            slope, _, _, _, _ = linregress(times, log_baseflow)
            k = -slope
            k_values.append(k)
        except Exception as e:
            print(f"Error processing recession period {start}-{end}: {e}")

    if k_values:
        mean_k = np.mean(k_values)
    else:
        print("No valid recession periods found.")
        mean_k = None

    return mean_k, k_values

def calculate_annual_volumes(data, area):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Volume_Baseflow'] = data['Baseflow'] * 86400  # Daily volume in m³
    data['Volume_Total'] = data['Discharge'] * 86400

    data['Year'] = data['Date'].dt.year

    data = data[data['Discharge'] > 0]

    annual_volumes = data.groupby('Year').agg({
        'Volume_Baseflow': 'sum',
        'Volume_Total': 'sum'
    }).dropna()

    valid_years = []
    for year, group in data.groupby('Year'):
        days_in_year = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31').size
        num_days = group['Date'].nunique()
        if num_days / days_in_year >= 0.9:  # At least 90% data available
            valid_years.append(year)

    valid_annual_volumes = annual_volumes.loc[valid_years]

    avg_baseflow_volume_m3 = valid_annual_volumes['Volume_Baseflow'].mean()
    avg_total_volume_m3 = valid_annual_volumes['Volume_Total'].mean()

    area_m2 = area * 1e6  # Convert km² to m²
    avg_baseflow_volume_mm = (avg_baseflow_volume_m3 / area_m2) * 1000  # Convert m to mm
    avg_total_volume_mm = (avg_total_volume_m3 / area_m2) * 1000

    return avg_baseflow_volume_m3, avg_total_volume_m3, avg_baseflow_volume_mm, avg_total_volume_mm

def process_all_basins(directory_path, excel_file_path, output_folder, start_date, end_date, beta=2, npass=2):
    info_df = pd.read_excel(excel_file_path, dtype={'gauge_id': str})
    os.makedirs(output_folder, exist_ok=True)

    summary_file_path = os.path.join(output_folder, "summary.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.write("Gauge ID\tBasin ID\tGauge name\tTarget BFI\tOptimized alpha\tMean k\tRecession time (days)\tAvg Baseflow Volume (m³)\tAvg Total Volume (m³)\tAvg Baseflow Volume (mm)\tAvg Total Volume (mm)\n")

        for _, row in info_df.iterrows():
            gauge_id = row['gauge_id']
            basin_id = row['basin_id']
            target_BFI = row['Target_BFI']
            gauge_name = row['gauge_name']
            area = row['area']  # Area in km²

            file_path = os.path.join(directory_path, f"{gauge_id}.txt")
            output_plot = os.path.join(output_folder, f"{gauge_id}_plot.png")

            if not os.path.exists(file_path):
                print(f"File for gauge_id '{gauge_id}' not found. Skipping...")
                continue

            data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
            data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
            data = data[data['Discharge'] > 0].copy()

            Qtotal_full = data['Discharge'].values
            optimal_alpha = optimize_alpha(target_BFI, Qtotal_full, beta=beta, npass=npass)

            Qbase_full, Qquick_full = apply_LH_filter(Qtotal_full, alpha=optimal_alpha, beta=beta, npass=npass, initopt=0)
            data['Baseflow'] = Qbase_full
            data['Quickflow'] = Qquick_full
            data['Baseflow_smoothed'] = data['Baseflow'].rolling(window=3, center=True).mean()

            mean_k, _ = calculate_recession_constant(data['Baseflow_smoothed'].dropna(), data['Date'].loc[data['Baseflow_smoothed'].dropna().index])
            recession_time = 1 / mean_k if mean_k is not None and mean_k != 0 else None

            avg_baseflow_vol_m3, avg_total_vol_m3, avg_baseflow_vol_mm, avg_total_vol_mm = calculate_annual_volumes(data, area)

            # Prepare strings for mean_k and recession_time
            mean_k_str = f"{mean_k:.4f}" if mean_k is not None else "N/A"
            recession_time_str = f"{recession_time:.2f}" if recession_time is not None else "N/A"

            summary_file.write(f"{gauge_id}\t{basin_id}\t{gauge_name}\t{target_BFI:.2f}\t{optimal_alpha:.4f}\t{mean_k_str}\t{recession_time_str}\t{avg_baseflow_vol_m3:.2f}\t{avg_total_vol_m3:.2f}\t{avg_baseflow_vol_mm:.2f}\t{avg_total_vol_mm:.2f}\n")

            filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

            plt.figure(figsize=(12, 8))
            plt.plot(filtered_data['Date'], filtered_data['Discharge'], label='Total Discharge', color='black')
            plt.plot(filtered_data['Date'], filtered_data['Baseflow_smoothed'], label='Baseflow', linestyle='--', color='green')
            plt.title(f"Station: {gauge_id} ({gauge_name})\nBasin ID: {basin_id}, Target BFI: {target_BFI:.2f}, Optimized Alpha: {optimal_alpha:.4f}, Mean k: {mean_k_str}, Recession Time: {recession_time_str} days")
            plt.xlabel('Date')
            plt.ylabel('Discharge (m³/s)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_plot)
            plt.close()

directory_path = r'D:\My Documents\LoFlowMaas\Discharge\interpolated'
excel_file_path = r'D:\My Documents\LoFlowMaas\Discharge\Info_EStreams.xlsx'
output_folder = r'D:\My Documents\LoFlowMaas\Discharge\recession'
start_date = '2010-01-01'  # Just for zooming in the plot
end_date = '2010-12-31'

process_all_basins(directory_path, excel_file_path, output_folder, start_date, end_date, beta=2, npass=2)

#%%
