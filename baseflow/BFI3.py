#%% CONVERTS TO BFI3.0 accepted format
import os

# Directories for input and output files
input_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\All"
output_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\new"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to reformat files
def reformat_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("Date\tDischarge\n")  # Write the single header
        for i, line in enumerate(infile):
            if i == 0 and "Date" in line:  # Skip the original header if present
                continue
            parts = line.strip().split()
            if len(parts) == 2:  # Ensure valid lines with exactly two parts
                raw_date, discharge = parts
                # Convert date from YYYYMMDD to YYYY-MM-DD
                formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                outfile.write(f"{formatted_date}\t{discharge}\n")

# Process all files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.txt'):  # Only process .txt files
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        reformat_file(input_file, output_file)

print("Files reformatted and saved to:", output_dir)

#%%  Processes the outputs, calculates mean BFI (whole period), the 10 longest recession curves (duration) and computes the k and T for each
import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Paths
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\output"
output_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\recession"
summary_file = os.path.join(output_folder, "summary.txt")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to process a single basin file
def process_basin(file_path):
    # Load the data
    df = pd.read_csv(file_path, sep="\t")
    
    # Ensure column names are consistent
    df.columns = [col.strip() for col in df.columns]
    
    # Clean and preprocess columns
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce')  # Convert to datetime
    df['Discharge'] = pd.to_numeric(df['Discharge'], errors='coerce')  # Convert to float
    df['BFI Index'] = pd.to_numeric(df['BFI Index'], errors='coerce')  # Convert to float
    
    # Drop rows with invalid data
    df = df.dropna(subset=['Date', 'Discharge', 'BFI Index'])
    
    # Filter data within the year range 1980-2020
    df = df[(df['Date'].dt.year >= 1980) & (df['Date'].dt.year <= 2020)]
    if df.empty:
        return "No valid data in file after filtering by year range (1980-2020).\n", None, None, None

    # Initialize variables
    results = []
    current_recession = []
    tolerance = 0.05  # Allow up to 5% increase in discharge
    max_increase_days = 3  # Allow up to 3 consecutive days of increase
    max_bfi_below_09_days = 3  # Allow BFI below 0.9 for up to 3 days
    rolling_window = 5  # Rolling window to detect overall trends

    # Loop through data to identify recession periods
    for i in range(1, len(df)):
        discharge_change = (df['Discharge'].iloc[i] - df['Discharge'].iloc[i - 1]) / df['Discharge'].iloc[i - 1]

        # Detect overall trend using rolling mean
        if i >= rolling_window:
            recent_trend = np.polyfit(
                range(rolling_window), df['Discharge'].iloc[i - rolling_window + 1 : i + 1], 1
            )[0]  # Slope of the trend
        else:
            recent_trend = -1  # Assume downward trend initially

        # General downward trend and BFI criteria
        if recent_trend < 0 or df['BFI Index'].iloc[i] >= 0.9 or (
            df['BFI Index'].iloc[i] >= 0.7 and len(current_recession) >= 3
        ):
            current_recession.append(i)
        elif discharge_change > tolerance:
            if len(current_recession) >= max_increase_days:  # Too many increases
                results.append(record_recession(df, current_recession))
                current_recession = []
        else:
            # Reset if conditions aren't met
            if len(current_recession) > 5:  # Valid recession
                results.append(record_recession(df, current_recession))
            current_recession = []

    # Add the last recession if valid
    if len(current_recession) > 5:
        results.append(record_recession(df, current_recession))
    
    # Convert results to DataFrame
    if not results:
        return "No valid recession periods found.\n", None, None, None
    results_df = pd.DataFrame(results)
    
    # Filter recessions with R² >= 0.7
    results_df = results_df[results_df['R²'] >= 0.7]
    if results_df.empty:
        return "No valid recession periods with R² >= 0.7 found.\n", None, None, None

    # Select the 10 longest recession periods
    longest_recessions = results_df.nlargest(10, "Duration")

    # Prepare the output
    output = f"Mean BFI: {df['BFI Index'].mean():.2f}\n\n"
    output += "10 Longest Recessions (R² >= 0.7):\n"
    for _, row in longest_recessions.iterrows():
        output += (
            f"Start: {row['Start'].strftime('%Y-%m-%d')}, "
            f"End: {row['End'].strftime('%Y-%m-%d')}, Duration: {row['Duration']} days, "
            f"Mean BFI: {row['Mean BFI']:.2f}, k: {row['k']:.4f}, T: {row['T']:.2f} days, "
            f"R²: {row['R²']:.2f}\n"
        )
    
    # Calculate averages
    avg_k = longest_recessions['k'].mean()
    avg_T = longest_recessions['T'].mean()
    avg_duration = longest_recessions['Duration'].mean()
    avg_r_squared = longest_recessions['R²'].mean()
    output += f"\nAverage Duration: {avg_duration:.2f} days\n"
    output += f"Average Recession Constant (k): {avg_k:.4f} days⁻¹\n"
    output += f"Average Time Constant (T): {avg_T:.2f} days\n"
    output += f"Average R²: {avg_r_squared:.2f}\n"

    return output, avg_k, avg_T, avg_r_squared

# Helper function to record recession details
def record_recession(df, recession_indices):
    start_idx = recession_indices[0]
    end_idx = recession_indices[-1]
    duration = (df['Date'].iloc[end_idx] - df['Date'].iloc[start_idx]).days
    q_start = df['Discharge'].iloc[start_idx]
    q_end = df['Discharge'].iloc[end_idx]
    ln_q = np.log(df['Discharge'].iloc[recession_indices])
    days = np.arange(len(ln_q))
    slope, intercept, r_value, _, _ = linregress(days, ln_q)
    k = -slope
    T = 1 / k if k and k > 0 else None
    r_squared = r_value**2
    return {
        "Start": df['Date'].iloc[start_idx],
        "End": df['Date'].iloc[end_idx],
        "Duration": duration,
        "Mean BFI": df['BFI Index'].iloc[start_idx:end_idx + 1].mean(),
        "k": k,
        "T": T,
        "R²": r_squared,
    }

# Initialize summary data
summary_data = []

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_folder, file_name)
        basin_code = os.path.splitext(file_name)[0]
        
        # Process the basin file
        output, avg_k, avg_T, avg_r_squared = process_basin(file_path)
        
        # Save the results to a text file
        output_file = os.path.join(output_folder, f"{basin_code}.txt")
        with open(output_file, "w") as f:
            f.write(output)
        
        # Append summary data
        if avg_k is not None and avg_T is not None:
            summary_data.append((basin_code, avg_k, avg_T, avg_r_squared))

# Write the summary to a single file
with open(summary_file, "w") as f:
    f.write(f"{'Station':<20}{'Avg k (days⁻¹)':<20}{'Avg T (days)':<20}{'Avg R²':<10}\n")
    f.write("-" * 70 + "\n")
    for station, avg_k, avg_T, avg_r_squared in summary_data:
        f.write(f"{station:<20}{avg_k:<20.4f}{avg_T:<20.2f}{avg_r_squared:<10.2f}\n")

print(f"Summary saved to {summary_file}.")


#%% Calculates baseflow volumes
import os
import pandas as pd

# Paths
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\output"
output_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\baseflow_volumes"
summary_file = os.path.join(output_folder, "summary.txt")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Constants
MONTHS_WINTER = [12, 1, 2]  # December (previous year), January, February
MONTHS_SPRING = [3, 4, 5]  # March, April, May
MONTHS_SUMMER = [6, 7, 8]  # June, July, August
MONTHS_AUTUMN = [9, 10, 11]  # September, October, November

DAYS_IN_SEASON = 92  # Approximate days in each season
MISSING_DATA_THRESHOLD = 0.05  # 5% missing data threshold
YEAR_START = 1980  # Start year for analysis
YEAR_END = 2020  # End year for analysis

# Function to calculate baseflow volume
def calculate_baseflow_volume(file_path, basin_code):
    # Load the data
    df = pd.read_csv(file_path, sep="\t")
    
    # Ensure column names are consistent
    df.columns = [col.strip() for col in df.columns]
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce')
    
    # Convert numeric columns
    df['Baseflow'] = pd.to_numeric(df['Baseflow'], errors='coerce')
    df['BFI Index'] = pd.to_numeric(df['BFI Index'], errors='coerce')
    
    # Drop rows with invalid data
    df = df.dropna(subset=['Date', 'Baseflow', 'BFI Index'])
    if df.empty:
        return f"No valid data found for basin {basin_code}.\n", None

    # Add year and month columns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Adjust December to belong to the next year's winter
    df.loc[df['Month'] == 12, 'HydroYear'] = df['Year'] + 1
    df['HydroYear'].fillna(df['Year'], inplace=True)
    
    # Filter data to only include years 1980–2020
    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    if df.empty:
        return f"No valid data in the year range {YEAR_START}-{YEAR_END} for basin {basin_code}.\n", None
    
    # Initialize results
    seasonal_volumes = {"Winter": [], "Spring": [], "Summer": [], "Autumn": []}
    excluded_years = []
    valid_bfi = []

    # Loop through each year
    for year, year_data in df.groupby('Year'):
        # Filter data for each season
        season_data = {
            "Winter": df[(df['HydroYear'] == year) & (df['Month'].isin(MONTHS_WINTER))],
            "Spring": year_data[year_data['Month'].isin(MONTHS_SPRING)],
            "Summer": year_data[year_data['Month'].isin(MONTHS_SUMMER)],
            "Autumn": year_data[year_data['Month'].isin(MONTHS_AUTUMN)]
        }
        
        # Check missing data for each season
        skip_year = False
        for season, data in season_data.items():
            if len(data) < (1 - MISSING_DATA_THRESHOLD) * DAYS_IN_SEASON:
                excluded_years.append(year)
                skip_year = True
                break
        if skip_year:
            continue  # Skip years with more than 5% missing data in any season
        
        # Calculate baseflow volume for each season [MCM]
        for season, data in season_data.items():
            seasonal_volumes[season].append(data['Baseflow'].sum() * 86400 / 1e6)
        
        # Store BFI Index
        valid_bfi.append(year_data['BFI Index'].mean())
    
    # Calculate averages for summary
    avg_volumes = {season: sum(volumes) / len(volumes) if volumes else None for season, volumes in seasonal_volumes.items()}
    avg_bfi = sum(valid_bfi) / len(valid_bfi) if valid_bfi else None
    
    # Generate output text for the basin file
    output = f"Basin: {basin_code}\n"
    output += f"Total Years Processed: {len(seasonal_volumes['Winter'])}\n"
    output += f"Excluded Years (Seasonal Gaps): {', '.join(map(str, excluded_years)) if excluded_years else 'None'}\n\n"
    output += "Baseflow Volumes (MCM):\n"
    output += "Year\tSeason\tVolume (MCM)\n"
    output += "-" * 30 + "\n"
    
    for year in df['Year'].unique():
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            if len(seasonal_volumes[season]) == len(df['Year'].unique()):
                output += f"{year}\t{season}\t{seasonal_volumes[season][df['Year'].unique().tolist().index(year)]:.2f}\n"
    
    return output, (basin_code, avg_volumes["Winter"], avg_volumes["Spring"], avg_volumes["Summer"], avg_volumes["Autumn"], avg_bfi)

# Process all files and write to individual text files
summary_data = []
for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_folder, file_name)
        basin_code = os.path.splitext(file_name)[0].replace("Out_", "")  # Remove "Out_" prefix
        
        # Calculate baseflow volumes for the file
        output, summary_entry = calculate_baseflow_volume(file_path, basin_code)
        
        # Save the results to a text file
        output_file = os.path.join(output_folder, f"{basin_code}_baseflow.txt")
        with open(output_file, "w") as f:
            f.write(output)
        
        print(f"Processed basin {basin_code}. Results saved to {output_file}.")
        
        # Append summary data
        if summary_entry:
            summary_data.append(summary_entry)

# Write the summary to a single file
with open(summary_file, "w") as f:
    f.write(f"{'Station':<20}{'Winter':<15}{'Spring':<15}{'Summer':<15}{'Autumn':<15}{'Avg BFI':<10}\n")
    f.write("-" * 90 + "\n")
    for station, winter, spring, summer, autumn, avg_bfi in summary_data:
        f.write(f"{station:<20}{(f'{winter:.2f}' if winter else 'N/A'):<15}{(f'{spring:.2f}' if spring else 'N/A'):<15}{(f'{summer:.2f}' if summer else 'N/A'):<15}{(f'{autumn:.2f}' if autumn else 'N/A'):<15}{(f'{avg_bfi:.2f}' if avg_bfi else 'N/A'):<10}\n")

print(f"Summary saved to {summary_file}.")

#%%  Generates a summary based on the season

import os
import pandas as pd

# Paths
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\output"
output_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\baseflow_volumes"
summary_file = os.path.join(output_folder, "summary2.txt")
seasonal_bfi_file = os.path.join(output_folder, "seasonal_bfi_summary2.txt")

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Constants
MONTHS_WINTER = [12, 1, 2]
MONTHS_SPRING = [3, 4, 5]
MONTHS_SUMMER = [6, 7, 8]
MONTHS_AUTUMN = [9, 10, 11]
DAYS_IN_SEASON = 92
MISSING_DATA_THRESHOLD = 0.05
YEAR_START = 1980
YEAR_END = 2020

def calculate_baseflow_volume(file_path, basin_code):
    df = pd.read_csv(file_path, sep="\t")
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce')
    df['Baseflow'] = pd.to_numeric(df['Baseflow'], errors='coerce')
    df['BFI Index'] = pd.to_numeric(df['BFI Index'], errors='coerce')
    df = df.dropna(subset=['Date', 'Baseflow', 'BFI Index'])
    if df.empty:
        return f"No valid data found for basin {basin_code}.\n", None, None

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df.loc[df['Month'] == 12, 'HydroYear'] = df['Year'] + 1
    df['HydroYear'].fillna(df['Year'], inplace=True)
    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    if df.empty:
        return f"No valid data in the year range {YEAR_START}-{YEAR_END} for basin {basin_code}.\n", None, None

    seasonal_volumes = {"Winter": [], "Spring": [], "Summer": [], "Autumn": []}
    seasonal_bfi = {"Winter": [], "Spring": [], "Summer": [], "Autumn": []}
    excluded_years = []

    for year, year_data in df.groupby('Year'):
        season_data = {
            "Winter": df[(df['HydroYear'] == year) & (df['Month'].isin(MONTHS_WINTER))],
            "Spring": year_data[year_data['Month'].isin(MONTHS_SPRING)],
            "Summer": year_data[year_data['Month'].isin(MONTHS_SUMMER)],
            "Autumn": year_data[year_data['Month'].isin(MONTHS_AUTUMN)]
        }

        skip_year = False
        for season, data in season_data.items():
            if len(data) < (1 - MISSING_DATA_THRESHOLD) * DAYS_IN_SEASON:
                excluded_years.append(year)
                skip_year = True
                break
        if skip_year:
            continue

        for season, data in season_data.items():
            seasonal_volumes[season].append(data['Baseflow'].sum() * 86400 / 1e6)
            seasonal_bfi[season].append(data['BFI Index'].mean())
    
    avg_volumes = {season: sum(volumes) / len(volumes) if volumes else None for season, volumes in seasonal_volumes.items()}
    avg_bfi = {season: sum(bfi) / len(bfi) if bfi else None for season, bfi in seasonal_bfi.items()}
    
    return None, (basin_code, avg_volumes["Winter"], avg_volumes["Spring"], avg_volumes["Summer"], avg_volumes["Autumn"]), (basin_code, avg_bfi["Winter"], avg_bfi["Spring"], avg_bfi["Summer"], avg_bfi["Autumn"])

summary_data = []
seasonal_bfi_data = []
for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(input_folder, file_name)
        basin_code = os.path.splitext(file_name)[0].replace("Out_", "")
        _, summary_entry, seasonal_bfi_entry = calculate_baseflow_volume(file_path, basin_code)
        if summary_entry:
            summary_data.append(summary_entry)
        if seasonal_bfi_entry:
            seasonal_bfi_data.append(seasonal_bfi_entry)

with open(summary_file, "w") as f:
    f.write(f"{'Station':<20}{'Winter':<15}{'Spring':<15}{'Summer':<15}{'Autumn':<15}\n")
    f.write("-" * 75 + "\n")
    for station, winter, spring, summer, autumn in summary_data:
        f.write(f"{station:<20}{(f'{winter:.2f}' if winter else 'N/A'):<15}{(f'{spring:.2f}' if spring else 'N/A'):<15}{(f'{summer:.2f}' if summer else 'N/A'):<15}{(f'{autumn:.2f}' if autumn else 'N/A'):<15}\n")

with open(seasonal_bfi_file, "w") as f:
    f.write(f"{'Station':<20}{'Winter':<15}{'Spring':<15}{'Summer':<15}{'Autumn':<15}\n")
    f.write("-" * 75 + "\n")
    for station, winter, spring, summer, autumn in seasonal_bfi_data:
        f.write(f"{station:<20}{(f'{winter:.2f}' if winter else 'N/A'):<15}{(f'{spring:.2f}' if spring else 'N/A'):<15}{(f'{summer:.2f}' if summer else 'N/A'):<15}{(f'{autumn:.2f}' if autumn else 'N/A'):<15}\n")

print(f"Summary saved to {summary_file}.")
print(f"Seasonal BFI summary saved to {seasonal_bfi_file}.")


#%% Graphs 1

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paths
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\output"
graphs_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\graphs"
os.makedirs(graphs_folder, exist_ok=True)

# Selected stations and their Gauge IDs
gauge_info = {
    "Stenay": "B315002002",
    "Chauvency-le-Château": "B460101001",
    "Treignes": "90211002",
    "Membre Pont": "94341002",
    "Chooz": "B720000002",
    "Amay": "71321002",
    "Gendron": "82211002",
    "Salzinnes": "L7319",
    "Tabreux": "59211002",
    "Chaud Fontaine Piscine": "62281002",
    "Martinrive": "66211002",
    "Eijsden grens": "Eijsden grens",
    "Borgharen": "6421500",
    "EbenEmael": "63400000",
    "Brommelen": "6Q18",
    "Meerssen": "10Q36",
    "Stah": "2829100000100"
}

# Seasons (for ordering purposes)
seasons = ["Winter", "Spring", "Summer", "Autumn"]

# Global matplotlib settings
rcParams['font.family'] = 'Calibri'
rcParams['font.size'] = 18
rcParams['figure.facecolor'] = 'none'

# Predefined station groups for scaling
high_scale_stations = ["Stenay", "Eijsden grens", "Borgharen", "Amay", "Chooz"]
low_scale_stations = [station for station in gauge_info.keys() if station not in high_scale_stations]

# Constants (as used in your volume script)
MONTHS_WINTER = [12, 1, 2]       # December (belongs to next winter), January, February
MONTHS_SPRING = [3, 4, 5]        # March, April, May
MONTHS_SUMMER = [6, 7, 8]        # June, July, August
MONTHS_AUTUMN = [9, 10, 11]      # September, October, November
DAYS_IN_SEASON = 92              # Approximate days per season
MISSING_DATA_THRESHOLD = 0.05    # 5% missing days allowed
YEAR_START = 1980                # Analysis start year
YEAR_END = 2020                  # Analysis end year

def compute_station_averages(file_path):
    """
    Reads the file and computes (per valid year) the baseflow volume (in MCM)
    and the total volume (baseflow volume divided by mean BFI Index) for each season.
    Then averages these seasonal values over all valid years.
    """
    df = pd.read_csv(file_path, sep="\t")
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce')
    df['Baseflow'] = pd.to_numeric(df['Baseflow'], errors='coerce')
    df['BFI Index'] = pd.to_numeric(df['BFI Index'], errors='coerce')
    df = df.dropna(subset=['Date', 'Baseflow', 'BFI Index'])
    if df.empty:
        return None

    # Set up year and month; assign December to the next winter (HydroYear)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df.loc[df['Month'] == 12, 'HydroYear'] = df['Year'] + 1
    df['HydroYear'].fillna(df['Year'], inplace=True)
    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    if df.empty:
        return None

    # Dictionaries to hold seasonal values for each valid year
    seasonal_baseflow = {season: [] for season in seasons}
    seasonal_total = {season: [] for season in seasons}

    # Group by original Year (note: winter is based on HydroYear)
    for year, year_data in df.groupby('Year'):
        # Define seasonal subsets exactly as in the volume code:
        season_data = {
            "Winter": df[(df['HydroYear'] == year) & (df['Month'].isin(MONTHS_WINTER))],
            "Spring": year_data[year_data['Month'].isin(MONTHS_SPRING)],
            "Summer": year_data[year_data['Month'].isin(MONTHS_SUMMER)],
            "Autumn": year_data[year_data['Month'].isin(MONTHS_AUTUMN)]
        }
        # Exclude the entire year if any season is missing too many days
        valid_year = True
        for season, data in season_data.items():
            if len(data) < (1 - MISSING_DATA_THRESHOLD) * DAYS_IN_SEASON:
                valid_year = False
                break
        if not valid_year:
            continue

        # For each season, calculate the baseflow volume (MCM) and total volume.
        # (Volume = sum(Baseflow) * 86400 / 1e6; total = volume / mean(BFI Index))
        for season, data in season_data.items():
            bf_volume = data['Baseflow'].sum() * 86400 / 1e6
            bfi_mean = data['BFI Index'].mean()
            if pd.isna(bfi_mean) or bfi_mean == 0:
                continue
            total_volume = bf_volume / bfi_mean
            seasonal_baseflow[season].append(bf_volume)
            seasonal_total[season].append(total_volume)

    # Compute the average (over valid years) for each season
    avg_seasonal_baseflow = {}
    avg_seasonal_total = {}
    for season in seasons:
        if seasonal_baseflow[season]:
            avg_seasonal_baseflow[season] = sum(seasonal_baseflow[season]) / len(seasonal_baseflow[season])
        else:
            avg_seasonal_baseflow[season] = 0
        if seasonal_total[season]:
            avg_seasonal_total[season] = sum(seasonal_total[season]) / len(seasonal_total[season])
        else:
            avg_seasonal_total[season] = 0

    return {'baseflow': avg_seasonal_baseflow, 'total': avg_seasonal_total}

# First, loop over stations to compute averages and also determine maximum total volumes for scaling.
high_scale_max = 0
low_scale_max = 0
station_averages = {}  # store computed seasonal averages for each station
for basin_name, gauge_id in gauge_info.items():
    file_path = os.path.join(input_folder, f"Out_{gauge_id}.txt")
    if not os.path.exists(file_path):
        continue
    averages = compute_station_averages(file_path)
    if averages is None:
        continue
    station_averages[basin_name] = averages
    station_max = max(averages['total'].values())
    if basin_name in high_scale_stations:
        high_scale_max = max(high_scale_max, station_max)
    else:
        low_scale_max = max(low_scale_max, station_max)

def plot_basin_graph(basin_name, averages, scale_max):
    """
    Plot the averaged seasonal volumes for a given station.
    'averages' is a dictionary with keys 'baseflow' and 'total', each a dict for the seasons.
    """
    baseflow_volumes = [averages['baseflow'][season] for season in seasons]
    total_volumes = [averages['total'][season] for season in seasons]
    x = range(len(seasons))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, total_volumes, width=bar_width, label="Total Volume", color='blue', edgecolor='black', align='center')
    ax.bar(x, baseflow_volumes, width=bar_width, label="Baseflow Volume", color='red', edgecolor='black', hatch='//', align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Volume (MCM)")
    ax.set_title(f"Seasonal Volumes for {basin_name}")
    ax.legend()
    ax.set_ylim(0, scale_max * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_folder, f"{basin_name}_seasonal_volumes.png"), transparent=True)
    plt.close()

# Finally, loop over each station and generate the graph using the computed averages.
for basin_name, gauge_id in gauge_info.items():
    file_path = os.path.join(input_folder, f"Out_{gauge_id}.txt")
    if not os.path.exists(file_path):
        print(f"File not found for {basin_name} ({gauge_id}), skipping.")
        continue
    if basin_name not in station_averages:
        print(f"No valid data for {basin_name}, skipping.")
        continue
    averages = station_averages[basin_name]
    scale_max = high_scale_max if basin_name in high_scale_stations else low_scale_max
    plot_basin_graph(basin_name, averages, scale_max)
    print(f"Graph generated for {basin_name}.")

#%% Graphs 2 Log

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paths
input_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\output"
graphs_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\graphs\Log"
os.makedirs(graphs_folder, exist_ok=True)

# Selected stations and their Gauge IDs
gauge_info = {
    "Stenay": "B315002002",
    "Chauvency-le-Château": "B460101001",
    "Treignes": "90211002",
    "Membre Pont": "94341002",
    "Chooz": "B720000002",
    "Amay": "71321002",
    "Gendron": "82211002",
    "Salzinnes": "L7319",
    "Tabreux": "59211002",
    "Chaud Fontaine Piscine": "62281002",
    "Martinrive": "66211002",
    "Eijsden grens": "Eijsden grens",
    "Borgharen": "6421500",
    "EbenEmael": "63400000",
    "Brommelen": "6Q18",
    "Meerssen": "10Q36",
    "Stah": "2829100000100"
}

# Seasons (for ordering)
seasons = ["Winter", "Spring", "Summer", "Autumn"]

# Global matplotlib settings
rcParams['font.family'] = 'Calibri'
rcParams['font.size'] = 18
rcParams['figure.facecolor'] = 'none'

# Constants (matching the volume script)
MONTHS_WINTER = [12, 1, 2]       # December is assigned to the following winter, then Jan, Feb.
MONTHS_SPRING = [3, 4, 5]
MONTHS_SUMMER = [6, 7, 8]
MONTHS_AUTUMN = [9, 10, 11]
DAYS_IN_SEASON = 92              # Approximate days per season
MISSING_DATA_THRESHOLD = 0.05    # Allow up to 5% missing days per season
YEAR_START = 1980                # Analysis start year
YEAR_END = 2020                  # Analysis end year

def compute_station_averages(file_path):
    """
    Reads the baseflow file and computes for each valid year the baseflow volume (in MCM)
    and the total volume (baseflow volume divided by the mean BFI Index) for each season.
    A year is only used if every season has at least (1 - MISSING_DATA_THRESHOLD) * DAYS_IN_SEASON days of data.
    Returns a dictionary with averaged seasonal volumes.
    """
    df = pd.read_csv(file_path, sep="\t")
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'].str.strip(), errors='coerce')
    df['Baseflow'] = pd.to_numeric(df['Baseflow'], errors='coerce')
    df['BFI Index'] = pd.to_numeric(df['BFI Index'], errors='coerce')
    df = df.dropna(subset=['Date', 'Baseflow', 'BFI Index'])
    if df.empty:
        return None

    # Set up year, month, and HydroYear (assign December to the following winter)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df.loc[df['Month'] == 12, 'HydroYear'] = df['Year'] + 1
    df['HydroYear'].fillna(df['Year'], inplace=True)
    df = df[(df['Year'] >= YEAR_START) & (df['Year'] <= YEAR_END)]
    if df.empty:
        return None

    # Dictionaries to accumulate seasonal volumes for valid years
    seasonal_baseflow = {season: [] for season in seasons}
    seasonal_total = {season: [] for season in seasons}

    # Loop over years (grouped by the calendar Year)
    for year, year_data in df.groupby('Year'):
        # Define seasonal subsets exactly as in the volume script:
        season_data = {
            "Winter": df[(df['HydroYear'] == year) & (df['Month'].isin(MONTHS_WINTER))],
            "Spring": year_data[year_data['Month'].isin(MONTHS_SPRING)],
            "Summer": year_data[year_data['Month'].isin(MONTHS_SUMMER)],
            "Autumn": year_data[year_data['Month'].isin(MONTHS_AUTUMN)]
        }
        # Skip the year if any season has too few days
        valid_year = True
        for season, data in season_data.items():
            if len(data) < (1 - MISSING_DATA_THRESHOLD) * DAYS_IN_SEASON:
                valid_year = False
                break
        if not valid_year:
            continue

        # For each season, compute the baseflow volume [MCM] and total volume.
        for season, data in season_data.items():
            bf_volume = data['Baseflow'].sum() * 86400 / 1e6
            bfi_mean = data['BFI Index'].mean()
            if pd.isna(bfi_mean) or bfi_mean == 0:
                continue
            total_volume = bf_volume / bfi_mean
            seasonal_baseflow[season].append(bf_volume)
            seasonal_total[season].append(total_volume)

    # Compute the average over all valid years for each season
    avg_seasonal_baseflow = {}
    avg_seasonal_total = {}
    for season in seasons:
        if seasonal_baseflow[season]:
            avg_seasonal_baseflow[season] = sum(seasonal_baseflow[season]) / len(seasonal_baseflow[season])
        else:
            avg_seasonal_baseflow[season] = 0
        if seasonal_total[season]:
            avg_seasonal_total[season] = sum(seasonal_total[season]) / len(seasonal_total[season])
        else:
            avg_seasonal_total[season] = 0

    return {'baseflow': avg_seasonal_baseflow, 'total': avg_seasonal_total}

# Compute station averages for all stations and store in a dictionary.
station_averages = {}
for basin_name, gauge_id in gauge_info.items():
    file_path = os.path.join(input_folder, f"Out_{gauge_id}.txt")
    if not os.path.exists(file_path):
        continue
    averages = compute_station_averages(file_path)
    if averages is None:
        continue
    station_averages[basin_name] = averages

# To use a consistent log scale across all graphs, compute the global minimum and maximum
# from all station total volumes (only consider positive values).
all_totals = []
for averages in station_averages.values():
    for season in seasons:
        val = averages['total'][season]
        if val > 0:
            all_totals.append(val)
if all_totals:
    global_max = max(all_totals)
    global_min = min(all_totals)
else:
    global_max = 1
    global_min = 0.1

# Define a plotting function that uses a logarithmic y-axis.
def plot_basin_graph(basin_name, averages, y_bottom, y_top):
    baseflow_volumes = [averages['baseflow'][season] for season in seasons]
    total_volumes = [averages['total'][season] for season in seasons]
    
    # Use a spacing factor to reduce the gap between seasonal columns.
    spacing = 0.8
    x = np.arange(len(seasons)) * spacing

    # Use one column per season (overlapping bars).
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw the total volume bar.
    ax.bar(x, total_volumes, width=bar_width, label="Total volume",
           color='lightblue', edgecolor='black')
    # Draw the baseflow volume bar on top.
    ax.bar(x, baseflow_volumes, width=bar_width, label="Baseflow volume",
           color='purple', edgecolor='black', hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Volume (MCM)")
    ax.set_title(f"Seasonal Volumes for {basin_name}")
    ax.legend()

    # Set the y-axis to a logarithmic scale and use the global limits.
    ax.set_yscale('log')
    ax.set_ylim(y_bottom, y_top)

    plt.tight_layout()
    # Save file with "_log" in the name.
    plt.savefig(os.path.join(graphs_folder, f"{basin_name}_seasonal_volumes_log.png"), transparent=True)
    plt.close()

# Use the global log-scale limits (same for all stations).
# For the lower limit, ensure it is nonzero (here we use global_min/1.1).
y_lower = global_min / 1.1
y_upper = 10000

for basin_name, gauge_id in gauge_info.items():
    file_path = os.path.join(input_folder, f"Out_{gauge_id}.txt")
    if not os.path.exists(file_path):
        print(f"File not found for {basin_name} ({gauge_id}), skipping.")
        continue
    if basin_name not in station_averages:
        print(f"No valid data for {basin_name}, skipping.")
        continue
    averages = station_averages[basin_name]
    plot_basin_graph(basin_name, averages, y_lower, y_upper)
    print(f"Graph generated for {basin_name}.")


#%% Graphs 3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Paths
txt_file_path = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\group.txt"
output_group_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\BFI3\graphs\group\new"
os.makedirs(output_group_folder, exist_ok=True)

# Read the data from the text file
df = pd.read_csv(txt_file_path, sep="\t")

# Ensure consistent column names
df.columns = [col.strip() for col in df.columns]

# Subgroups to plot
subgroups_to_plot = [
    "Upper Meuse",
    "Chiers",
    "Upper Meuse, Viroin, Semois",
    "Lesse, Middle Meuse, Amay (limit)",
    "Sambre",
    "Ourthe, Vesdre, Ambleve, Eijsden (limit)",
    "Geer, Geleenbeek, Geul",
    "Rur"
]

# Subgroup column
subgroup_column = "Sub-group"

# Filter the data for these subgroups
df_subgroups = df[df[subgroup_column].isin(subgroups_to_plot)].copy()

# Global plot settings
rcParams["font.family"] = "Calibri"
rcParams["font.size"] = 30

# Define colors for human influence signals
signal_colors = {
    "Aggravated": "red",
    "Alleviated": "green",
    "NoChange": "orange"
}

# Get max volume across all subgroups for consistent scaling
max_total_volume = 1  # Avoid log(0) issues
for _, row in df_subgroups.iterrows():
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        volume = pd.to_numeric(row[f"Avg {season} Volume (MCM)"], errors="coerce")
        if not np.isnan(volume):
            max_total_volume = max(max_total_volume, volume)

# Function to plot graphs for subgroups
def plot_subgroup(subgroup, row):
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    # Extract and convert volumes to numeric
    bf_volumes = np.array([pd.to_numeric(row[f"Avg BF {season} Volume (MCM)"], errors="coerce") for season in seasons])
    total_volumes = np.array([pd.to_numeric(row[f"Avg {season} Volume (MCM)"], errors="coerce") for season in seasons])
    signals = [row[f"Human influence signal {season}"] for season in seasons]

    # Handle cases with invalid or missing data
    if np.isnan(bf_volumes).any() or np.isnan(total_volumes).any():
        print(f"Invalid or missing volume data for subgroup: {subgroup}")
        return

    x = np.arange(len(seasons))
    bar_width = 0.4  # Adjust bar width

    fig, ax = plt.subplots(figsize=(12, 9))
    # Plot stacked bars: Baseflow inside Total Volume
    # Plot stacked bars: Baseflow inside Total Volume
    ax.bar(x, total_volumes, width=bar_width, color="blue", edgecolor="black", label="Total volume")  # Sky Blue
    ax.bar(x, bf_volumes, width=bar_width, color="lightblue", edgecolor="black", hatch="//", label="Baseflow volume")  # Slate Gray

    # Set log scale
    ax.set_yscale("log")
    ax.set_ylim(1, max_total_volume * 2)  # Ensure consistent scale across subgroups
    ax.set_xlim(-0.5, len(seasons) - 0.5)  # Ensure same x-axis range for all

    # Set axis labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Volume (log MCM)")
    ax.set_title(f"{subgroup}")
    ax.legend()

    # Add signal boxes below x-axis
    for i, (season, signal) in enumerate(zip(seasons, signals)):
        color = signal_colors.get(signal, "gray")
        ax.annotate(
            signal,
            xy=(i, -0.1),  # Position below x-axis
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            fontsize=20,
            bbox=dict(facecolor=color, edgecolor="black", boxstyle="round,pad=0.5", alpha=0.7)
        )

    # Save the figure
    plt.tight_layout()
    out_file = os.path.join(output_group_folder, f"{subgroup.replace(' ', '_')}_logscale_stacked.png")
    plt.savefig(out_file, transparent=True)
    plt.close()
    print(f"Graph saved for subgroup: {subgroup}")

# Generate graphs for each subgroup
for subgroup in subgroups_to_plot:
    row = df_subgroups[df_subgroups[subgroup_column] == subgroup]
    if row.empty:
        print(f"No data found for subgroup: {subgroup}")
        continue
    plot_subgroup(subgroup, row.iloc[0])



#%%