#%% Developed by Deborah, 2nd of October 2024

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# This codes generate the availability for all stations, but monthly inspection for each station
# To generate all together, copy the data data was saved separetely for each station folder in a folder with all stations
input_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated'

# List all average discharge files
files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

# Iterate over each station file
for file in files:
    file_path = os.path.join(input_dir, file)
    station_data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    
    # Convert the 'Date' column to datetime format
    station_data['Date'] = pd.to_datetime(station_data['Date'], format='%Y%m%d')
    
    # Extract station name from the file name
    station_name = file.split('_')[0]
    
    # Extract year and month for grouping
    station_data['Year'] = station_data['Date'].dt.year
    station_data['Month'] = station_data['Date'].dt.month

    # Filter data for years starting from 1980
    station_data = station_data[station_data['Year'] >= 1980]
    
    # Group by Year and Month, and calculate the count of available data
    monthly_availability = station_data.groupby(['Year', 'Month']).apply(lambda x: x['Discharge'].count())
    
    # Create a complete index for Year and Month, and reindex the grouped data
    all_years = station_data['Year'].unique()
    full_index = pd.MultiIndex.from_product([all_years, range(1, 13)], names=['Year', 'Month'])
    monthly_availability = monthly_availability.reindex(full_index, fill_value=0)
    
    # Get the total number of days in each month
    days_per_month = station_data.groupby(['Year', 'Month']).apply(lambda x: x['Date'].dt.days_in_month.max())
    days_per_month = days_per_month.reindex(full_index, fill_value=np.nan)
    
    # Calculate percentage availability
    monthly_percentage = (monthly_availability / days_per_month) * 100
    
    # Replace missing data (NaN) with 0
    monthly_percentage = monthly_percentage.fillna(0)
    availability_pivot = monthly_percentage.unstack(level=1)
    
    # Plot the heatmap for this station
    plt.figure(figsize=(12, 8))
    
    # Create a heatmap for the station
    sns.heatmap(availability_pivot, cmap="RdYlGn", cbar_kws={'label': '% Data Availability'}, vmin=0, vmax=100, annot=True, fmt=".0f")
    
    # Set plot labels and title
    plt.title(f'Monthly Data Availability for {station_name} (by Year)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    # Set x-axis ticks for months in the middle of each cell
    plt.xticks(ticks=[i + 0.5 for i in range(12)], labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Show the plot
    plt.tight_layout()
   
    #plt.savefig(f"{station_name}_data_availability.png")
    plt.show()

#%% (All) This codes generate the availability for all stations together, yearly percentage

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# To generate all together, copy the data data was saved separetely for each station folder in a folder with all stations
input_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated'

# List all average discharge files
files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

# Create an empty DataFrame to hold all data
all_data = pd.DataFrame()

# Read each file and append the data
for file in files:
    file_path = os.path.join(input_dir, file)
    station_data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    
    # Convert the 'Date' column to datetime format
    station_data['Date'] = pd.to_datetime(station_data['Date'], format='%Y%m%d')
    
    # Extract station name from the file name
    station_name = file.split('_')[0]
    # Add station name as a column
    station_data['Station'] = station_name
    
    # Append to all_data
    all_data = pd.concat([all_data, station_data])

# Extract year for grouping
all_data['Year'] = all_data['Date'].dt.year

# Filter data for years starting from 1980
all_data = all_data[all_data['Year'] >= 1980]

# Get a complete list of years and stations
all_stations = all_data['Station'].unique()
all_years = all_data['Year'].unique()

# Create a full index of all stations and years
full_index = pd.MultiIndex.from_product([all_stations, all_years], names=['Station', 'Year'])

# Group by Station and Year, and calculate the count of available data
yearly_availability = all_data.groupby(['Station', 'Year']).apply(lambda x: x['Discharge'].count())

# Reindex the availability data to ensure all years are represented for all stations
yearly_availability = yearly_availability.reindex(full_index, fill_value=0)

# Get the total number of days in each year for each station
days_per_year = all_data.groupby(['Station', 'Year']).apply(lambda x: x['Date'].dt.dayofyear.max())
days_per_year = days_per_year.reindex(full_index, fill_value=np.nan)  # Keep NaN for missing data

# Calculate percentage availability
yearly_percentage = (yearly_availability / days_per_year) * 100

# Replace missing data (NaN) with 0
yearly_percentage = yearly_percentage.fillna(0)

# Pivot the data to create a heatmap-friendly format (Years as columns, Stations as rows)
availability_pivot = yearly_percentage.unstack(level=1)

# Determine the number of stations and years
num_stations = availability_pivot.shape[0]  # Number of stations (rows)
num_years = availability_pivot.shape[1]     # Number of years (columns)
fig_width = max(8, num_years * 0.5)  
fig_height = max(2, num_stations * 0.5) 

# Plotting using a color ramp from red (0%) to green (100%)
plt.figure(figsize=(fig_width, fig_height)) 

# Create a color map using seaborn with thin lines and larger annotation font size
sns.heatmap(
    availability_pivot, cmap="RdYlGn", cbar_kws={'label': '% Data Availability'}, 
    vmin=0, vmax=100, annot=True, fmt=".0f", linewidths=0.5, linecolor='gray',
    annot_kws={"size": 8}  # Font size for percentage annotations
)

# Set plot labels and title
plt.title('Yearly Data Availability by Station', fontsize=20)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Station', fontsize=20)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#%%  (All) but instead of plotting all from a folder, it reads from a folder but plots only the ones listed in the excel 
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

input_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated'

# Path to the Excel file
excel_file_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\Info_EStreams3.xlsx' # Replace if updated

# Load the gauge IDs from the spreadsheet
gauge_info = pd.read_excel(excel_file_path, dtype=str)

# Function to clean and format gauge IDs, handling both numeric and non-numeric IDs
def format_gauge_id(x):
    try:
        # Try converting to a float and then to an integer string if possible
        return str(int(float(x)))
    except ValueError:
        # If conversion fails (non-numeric), return the original string
        return str(x).strip()  # Remove any extra whitespace just in case

# Apply the formatting function to both 'gauge_id' and 'paired_id' columns
gauge_info['gauge_id'] = gauge_info['gauge_id'].apply(lambda x: format_gauge_id(x) if pd.notna(x) else None)
gauge_info['paired_id'] = gauge_info['paired_id'].apply(lambda x: format_gauge_id(x) if pd.notna(x) else None)

# Collect gauge IDs from both 'gauge_id' and 'paired_id' columns, removing any NaN values
gauge_ids = set(gauge_info['gauge_id'].dropna().tolist() + gauge_info['paired_id'].dropna().tolist())

# Add ".txt" extension to each gauge ID for matching
gauge_ids_with_txt = {f"{gauge_id}.txt" for gauge_id in gauge_ids}

# List all files in the directory and print them to verify
all_files = os.listdir(input_dir)

# Filter files in the directory to match only those with names in gauge_ids_with_txt
files = [f for f in all_files if f in gauge_ids_with_txt]

# Check if any files are found and print them for confirmation
if not files:
    print("No matching files found based on gauge IDs.")
else:
    print("Matching files found:", files)

# Initialize an empty DataFrame to hold all data
all_data = pd.DataFrame()

# Read each file and append the data
for file in files:
    file_path = os.path.join(input_dir, file)
    station_data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    
    # Check if the 'Date' column was successfully created
    if 'Date' in station_data.columns:
        # Convert the 'Date' column to datetime format
        station_data['Date'] = pd.to_datetime(station_data['Date'], format='%Y%m%d', errors='coerce')
        
        # Drop rows with invalid dates to avoid issues
        station_data = station_data.dropna(subset=['Date'])
        
        # Extract station name from the file name (gauge ID without ".txt")
        station_name = file.split('_')[0].replace('.txt', '')
        # Add station name as a column
        station_data['Station'] = station_name
        
        # Append to all_data
        all_data = pd.concat([all_data, station_data])
    else:
        print(f"Warning: 'Date' column missing in file {file}")

# Check if all_data has entries before proceeding
if not all_data.empty:
    # Extract year for grouping
    all_data['Year'] = all_data['Date'].dt.year

    # Filter data for years from 1980 to 2020
    all_data = all_data[(all_data['Year'] >= 1980) & (all_data['Year'] <= 2020)]


    # Get a complete list of years and stations
    all_stations = all_data['Station'].unique()
    all_years = all_data['Year'].unique()

    # Create a full index of all stations and years
    full_index = pd.MultiIndex.from_product([all_stations, all_years], names=['Station', 'Year'])

    # Group by Station and Year, and calculate the count of available data
    yearly_availability = all_data.groupby(['Station', 'Year']).apply(lambda x: x['Discharge'].count())

    # Reindex the availability data to ensure all years are represented for all stations
    yearly_availability = yearly_availability.reindex(full_index, fill_value=0)

    # Get the total number of days in each year for each station
    days_per_year = all_data.groupby(['Station', 'Year']).apply(lambda x: x['Date'].dt.dayofyear.max())
    days_per_year = days_per_year.reindex(full_index, fill_value=np.nan)  # Keep NaN for missing data

    # Calculate percentage availability
    yearly_percentage = (yearly_availability / days_per_year) * 100

    # Replace missing data (NaN) with 0
    yearly_percentage = yearly_percentage.fillna(0)

    # Pivot the data to create a heatmap-friendly format (Years as columns, Stations as rows)
    availability_pivot = yearly_percentage.unstack(level=1)

    # Determine the number of stations and years
    num_stations = availability_pivot.shape[0]
    num_years = availability_pivot.shape[1]    

    # Define dynamic figure size
    fig_width = max(8, num_years * 0.5)  
    fig_height = max(2, num_stations * 0.5)  

    # Plotting using a color ramp from red (0%) to green (100%)
    plt.figure(figsize=(fig_width, fig_height))

    # Create a color map using seaborn with thin lines and larger annotation font size
    sns.heatmap(
        availability_pivot, 
        cmap="RdYlGn", 
        cbar_kws={'label': '% Data Availability', 'shrink': 0.5},  # Adjust color bar size
        vmin=0, vmax=100, 
        annot=True, 
        fmt=".0f", 
        linewidths=0.5, 
        linecolor='gray',
        annot_kws={"size": 6}  # Reduce annotation font size inside squares
    )

    # Increase the font size of the color bar labels
    cbar = plt.gca().collections[0].colorbar  # Get the color bar
    cbar.ax.tick_params(labelsize=20)  # Set color bar font size
    cbar.set_label('% Data Availability', fontsize=20)  # Increase label font size


    # Set plot labels and title
    plt.title('Yearly Data Availability by Station', fontsize=20)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Station', fontsize=20)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

#%%
