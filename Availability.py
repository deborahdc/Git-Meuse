#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# Directory containing the _average.txt files
input_dir = r'D:\My Documents\LoFlowMaas\Discharge\BE-W\Wallonie\station_outputs'

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
    
    # Unstack to create a heatmap-friendly format (Years as rows, Months as columns)
    availability_pivot = monthly_percentage.unstack(level=1)
    
    # Plot the heatmap for this station
    plt.figure(figsize=(12, 8))
    
    # Create a heatmap for the station, ensuring 0 is displayed for missing data (in red)
    sns.heatmap(availability_pivot, cmap="RdYlGn", cbar_kws={'label': '% Data Availability'}, vmin=0, vmax=100, annot=True, fmt=".0f")
    
    # Set plot labels and title
    plt.title(f'Monthly Data Availability for {station_name} (by Year)')
    plt.xlabel('Month')
    plt.ylabel('Year')
    
    # Set x-axis ticks for months in the middle of each cell
    plt.xticks(ticks=[i + 0.5 for i in range(12)], labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Show the plot
    plt.tight_layout()
    
    # Save the figure for each station
    #plt.savefig(f"{station_name}_data_availability.png")
    plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# Directory containing the _average.txt files
input_dir = r'D:\My Documents\LoFlowMaas\Discharge\BE-W\Wallonie\station_outputs'

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

# Define dynamic figure size based on the number of stations and years
fig_width = max(8, num_years * 0.5)  # Minimum width of 8, scale by 0.5 per year
fig_height = max(2, num_stations * 0.5)  # Minimum height of 2, scale by 0.5 per station

# Plotting using a color ramp from red (0%) to green (100%)
plt.figure(figsize=(fig_width, fig_height))  # Dynamic plot size

# Create a color map using seaborn with thin lines and larger annotation font size
sns.heatmap(
    availability_pivot, cmap="RdYlGn", cbar_kws={'label': '% Data Availability'}, 
    vmin=0, vmax=100, annot=True, fmt=".0f", linewidths=0.5, linecolor='gray',
    annot_kws={"size": 8}  # Font size for percentage annotations
)

# Set plot labels and title
plt.title('Yearly Data Availability by Station', fontsize=12)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Station', fontsize=10)

# Adjust the size of x-axis (years) and y-axis (stations) labels
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# Rotate x-axis labels to horizontal
plt.xticks(rotation=0)

# Show the plot
plt.tight_layout()
plt.show()


#%%