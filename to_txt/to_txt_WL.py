#%% Developed by Deborah, November 2024
# Sequence of steps to save the data provided by Marco from Waterschap Limburg into the correct format

# Selects and saves selected stations
import pandas as pd
import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point

# File paths
input_excel_path = r"D:\My Documents\LoFlowMaas\Waterschap Limburg Stations_Marco\Metadata.xlsx"
output_shp_path = r"D:\My Documents\LoFlowMaas\GIS\Data\Saved\stations_WL.shp"

# Load the metadata from the Excel file
df = pd.read_excel(input_excel_path, usecols=['ID', 'Naam', 'X', 'Y'])

# Create a GeoDataFrame, with points from the X and Y coordinates
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['X'], df['Y']),
    crs=CRS.from_epsg(28992) 
)

# Reproject to WGS84 
gdf = gdf.to_crs(epsg=4326)

# Save to shapefile
gdf.to_file(output_shp_path, driver='ESRI Shapefile')

print("Shapefile created successfully at:", output_shp_path)


#%% Calculates the correct areas

import geopandas as gpd
import pandas as pd

# Define file paths
stations_path = r"D:\My Documents\LoFlowMaas\GIS\Data\Saved\stations_WL.shp"
stroomgebied_geul_path = r"D:\My Documents\LoFlowMaas\Waterschap Limburg Stations_Marco\stroomgebiedenQ\Geul_stroomgebiedenQ.shp"
stroomgebied_geleenbeek_path = r"D:\My Documents\LoFlowMaas\Waterschap Limburg Stations_Marco\stroomgebiedenQ\Geleenbeek_stroomgebiedenQ.shp"
output_stations_path = r"D:\My Documents\LoFlowMaas\GIS\Data\Saved\stations_WL.shp"

# Load the shapefiles
stations = gpd.read_file(stations_path)
stroomgebied_geul = gpd.read_file(stroomgebied_geul_path)
stroomgebied_geleenbeek = gpd.read_file(stroomgebied_geleenbeek_path)

# Convert both stroomgebied files to DataFrames for easier merging
stroomgebied_geul_df = stroomgebied_geul[['Qmeter', 'Shape_Area']].rename(columns={'Qmeter': 'ID', 'Shape_Area': 'Geul_Area'})
stroomgebied_geleenbeek_df = stroomgebied_geleenbeek[['Qmeter', 'Shape_Area']].rename(columns={'Qmeter': 'ID', 'Shape_Area': 'Geleenbeek_Area'})

# Merge the stations with the Geul stroomgebied areas based on ID
stations_with_areas = stations.merge(stroomgebied_geul_df, on='ID', how='left')

# Merge the result with the Geleenbeek stroomgebied areas
stations_with_areas = stations_with_areas.merge(stroomgebied_geleenbeek_df, on='ID', how='left')

# Calculate Total_Area by summing Geul_Area and Geleenbeek_Area (fill NaN with 0)
stations_with_areas['Total_Area'] = stations_with_areas['Geul_Area'].fillna(0) + stations_with_areas['Geleenbeek_Area'].fillna(0)

# Save the updated stations with area information
stations_with_areas.to_file(output_stations_path, driver='ESRI Shapefile')


#%% Creates the txt files in the correct format
import pandas as pd
import numpy as np
import os

# Load the CSV and define output dir
file_path = r'D:\My Documents\LoFlowMaas\Waterschap Limburg Stations_Marco\Export watersysteem WL.csv'
data = pd.read_csv(file_path, delimiter=',', header=0)  # Use header=0 to read the first row as headers
output_dir = r'D:\My Documents\LoFlowMaas\Discharge\WL'
os.makedirs(output_dir, exist_ok=True)


# Print the raw data to check the structure
print("Raw Data Preview:")
print(data.head())

# Print column names for debugging
print("Original Columns:", data.columns.tolist())

# Clean column names
data.columns = data.columns.str.strip()

# Ensure that the Date column is parsed as datetime
try:
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
except Exception as e:
    print("Error parsing dates:", e)

# Check if 'Date' column exists after cleaning
if 'Date' not in data.columns:
    raise KeyError("'Date' column not found in the DataFrame. Available columns: {}".format(data.columns.tolist()))

# Process each station
for column in data.columns[1:]:  # Skip the first column (Date)
    # Get Date and discharge data
    station_data = data[['Date', column]].copy()
    station_data.columns = ['DateTime', 'Discharge']

    # Set DateTime as index
    station_data.set_index('DateTime', inplace=True)

    # Replace values lower than -10 with NaN
    station_data['Discharge'] = station_data['Discharge'].replace(-999, np.nan)  # Replace -999 with NaN
    station_data['Discharge'] = np.where(station_data['Discharge'] < -10, np.nan, station_data['Discharge'])

    # Resample to daily mean, ignoring NaN values
    daily_data = station_data.resample('D').mean()

    # Reset index to have Date as a column
    daily_data.reset_index(inplace=True)

    # Convert Date format to yyyymmdd and round discharge values
    daily_data['Date'] = daily_data['DateTime'].dt.strftime('%Y%m%d')
    daily_data['Discharge'] = daily_data['Discharge'].round(2)
    
    # Prepare final DataFrame with only Date and Discharge columns
    final_data = daily_data[['Date', 'Discharge']]

    # Get the gauge_id from the column name
    gauge_id = column
    
    # Save to text file, using NaN where appropriate
    output_file_path = os.path.join(output_dir, f'{gauge_id}.txt')
    final_data.to_csv(output_file_path, sep=' ', index=False, header=True, na_rep='NaN')

    print(f"Processed and saved file for gauge: {gauge_id}")

print("Files created successfully.")


#%% Inspects the txt with plot
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set the directory containing the text files
input_dir = r'D:\My Documents\LoFlowMaas\Discharge\WL'

# List to store DataFrames
dataframes = []

# Read each text file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)
        
        # Read the text file
        df = pd.read_csv(file_path, sep=' ', na_values='NaN', parse_dates=['Date'])
        
        # Add a column for the gauge ID
        df['Gauge ID'] = filename[:-4]  # Remove the .txt extension
        
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
all_data = pd.concat(dataframes, ignore_index=True)

# Plotting
plt.figure(figsize=(14, 8))

# Loop through each gauge ID and create a time series plot
for gauge_id, group in all_data.groupby('Gauge ID'):
    plt.plot(group['Date'], group['Discharge'], label=gauge_id)

# Adding plot details
plt.title('Daily Mean Discharge for Each Gauge')
plt.xlabel('Date')
plt.ylabel('Discharge (m³/s)')
plt.legend(title='Gauge ID')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
#%%