#%% Developed by Deborah, 1st of October 2024
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS
import contextily as ctx

# Load the CSV file with a semicolon delimiter
file_path = r'D:\My Documents\LoFlowMaas\Discharge\info.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')

# Define a function to assign the correct CRS based on the projection column
def assign_crs(projection):
    if 'ETRS89LCC' in projection:
        return 'EPSG:3034'  # Assuming ETRS89 / LCC
    elif 'Lambert 93' in projection:
        return 'EPSG:2154'  # Lambert 93
    elif 'Lambert2008' in projection:
        return 'EPSG:3812'  # Lambert 2008
    elif 'Lambert2010' in projection:
        return 'EPSG:3035'  # Lambert 2010
    else:
        raise ValueError(f"Unknown projection: {projection}")

# Apply the correct CRS to each row
gdfs = []  # List to store the GeoDataFrames
for _, row in df.iterrows():
    crs_code = assign_crs(row['projection'])  # Determine CRS based on the projection
    gdf = gpd.GeoDataFrame([row], geometry=gpd.points_from_xy([row['long']], [row['lat']]), crs=crs_code)
    gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
    gdfs.append(gdf)

# Concatenate all GeoDataFrames
gdf_combined = pd.concat(gdfs)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the station locations
gdf_combined.plot(ax=ax, marker='o', color='red', markersize=50, label='Stations')

# Add labels to the points (station names)
for x, y, label in zip(gdf_combined.geometry.x, gdf_combined.geometry.y, gdf_combined['StationName']):
    ax.text(x, y, label, fontsize=8, ha='right')

# Add the Google Satellite basemap
ctx.add_basemap(ax, crs=gdf_combined.crs.to_string(), source=ctx.providers.Esri.WorldImagery)

# Set title and labels
ax.set_title('Station Locations with Correct Coordinates')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Show the plot
plt.legend()
plt.show()


#%% 
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

# Load the CSV file with a semicolon delimiter
file_path = r'D:\My Documents\LoFlowMaas\Discharge\info.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1', delimiter=';')

# Define a function to transform coordinates to WGS84
def transform_to_wgs84(row):
    # Define the transformer based on the current projection
    if 'ETRS89LCC' in row['projection']:
        transformer = Transformer.from_crs('EPSG:3034', 'EPSG:4326', always_xy=True)
    elif 'Lambert 93' in row['projection']:
        transformer = Transformer.from_crs('EPSG:2154', 'EPSG:4326', always_xy=True)
    elif 'Lambert2008' in row['projection']:
        transformer = Transformer.from_crs('EPSG:3812', 'EPSG:4326', always_xy=True)
    elif 'Lambert2010' in row['projection']:
        transformer = Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
    else:
        return pd.Series([None, None])  # In case of unknown projection, return None
    
    # Transform the long and lat
    long_wgs84, lat_wgs84 = transformer.transform(row['long'], row['lat'])
    return pd.Series([long_wgs84, lat_wgs84])

# Apply the transformation function and add new columns for WGS84 coordinates
df[['long_WGS84', 'lat_WGS84']] = df.apply(transform_to_wgs84, axis=1)

# Save the updated CSV file with the new columns
output_path = r'D:\My Documents\LoFlowMaas\Discharge\info_with_WGS84_coordinates.csv'
df.to_csv(output_path, index=False, sep=';')

print(f"New CSV file with WGS84 coordinates saved at {output_path}")


#%%