#%% Developed by Deborah, October 2024
# Selects and saves selected stations

# Abstraction is actually the maximum demand (update November 2024)

import geopandas as gpd

# Define file paths
catchment_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/estreams_catchments.shp"
stations_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/selected_downloaded.shp"
output_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/estreams_selected.shp"

# Load the shapefiles
catchments = gpd.read_file(catchment_path)
stations = gpd.read_file(stations_path)

# Ensure both shapefiles have the same coordinate reference system (CRS)
catchments = catchments.to_crs("EPSG:4326")
stations = stations.to_crs("EPSG:4326")

# Filter catchments based on gauge_id present in selected stations
matching_gauges = stations['gauge_id'].unique()  # List of unique gauge_ids in stations
filtered_catchments = catchments[catchments['gauge_id'].isin(matching_gauges)]  # Filter catchments with matching gauge_ids

# Sort the filtered catchments from smallest to largest based on 'area_calc' column
sorted_catchments = filtered_catchments.sort_values(by='area_calc', ascending=False)

# Save the sorted filtered catchments to a new shapefile
sorted_catchments.to_file(output_path, driver="ESRI Shapefile")

print("Filtered and sorted catchments have been saved as estreams_selected.shp")


#%% Takes the selected and compare with the nodes

import geopandas as gpd

# Define file paths
basins_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/estreams_selected.shp"
nodes_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/NODES.shp"
output_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/Nodes_basin.shp"

# Load the shapefiles
basins = gpd.read_file(basins_path)
nodes = gpd.read_file(nodes_path)

# Ensure both shapefiles have the same coordinate reference system (CRS)
basins = basins.to_crs("EPSG:4326")
nodes = nodes.to_crs("EPSG:4326")

# Perform spatial join to find nodes within each basin
joined = gpd.sjoin(nodes, basins[['gauge_id', 'geometry']], how="inner", predicate="within")

# Group by nodes and aggregate gauge_id values for nodes within multiple basins
joined['gauge_ids'] = joined.groupby(joined.index)['gauge_id'].transform(lambda x: ','.join(x.astype(str)))
joined_unique = joined.drop_duplicates(subset='geometry')  # Remove duplicate rows based on node geometry

# Create final GeoDataFrame with 'gauge_ids' column
nodes_in_basins = nodes.merge(joined_unique[['gauge_ids']], left_index=True, right_index=True, how="inner")

# Save the result to a new shapefile
nodes_in_basins.to_file(output_path, driver="ESRI Shapefile")

print("Nodes within basins have been saved as Nodes_basin.shp with gauge_id column.")

#%% Add the abstraction data from excel to the shp

import pandas as pd
import geopandas as gpd

# Define file paths
nodes_basin_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/Nodes.shp"
excel_path = "D:/My Documents/LoFlowMaas/Deltares_Marjolein/MeuseWaterUsersAndWaterInfrastructure.xlsx"
output_path = "D:/My Documents/LoFlowMaas/GIS/Data/Saved/Nodes_abstraction.shp"

# Load the Nodes_basin shapefile
nodes_basin = gpd.read_file(nodes_basin_path)

# Load the abstraction data from the Excel file
excel_data = pd.read_excel(excel_path, sheet_name="Water users")

# Filter only the relevant columns from the Excel data
abstraction_data = excel_data[['NodeId', 'Abstraction (sink) expected (m3/s)']]

# Merge the abstraction data with nodes_basin based on 'ID' in nodes_basin and 'NodeId' in the Excel data
nodes_basin = nodes_basin.merge(abstraction_data, left_on='ID', right_on='NodeId', how='left')
# Ensure that the abstraction column is numeric
nodes_basin['Abstraction (sink) expected (m3/s)'] = pd.to_numeric(nodes_basin['Abstraction (sink) expected (m3/s)'], errors='coerce')

# Save the updated shapefile
nodes_basin.to_file(output_path, driver="ESRI Shapefile")

print("Nodes with abstraction data have been saved as Nodes_basin_with_abstraction.shp")

#%%



