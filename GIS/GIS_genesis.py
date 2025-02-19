#%% Developed by Deborah, November 2024

import os
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import numpy as np


excel_file_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\Info_EStreams.xlsx'  # Excel with lat/lon
discharge_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\deficit'  # Directory with deficit files
recession_summary_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\recession\summary.txt'  # Recession constants file
attributes_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\all_basins_attributes.txt'  # Attributes file (root_dep, soil_tawc, lit_dom)
shapefiles_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved'  # Directory with shapefiles
output_map_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Codes_outputs\LowFlowAnalysis.html'  # Output map path
output_plot_dir = r'C:\Users\ddo001\Documents\LoFloMeuse\Codes_outputs\plots'  # Directory to save generated plots
os.makedirs(output_plot_dir, exist_ok=True) 

# Shapefiles to include on the map
shapefiles = {
    'Meuse_Basin': 'Meuse_basin.shp',
    'Meuse_River': 'Meuse_River.shp',
    'Main_Rivers': 'Main_rivers.shp',
    'Dams_Reservoirs': 'Dams_Reservoirs.shp' 
}

df = pd.read_excel(excel_file_path, dtype={'gauge_id': str})
df = df.dropna(subset=['lat', 'lon'])  

# Load recession summary data
recession_data = pd.read_csv(recession_summary_path, sep='\t', dtype={'Gauge ID': str})

# Load attributes data
attributes_data = pd.read_csv(attributes_path, sep='\t')

# Merge attributes and recession data with the main DataFrame
df = df.merge(recession_data, how='left', left_on='gauge_id', right_on='Gauge ID')
df = df.merge(attributes_data, how='left', left_on='basin_id', right_on='Basin_ID')

global_min_volume = float('inf')
global_max_volume = float('-inf')
global_min_duration = float('inf')
global_max_duration = float('-inf')

# Extract duration and volume from deficit folder
for _, row in df.iterrows():
    gauge_id = row['gauge_id']
    volume_file = os.path.join(discharge_dir, f"{gauge_id}_volume.txt")
    duration_file = os.path.join(discharge_dir, f"{gauge_id}_duration.txt")

    if os.path.exists(volume_file) and os.path.exists(duration_file):
        volume_data = pd.read_csv(volume_file, sep='\t')
        duration_data = pd.read_csv(duration_file, sep='\t')
        volume_values = volume_data[['Deficit_mm_Human', 'Deficit_mm_Natural']].values.flatten()
        global_min_volume = min(global_min_volume, np.nanmin(volume_values))
        global_max_volume = max(global_max_volume, np.nanmax(volume_values))
        duration_values = duration_data[['Days_Below_Threshold_Human', 'Days_Below_Threshold_Natural']].values.flatten()
        global_min_duration = min(global_min_duration, np.nanmin(duration_values))
        global_max_duration = max(global_max_duration, np.nanmax(duration_values))

y_limits_volume = (global_min_volume, global_max_volume)
y_limits_duration = (global_min_duration, global_max_duration)

# Generates the boxplot with deficit (duration and volume) for both paired and observed
def generate_average_deficit_boxplot_aggregated(volume_file, duration_file, station_name, y_limits_volume, y_limits_duration):
    try:

        volume_data = pd.read_csv(volume_file, sep='\t')
        duration_data = pd.read_csv(duration_file, sep='\t')

        volume_data_melted = volume_data.melt(
            id_vars=['Year'], 
            value_vars=['Deficit_mm_Human', 'Deficit_mm_Natural'], 
            var_name='Type', 
            value_name='Deficit'
        )
        duration_data_melted = duration_data.melt(
            id_vars=['Year'], 
            value_vars=['Days_Below_Threshold_Human', 'Days_Below_Threshold_Natural'], 
            var_name='Type', 
            value_name='Deficit'
        )
        mean_volume_human = volume_data['Deficit_mm_Human'].mean()
        mean_volume_natural = volume_data['Deficit_mm_Natural'].mean()
        mean_duration_human = duration_data['Days_Below_Threshold_Human'].mean()
        mean_duration_natural = duration_data['Days_Below_Threshold_Natural'].mean()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        sns.boxplot(
            data=volume_data_melted,
            x='Type',
            y='Deficit',
            ax=axes[0],
            palette={'Deficit_mm_Human': 'lightcoral', 'Deficit_mm_Natural': 'lightgreen'}
        )
        axes[0].set_title(f'{station_name} - Average Volume Deficit')
        axes[0].set_xlabel('Type')
        axes[0].set_ylabel('Deficit (mm)')
        axes[0].set_ylim(y_limits_volume)  # Apply global y-axis limits (to make sure same scale)

        sns.boxplot(
            data=duration_data_melted,
            x='Type',
            y='Deficit',
            ax=axes[1],
            palette={'Days_Below_Threshold_Human': 'lightcoral', 'Days_Below_Threshold_Natural': 'lightgreen'}
        )
        axes[1].set_title(f'{station_name} - Average Duration Deficit')
        axes[1].set_xlabel('Type')
        axes[1].set_ylabel('Days Below Threshold')
        axes[1].set_ylim(y_limits_duration)  # Apply global y-axis limits (to make sure same scale)

        # Save the plot as an image
        plot_file_path = os.path.join(output_plot_dir, f"{station_name}_aggregated_deficit_boxplot.png")
        plt.tight_layout()
        plt.savefig(plot_file_path)
        plt.close(fig)

        # Convert the image to Base64
        with open(plot_file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8'), mean_volume_human, mean_volume_natural, mean_duration_human, mean_duration_natural
    except Exception as e:
        print(f"Error generating plot for {station_name}: {e}")
        return None, None, None, None, None


# Add Shapefile 
def add_shapefile_to_map(gdf, layer_name, color, weight, folium_map):
    """Helper function to add GeoJSON to the map with style."""
    geojson_data = gdf.to_json()
    folium.GeoJson(
        geojson_data, 
        name=layer_name, 
        style_function=lambda x: {
            'color': color, 
            'weight': weight
        }
    ).add_to(folium_map)

# Create base map
m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=8)

for layer_name, shapefile in shapefiles.items():
    shapefile_path = os.path.join(shapefiles_dir, shapefile)
    if os.path.exists(shapefile_path):
        gdf = gpd.read_file(shapefile_path)
        color = 'blue' if layer_name == 'Meuse_Basin' else 'darkblue'
        weight = 2 if layer_name == 'Main_Rivers' else 4
        if layer_name == 'Dams_Reservoirs': 
            color = 'darkblue'
        add_shapefile_to_map(gdf, layer_name, color, weight, m)
import base64

def encode_image_as_base64(image_path):
    """Convert an image file to a base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#  Add Deficit Markers 
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    gauge_id = row['gauge_id']
    gauge_name = row.get('gauge_name', 'Unknown Station')
    river_name = row.get('river', 'Unknown River') 
    lat = row['lat']
    lon = row['lon']
    basin_id = row['basin_id']
    root_dep = row['root_dep']
    soil_tawc = row['soil_tawc']
    lit_dom = row['lit_dom']
    recession_time = row.get('Recession time (days)', np.nan)  
    
    # Handle NaN for recession_time
    recession_time_display = f"{round(recession_time)}" if not np.isnan(recession_time) else "N/A"
    
    # Convert volume values to 10^6 m³
    avg_baseflow_volume = row.get('Avg Baseflow Volume (m³)', 0) / 1000000  # Convert to 10^6 m³
    avg_total_volume = row.get('Avg Total Volume (m³)', 0) / 1000000  # Convert to 10^6 m³

    # Locate deficit files
    volume_file = os.path.join(discharge_dir, f"{gauge_id}_volume.txt")
    duration_file = os.path.join(discharge_dir, f"{gauge_id}_duration.txt")
    
    # Locate PNG file for the gauge (Human influence image) - Needs to be previously generated from human_influence.py
    png_file_path = os.path.join('D:/My Documents/LoFlowMaas/Codes_outputs/human', f"{gauge_id}.png")

    if os.path.exists(volume_file) and os.path.exists(duration_file):
        img_base64, mean_volume_human, mean_volume_natural, mean_duration_human, mean_duration_natural = generate_average_deficit_boxplot_aggregated(
            volume_file, duration_file, gauge_name, y_limits_volume, y_limits_duration
        )
        if img_base64:
            if os.path.exists(png_file_path):
                png_base64 = encode_image_as_base64(png_file_path)
            else:
                png_base64 = None  

            # Create HTML content for the popup - if add more attributes, add here too
            html = f"""
            <h4 style="font-size:18px; font-weight:bold; text-align:center;">{gauge_name} ({river_name}) - Gauge ID: {gauge_id}</h4>
            
            <div style="text-align:center; overflow-y:auto; max-height:650px;">
                <h5>Deficit</h5>
                <img src="data:image/png;base64,{img_base64}" width="700px" height="400px"> 
                <br><br>
                <h5>Human Influence</h5>
                {"<img src='data:image/png;base64," + png_base64 + "' width='700px' height='400px'>" if png_base64 else "<p>No Human Influence image available</p>"}
                <hr>
                <table style="width:100%; font-size:16px; border:1px solid #ccc; padding:12px; margin-top: 20px;">
                    <tr><td><strong>Average Baseflow Volume (10<sup>6</sup> m³)</strong></td><td>{round(avg_baseflow_volume, 2)}</td></tr>
                    <tr><td><strong>Recession time (days)</strong></td><td>{recession_time_display}</td></tr>
                    <tr><td><strong>Dominant Lithology</strong></td><td>{lit_dom}</td></tr>
                    <tr><td><strong>Root Depth (cm)</strong></td><td>{round(root_dep)}</td></tr>
                    <tr><td><strong>Soil TAWC (mm)</strong></td><td>{round(soil_tawc)}</td></tr>
                </table>
            </div>
            """
            iframe = folium.IFrame(html=html, width=1000, height=850) 
            popup = folium.Popup(iframe, max_width=1000)

            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
        else:
            print(f"Failed to generate plot for {gauge_name}.")
    else:
        print(f"Deficit files missing for Gauge ID {gauge_id}.")

# Save
folium.LayerControl().add_to(m)  
m.save(output_map_path) 
print(f"Map saved to {output_map_path}")

#%%