#%%
import rasterio
from rasterio.enums import Resampling
import numpy as np
import os

# File paths
slope_raster_fp = r"C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved\NBS\Grouped_layers\grouped_slope.tif"
aquifer_raster_fp = r"C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved\NBS\Grouped_layers\grouped_aquifer.tif"
soil_raster_fp = r"C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved\NBS\Grouped_layers\grouped_soil_texture2.tif"
landuse_raster_fp = r"C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved\NBS\Grouped_layers\grouped_land_cover2.tif"
natura2000_raster_fp = r"C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved\NBS\Grouped_layers\natura2000.tif"
hand_raster_fp = r"C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved\NBS\Grouped_layers\grouped_hand.tif"

output_raster_directory = r"C:\Users\ddo001\Documents\LoFloMeuse\GIS\Data\Saved\NBS"

# Step 1: Load and resample raster data
def load_raster(raster_fp, reference_shape=None, reference_transform=None):
    with rasterio.open(raster_fp) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

        # Resample if necessary
        if reference_shape and data.shape != reference_shape:
            data_resampled = np.empty(reference_shape, dtype=data.dtype)
            with rasterio.open(raster_fp) as dataset:
                dataset.read(1, out=data_resampled, resampling=Resampling.nearest)
            return data_resampled, reference_transform, crs
        
        return data, transform, crs

# Load the reference raster (use slope as reference)
slope_data, slope_transform, slope_crs = load_raster(slope_raster_fp)
reference_shape = slope_data.shape
reference_transform = slope_transform

# Load and resample other rasters
aquifer_data, _, _ = load_raster(aquifer_raster_fp, reference_shape, reference_transform)
soil_data, _, _ = load_raster(soil_raster_fp, reference_shape, reference_transform)
landuse_data, _, _ = load_raster(landuse_raster_fp, reference_shape, reference_transform)
natura2000_data, _, _ = load_raster(natura2000_raster_fp, reference_shape, reference_transform)
hand_data, _, _ = load_raster(hand_raster_fp, reference_shape, reference_transform)

# Step 2: Check suitability - adapt for different NbS
def check_suitability(slope_data, aquifer_data, soil_data, landuse_data, natura2000_data, hand_data):
    suitability = (
        (np.isin(slope_data, [1,2,3,4])) &  # Suitable slope classes
        (np.isin(aquifer_data, [1,2,4,6])) &  # Suitable aquifer types
        (np.isin(soil_data, [1, 2, 3, 4, 5])) &  # Suitable soil types
        (np.isin(landuse_data, [2,3,5]))   # Suitable land use
        & (natura2000_data == 0) # Exclude Natura 2000 areas
        #& (hand_data == 1)  # Consider only areas with HAND value 1
    )
    return suitability

# Step 3: Calculate suitability
suitability = check_suitability(slope_data, aquifer_data, soil_data, landuse_data, natura2000_data, hand_data)

# Step 4: Save the suitability result as a raster
output_raster_fp = os.path.join(output_raster_directory, "suitability_drainagemod.tif")

# Ensure suitability is in int format (1 for suitable, 0 for unsuitable)
suitability_int = suitability.astype(np.int32)

# Open the output raster file for writing
with rasterio.open(output_raster_fp, 'w', driver='GTiff', height=suitability_int.shape[0], 
                   width=suitability_int.shape[1], count=1, dtype=suitability_int.dtype, 
                   crs=slope_crs, transform=slope_transform) as dst:
    dst.write(suitability_int, 1)
#%%


