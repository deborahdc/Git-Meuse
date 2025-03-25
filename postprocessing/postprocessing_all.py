#%%
# 📌 IMPORT LIBRARIES AND SETUP
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import time
from tqdm import tqdm
from datetime import datetime

# Define working directory
work_dir = r"C:/Users/ddo001/Documents/LoFloMeuse/Model_results"

# Define file paths
file_path = os.path.join(work_dir, "all_20250317_combined.nc")
stations_file = os.path.join(work_dir, "stations.yml")
catalog_file = os.path.join(work_dir, "run_catalog.yml")

# Create output directories for saving plots
plot_dir = os.path.join(work_dir, "NEW_PLOTS")
heatmap_dir = os.path.join(plot_dir, "heatmaps")
hydrograph_dir = os.path.join(plot_dir, "hydrographs")
climatology_dir = os.path.join(plot_dir, "climatology")
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(hydrograph_dir, exist_ok=True)
os.makedirs(climatology_dir, exist_ok=True)

# Track execution time
start_time = time.time()

#%%
# 📌 LOAD DATASETS (NetCDF, Stations, and Run Catalog)
print("Loading dataset:", file_path)
ds = xr.open_dataset(file_path)

# Load stations metadata
with open(stations_file, "r") as f:
    stations = yaml.safe_load(f)["stations"]

# Load scenario details from run catalog
with open(catalog_file, "r") as f:
    run_catalog = yaml.safe_load(f)

# Extract scenario names
scenarios = run_catalog["call_all"]

# Fix: Map scenario aliases to colors properly
scenario_styles = {}
for key, val in run_catalog.items():
    if key not in ["meta", "call_all"]:  # Ignore metadata keys
        alias = val.get("alias", key)  # Use alias if available
        color = val.get("color", "black")  # Default to black if missing
        scenario_styles[alias] = color

#%%
# 📌 REMOVE WARM-UP YEARS
def discard_warmup(combined, n_discarded: int = 3):
    """Removes the first 'n_discarded' years from the dataset to exclude the spin-up period."""
    years = combined.time.dt.year
    warmup_years = years[:n_discarded]
    return combined.sel(time=~combined.time.dt.year.isin(warmup_years))

ds = discard_warmup(ds, n_discarded=3)

#%%
# 📌 COMPUTE WET AND DRY SEASON DISCHARGE
months_dry = list(range(4, 10))  # April to September
months_wet = [i for i in range(1, 13) if i not in months_dry]

def calc_Qwet(ds, wet_months):
    wet_mask = ds['time.month'].isin(wet_months)
    return ds.where(wet_mask, drop=True).mean(dim='time')

def calc_Qdry(ds, dry_months):
    dry_mask = ds['time.month'].isin(dry_months)
    return ds.where(dry_mask, drop=True).mean(dim='time')

#%%
# 📌 GENERATE HEATMAPS FOR SCENARIOS AT EACH STATION
def plot_heatmap(combined_ds, station_name, station_index, reference_run='ref', vmin=-20, vmax=20, cmap="RdBu"):
    """Generates and saves a heatmap of scenario differences for a specific station."""
    variables = list(combined_ds.data_vars) + ["Qwet", "Qdry"]
    abs_dict = {}

    # Compute Qwet and Qdry
    Q_data = combined_ds.sel(index=str(station_index)).Q
    abs_dict["Qwet"] = calc_Qwet(Q_data, months_wet).values
    abs_dict["Qdry"] = calc_Qdry(Q_data, months_dry).values

    # Compute mean values for other variables
    for var in variables:
        if var not in ["Qwet", "Qdry"]:
            abs_dict[var] = combined_ds[var].sel(index=str(station_index)).mean(dim='time').values

    # Create DataFrame
    abs_df = pd.DataFrame(abs_dict, index=scenarios)

    # Compute relative change from reference run
    ref_row = abs_df.loc[reference_run]
    rel_df = ((abs_df / ref_row) - 1) * 100

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(rel_df.clip(vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, center=0, annot=True, fmt=".1f", linewidths=0.5, ax=ax)
    ax.set_title(f'Heatmap at {station_name} (Index {station_index})')
    plt.xlabel('Variables')
    plt.ylabel('Scenarios')

    # Save plot
    plot_filename = os.path.join(heatmap_dir, f"heatmap_{station_name.replace(' ', '_')}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    return plot_filename

#%%
# 📌 GENERATE HYDROGRAPHS
def plot_hydrograph(combined_ds, station_name, station_index):
    """Generates and saves a time-series hydrograph for a specific station."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for scenario in scenarios:
        Q_data = combined_ds.Q.sel(index=str(station_index), runs=scenario)
        ax.plot(Q_data.time, Q_data, label=scenario, color=scenario_styles.get(scenario, "black"))

    ax.set_title(f"Hydrograph at {station_name} (Index {station_index})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge (m³/s)")
    ax.legend()
    ax.grid()

    # Save plot
    plot_filename = os.path.join(hydrograph_dir, f"hydrograph_{station_name.replace(' ', '_')}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    return plot_filename

#%%
# 📌 GENERATE MONTHLY CLIMATOLOGY PLOTS
def plot_climatology(combined_ds, station_name, station_index):
    """Generates and saves a monthly climatology plot for a specific station."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario in scenarios:
        Q_data = combined_ds.Q.sel(index=str(station_index), runs=scenario)
        monthly_means = Q_data.groupby('time.month').mean()
        ax.plot(monthly_means.month, monthly_means, label=scenario, marker="o", color=scenario_styles.get(scenario, "black"))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title(f"Monthly Climatology at {station_name} (Index {station_index})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Discharge (m³/s)")
    ax.legend()
    ax.grid()

    # Save plot
    plot_filename = os.path.join(climatology_dir, f"climatology_{station_name.replace(' ', '_')}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    return plot_filename

#%%
# 📌 PROCESS ALL STATIONS AND SAVE SUMMARY
summary_data = []
for station_name, station_index in tqdm(stations.items(), desc="Processing Stations"):
    heatmap_file = plot_heatmap(ds, station_name, station_index)
    hydrograph_file = plot_hydrograph(ds, station_name, station_index)
    climatology_file = plot_climatology(ds, station_name, station_index)
    summary_data.append([station_name, station_index, heatmap_file, hydrograph_file, climatology_file])

print("\n✅ All analyses completed")
#%% 
