#%% Developed by Deborah, 12h of November 2024
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_low_flow_analysis_plot_with_paired(
    gauge_id, directory_path, excel_file_path, threshold_p_dir, p_factor_dir, start_date=None, end_date=None
):
    # Load gauge info
    gauge_info = pd.read_excel(excel_file_path)
    gauge_info['gauge_id'] = gauge_info['gauge_id'].astype(str)  
    gauge_row = gauge_info[gauge_info["gauge_id"] == gauge_id]

    if gauge_row.empty:
        print(f"Gauge ID {gauge_id} not found in the Excel file.")
        return

    paired_ids = str(gauge_row["paired_id"].values[0]).split(";")
    paired_areas = [float(a) for a in str(gauge_row["paired_area"].values[0]).split(";")]
    catchment_area_km2 = float(gauge_row["area"].values[0])
    station_name = gauge_row["gauge_name"].values[0]

    # Load observed discharge (needs to convert to mm/day)
    observed_file = os.path.join(directory_path, f"{gauge_id}.txt")
    try:
        observed_data = pd.read_csv(
            observed_file, sep=" ", header=None, names=["yyyymmdd", "Discharge"], skiprows=1
        )
        observed_data["Date"] = pd.to_datetime(observed_data["yyyymmdd"], format="%Y%m%d", errors="coerce")
        observed_data.set_index("Date", inplace=True)
        observed_data["Discharge_mm_per_day"] = (observed_data["Discharge"] * 86400) / (catchment_area_km2 * 1e6) * 1000
    except Exception as e:
        print(f"Error loading observed data for {gauge_id}: {e}")
        return

    # Load thresholds (already in mm/day)
    threshold_file = os.path.join(threshold_p_dir, f"{gauge_id}.txt")
    try:
        threshold_data = pd.read_csv(threshold_file, sep=" ", header=None, names=["yyyymmdd", "Threshold"])
        threshold_data["Date"] = pd.to_datetime(threshold_data["yyyymmdd"], format="%Y%m%d", errors="coerce")
        threshold_data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Error loading threshold data for {gauge_id}: {e}")
        return

    # Load p_factor data (used to correct paired data)
    p_factor_file = os.path.join(p_factor_dir, f"{gauge_id}.txt")
    try:
        p_factor_data = pd.read_csv(p_factor_file, sep=" ", header=None, names=["yyyymmdd", "p_factor"])
        p_factor_data["Date"] = pd.to_datetime(p_factor_data["yyyymmdd"], format="%Y%m%d", errors="coerce")
        p_factor_data.dropna(subset=["Date"], inplace=True)  # Drop invalid dates
        p_factor_data.set_index("Date", inplace=True)
    except Exception as e:
        print(f"Error loading p_factor data for {gauge_id}: {e}")
        return

    # Combine paired data (in case there is more than one)
    paired_combined = []
    for paired_id, paired_area in zip(paired_ids, paired_areas):
        paired_file = os.path.join(directory_path, f"{paired_id}.txt")
        try:
            paired_data = pd.read_csv(
                paired_file, sep=" ", header=None, names=["yyyymmdd", "Discharge"], skiprows=1
            )
            paired_data["Date"] = pd.to_datetime(paired_data["yyyymmdd"], format="%Y%m%d", errors="coerce")
            paired_data.set_index("Date", inplace=True)
            paired_data["Discharge_mm_per_day"] = (paired_data["Discharge"] * 86400) / (paired_area * 1e6) * 1000
            paired_data["Weighted_Discharge"] = paired_data["Discharge_mm_per_day"] * paired_area
            paired_combined.append(paired_data)
        except Exception as e:
            print(f"Error processing paired basin {paired_id}: {e}")

    if not paired_combined:
        print("No paired data was loaded.")
        return

    paired_combined_df = pd.concat(paired_combined)
    weighted_paired = (
        paired_combined_df.groupby(paired_combined_df.index)["Weighted_Discharge"].sum() / sum(paired_areas)
    ).to_frame(name="Discharge_mm_per_day")

    # Apply p_factor
    weighted_paired = weighted_paired.join(p_factor_data["p_factor"], how="left")
    weighted_paired["Adjusted_Discharge_mm_per_day"] = (
        weighted_paired["Discharge_mm_per_day"] * weighted_paired["p_factor"].fillna(1)
    )

    # Filter data for the specified time period
    if start_date:
        start_date = pd.to_datetime(start_date)
    if end_date:
        end_date = pd.to_datetime(end_date)

    # Find the common date range among the three datasets
    common_dates = observed_data.index.intersection(threshold_data.index).intersection(weighted_paired.index)

    # Reindex all datasets to the common date range
    observed_data = observed_data.reindex(common_dates)
    threshold_data = threshold_data.reindex(common_dates)
    weighted_paired = weighted_paired.reindex(common_dates)

    # Replace missing values with NaN
    observed_data["Discharge_mm_per_day"] = observed_data["Discharge_mm_per_day"].fillna(np.nan)
    weighted_paired["Adjusted_Discharge_mm_per_day"] = weighted_paired["Adjusted_Discharge_mm_per_day"].fillna(np.nan)
    threshold_data["Threshold"] = threshold_data["Threshold"].fillna(np.nan)

    # Check if aligned data is empty
    if observed_data.empty or threshold_data.empty or weighted_paired.empty:
        print("Aligned data contains empty DataFrames.")
        return

    # Plotting with missing data handled
    fig, ax = plt.subplots(figsize=(20, 10))

    # Filter out rows where any of the data columns are NaN
    merged_data = pd.concat([observed_data["Discharge_mm_per_day"], weighted_paired["Adjusted_Discharge_mm_per_day"], threshold_data["Threshold"]], axis=1)
    merged_data.columns = ["Observed", "Natural", "Threshold"]
    merged_data = merged_data.dropna()  

    # Plot
    ax.plot(merged_data.index, merged_data["Observed"], label="Observed discharge", color="black", linewidth=2)
    ax.plot(merged_data.index, merged_data["Natural"], label="Natural discharge", color="gray", linewidth=2)
    ax.plot(merged_data.index, merged_data["Threshold"], label="Threshold (natural)", color="black", linestyle="--", linewidth=2)

    # Add the decision threshold it exists in mm/day to the plot
    if "decision_threshold_m3s" in gauge_info.columns and not pd.isna(gauge_row["decision_threshold_m3s"].values[0]):
        decision_threshold_m3s = float(gauge_row["decision_threshold_m3s"].values[0])
        decision_threshold_mm_day = (decision_threshold_m3s * 86400) / (catchment_area_km2 * 1e6) * 1000
        ax.axhline(
            y=decision_threshold_mm_day,
            color="brown",
            linestyle="dotted",
            linewidth=2,
            label=f"Decision threshold ({decision_threshold_m3s:.0f} m³/s)"
        )

    # Add hydrological year markers
    for year in merged_data.index.year.unique():
        hydrological_year_start = pd.Timestamp(year=year, month=10, day=1)
        ax.axvline(x=hydrological_year_start, color="grey", linestyle="--", linewidth=0.5, label="Start hydrological year" if year == merged_data.index.year.unique()[0] else "")

    # 100% Human Alleviated
    ax.fill_between(merged_data.index, -0.5, -0.25, 
                    where=(merged_data['Observed'] >= merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Threshold']),
                    color="#009682", label='100% Human Alleviated', step='mid')

    # Human Alleviated
    ax.fill_between(merged_data.index, -0.5, -0.25,
                    where=(
                        (merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Observed']) 
                    ),
                    color="#83D3C2", label='Human Alleviated', step='mid')

    # Human Aggravated
    ax.fill_between(merged_data.index, -0.5, -0.25,
                    where=(
                        (merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] >= merged_data['Observed']) 
                    ),
                    color="#E6AC00", label='Human Aggravated', step='mid')

    # 100% Human Aggravated
    ax.fill_between(merged_data.index, -0.5, -0.25, 
                    where=(merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] >= merged_data['Threshold']),
                    color="#803300", label='100% Human Aggravated', step='mid')
    
    # Adding fill between


    ax.fill_between(merged_data.index, merged_data['Natural'], merged_data['Observed'], 
                    where=(merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Observed']),
                    color="grey", interpolate=True)

    ax.fill_between(merged_data.index, merged_data['Observed'], merged_data['Threshold'], 
                    where=(merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Observed']),
                    color="#83D3C2", interpolate=True)

    ax.fill_between(merged_data.index, merged_data['Natural'], merged_data['Observed'], 
                    where=(merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Threshold']) & 
                        (merged_data['Observed'] < merged_data['Natural']),
                    color="#E6AC00", interpolate=True)

    ax.fill_between(merged_data.index, merged_data['Natural'], merged_data['Threshold'], 
                    where=(merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] < merged_data['Threshold']) & 
                        (merged_data['Observed'] < merged_data['Natural']),
                    color="grey", interpolate=True, label='Climate-induced')

    ax.fill_between(merged_data.index, merged_data['Observed'], merged_data['Threshold'], 
                    where=(merged_data['Observed'] < merged_data['Threshold']) & 
                        (merged_data['Natural'] >= merged_data['Threshold']),
                    color="#803300", interpolate=True)

    ax.fill_between(merged_data.index, merged_data['Natural'], merged_data['Threshold'], 
                    where=(merged_data['Natural'] < merged_data['Threshold']) & 
                        (merged_data['Observed'] >= merged_data['Threshold']),
                    color="#009682", interpolate=True)


    # Title and labels
    title_info = f'{station_name} |  Area: {int(catchment_area_km2)} km²'
    ax.set_title(title_info, fontsize=22)
    ax.set_xlabel('Date', fontsize=18)
    if start_date and end_date:
        ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
    ax.set_ylabel('Discharge (mm/day)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16, loc='upper right')

    plt.tight_layout()

    # Save the plot as a PNG file
    output_directory = r"C:\Users\ddo001\Documents\LoFloMeuse\Codes_outputs\human"
    output_filename = os.path.join(output_directory, f"{gauge_id}.png")
    #fig.savefig(output_filename, dpi=300, bbox_inches="tight")

    plt.show()

# Example (to zoom in and inspect)
gauge_id = '2829100000100'
directory_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated\moving_average'
excel_file_path = r'C:\Users\ddo001\Documents\LoFloMeuse\Discharge\Info_EStreams3.xlsx'
threshold_p_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\threshold\threshold_p"
p_factor_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\threshold\p_factor"
start_date = '2002-10-01'
end_date = '2004-09-30'

generate_low_flow_analysis_plot_with_paired(
    gauge_id=gauge_id,
    directory_path=directory_path,
    excel_file_path=excel_file_path,
    threshold_p_dir=threshold_p_dir,
    p_factor_dir=p_factor_dir,
    start_date=start_date,
    end_date=end_date
)

#%%