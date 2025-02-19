#%% Developed by Deborah, November 2024 - updated February 2025

import pandas as pd
import numpy as np
import os


discharge_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\interpolated\moving_average"
excel_file_path = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\Info_EStreams3.xlsx"
thresholds_folder = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\threshold\threshold_p"
deficit_seasonal_output_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\deficit_seasonal"

os.makedirs(deficit_seasonal_output_dir, exist_ok=True)

def load_basin_info(excel_file_path):
    """Loads gauge data from Excel."""
    try:
        basin_info = pd.read_excel(excel_file_path)
        return basin_info
    except Exception:
        return pd.DataFrame()

def discharge_m3s_to_mmday(discharge_m3_per_s, catchment_area_km2):
    """Convert discharge (m³/s) to mm/day given an area (km²)."""
    discharge_m3_per_s = pd.to_numeric(discharge_m3_per_s, errors="coerce")
    if catchment_area_km2 is None or catchment_area_km2 <= 0:
        return np.nan
    return (discharge_m3_per_s * 86400) / (catchment_area_km2 * 1e6) * 1000

def load_threshold_file(thresholds_folder, gauge_id):
    """Load threshold data for a given gauge_id."""
    threshold_file_path = os.path.join(thresholds_folder, f"{gauge_id}.txt")
    if not os.path.exists(threshold_file_path):
        raise FileNotFoundError(f"Threshold file not found for gauge {gauge_id}: {threshold_file_path}")
    threshold_data = pd.read_csv(
        threshold_file_path, delimiter=" ", header=None, names=["Date", "Threshold"]
    )
    threshold_data["Threshold"] = pd.to_numeric(threshold_data["Threshold"], errors="coerce")
    threshold_data["Date"] = pd.to_datetime(threshold_data["Date"], format="%Y%m%d", errors="coerce")
    threshold_data = threshold_data.dropna(subset=["Date"]).set_index("Date")["Threshold"]
    return threshold_data

def assign_season(dates):
    """Assign each date to a season label."""
    return (dates.month % 12 // 3 + 1).map({1: "DJF", 2: "MAM", 3: "JJA", 4: "SON"})

def calculate_seasonal_deficits(discharge, threshold, min_duration_days=10):
    """Compute seasonal deficits for a single timeseries."""
    discharge = pd.to_numeric(discharge, errors="coerce")
    threshold = pd.to_numeric(threshold, errors="coerce")
    discharge, threshold = discharge.align(threshold, join="inner")
    season = assign_season(discharge.index)
    deficit = (threshold - discharge).clip(lower=0)
    days_below = (discharge < threshold).astype(int)

    df = pd.DataFrame({
        "Year": discharge.index.year,
        "Season": season,
        "Deficit_mm": deficit,
        "Days_Below": days_below
    })

    # Filter for valid events
    valid_events = (
        df.groupby(["Year", "Season"])["Days_Below"].sum()
        .reset_index(name="Total_Days_Below")
        .query("Total_Days_Below >= @min_duration_days")  # Skip seasons with no valid data
    )
    df_filtered = df.merge(valid_events[["Year", "Season"]], on=["Year", "Season"])

    # Generate seasonal summary
    seasonal_summary = df_filtered.groupby(["Year", "Season"]).agg(
        Deficit_mm=("Deficit_mm", "sum"),
        Days_Below=("Days_Below", "sum")
    ).reset_index()

    # Ensure all seasons are represented for each year
    all_seasons = ["DJF", "MAM", "JJA", "SON"]
    seasonal_summary = seasonal_summary.pivot(index="Year", columns="Season", values=["Deficit_mm", "Days_Below"])
    seasonal_summary = seasonal_summary.reindex(columns=pd.MultiIndex.from_product([["Deficit_mm", "Days_Below"], all_seasons]), fill_value=0)
    seasonal_summary.columns = [f"{col[0]}_{col[1]}" for col in seasonal_summary.columns]
    seasonal_summary = seasonal_summary.reset_index()

    return seasonal_summary


    df = pd.DataFrame({
        "Year": discharge.index.year,
        "Season": season,
        "Deficit_mm": deficit,
        "Days_Below": days_below
    })

    # Filter for valid events
    valid_events = (
        df.groupby(["Year", "Season"])["Days_Below"].sum()
        .reset_index(name="Total_Days_Below")
        .query("Total_Days_Below >= @min_duration_days")  # Skip seasons with no valid data
    )
    df_filtered = df.merge(valid_events[["Year", "Season"]], on=["Year", "Season"])

    # Generate seasonal summary
    seasonal_summary = df_filtered.groupby(["Year", "Season"]).agg(
        Deficit_mm=("Deficit_mm", "sum"),
        Days_Below=("Days_Below", "sum")
    ).reset_index()

    seasonal_summary = seasonal_summary.pivot(index="Year", columns="Season", values=["Deficit_mm", "Days_Below"])
    seasonal_summary.columns = [f"{col[0]}_{col[1]}" for col in seasonal_summary.columns]
    seasonal_summary = seasonal_summary.reset_index().dropna(how="all", subset=["Deficit_mm_DJF", "Deficit_mm_MAM", "Deficit_mm_JJA", "Deficit_mm_SON"])

    return seasonal_summary

def merge_human_natural_seasonal(human_df, natural_df):
    """Merge two seasonal DataFrames (human, natural) side by side."""
    human_renamed = human_df.rename(columns=lambda x: x + "_Human" if x != "Year" else x)
    natural_renamed = natural_df.rename(columns=lambda x: x + "_Natural" if x != "Year" else x)
    merged = pd.merge(human_renamed, natural_renamed, on="Year", how="inner")  # Only keep years with data in both
    return merged

def save_seasonal_comparison_to_txt(df, gauge_id, output_dir):
    """Save the seasonal comparison to a text file."""
    file_path = os.path.join(output_dir, f"{gauge_id}_seasonal_comparison.txt")
    df.to_csv(file_path, sep="\t", index=False)
    print(f"Saved seasonal comparison for {gauge_id} at {file_path}")



def main():
    basin_info_df = load_basin_info(excel_file_path)

    for _, row in basin_info_df.iterrows():
        gauge_id = str(row["gauge_id"])
        human_area_km2 = row["area"] if not pd.isna(row["area"]) else row["area_calc"]
        paired_id = str(row.get("paired_id", ""))
        if not paired_id or paired_id.lower() == "nan":
            continue
        paired_ids = [pid.strip() for pid in paired_id.split(";") if pid.strip()]
        paired_areas = [float(a) for a in str(row["paired_area"]).split(";")]

        try:
            thresholds = load_threshold_file(thresholds_folder, gauge_id)
        except FileNotFoundError as e:
            print(e)
            continue

        human_path = os.path.join(discharge_dir, f"{gauge_id}.txt")
        if not os.path.exists(human_path):
            continue
        human_data = pd.read_csv(human_path, sep=" ", names=["Date", "Discharge"], header=None)
        human_data["Date"] = pd.to_datetime(human_data["Date"], format="%Y%m%d", errors="coerce")
        human_data = human_data.dropna(subset=["Date"]).set_index("Date")
        human_data["Discharge_mm_per_day"] = discharge_m3s_to_mmday(human_data["Discharge"], human_area_km2)
        human_seasonal = calculate_seasonal_deficits(human_data["Discharge_mm_per_day"], thresholds)

        combined_paired_data = []
        for pid, area in zip(paired_ids, paired_areas):
            paired_path = os.path.join(discharge_dir, f"{pid}.txt")
            if not os.path.exists(paired_path):
                continue
            paired_data = pd.read_csv(paired_path, sep=" ", names=["Date", "Discharge"], header=None)
            paired_data["Date"] = pd.to_datetime(paired_data["Date"], format="%Y%m%d", errors="coerce")
            paired_data = paired_data.dropna(subset=["Date"]).set_index("Date")
            paired_data["Discharge_mm_per_day"] = discharge_m3s_to_mmday(paired_data["Discharge"], area)
            paired_data["Weighted_Discharge"] = paired_data["Discharge_mm_per_day"] * area
            combined_paired_data.append(paired_data)

        if not combined_paired_data:
            print(f"No paired data available for Gauge ID {gauge_id}.")
            continue

        # Combine paired data and compute weighted average
        combined_paired_df = pd.concat(combined_paired_data, axis=0).groupby("Date").sum()
        combined_paired_df["Discharge_mm_per_day"] = (
            combined_paired_df["Weighted_Discharge"] / sum(paired_areas)
        )

        natural_seasonal = calculate_seasonal_deficits(combined_paired_df["Discharge_mm_per_day"], thresholds)

        # Merge and save
        comparison_df = merge_human_natural_seasonal(human_seasonal, natural_seasonal)
        save_seasonal_comparison_to_txt(comparison_df, gauge_id, deficit_seasonal_output_dir)

if __name__ == "__main__":
    main()


#%% Summary


import pandas as pd
import os

# Paths
deficit_seasonal_output_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\deficit_seasonal"
summary_output_file = os.path.join(deficit_seasonal_output_dir, "seasonal_summary.txt")

# Helper function to compute the averages per season per basin
def compute_summary_for_gauges(deficit_seasonal_output_dir):
    summary_data = []

    for file_name in os.listdir(deficit_seasonal_output_dir):
        if file_name.endswith("_seasonal_comparison.txt"):
            gauge_id = file_name.replace("_seasonal_comparison.txt", "")
            file_path = os.path.join(deficit_seasonal_output_dir, file_name)

            try:
                # Read the data
                df = pd.read_csv(file_path, sep="\t")

                # Ensure numeric values for computations
                for col in df.columns:
                    if col != "Year":
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Initialize summary row with NaN values for all seasons
                summary_row = {"Gauge_id": gauge_id}
                for measure in ["Deficit_mm", "Days_Below"]:
                    for season in ["DJF", "MAM", "JJA", "SON"]:
                        summary_row[f"{measure}_{season}_Human_Avg"] = None
                        summary_row[f"{measure}_{season}_Natural_Avg"] = None
                        summary_row[f"{season}_Deficit_Percent_Diff"] = None
                        summary_row[f"{season}_Days_Percent_Diff"] = None

                # Compute averages and percentage differences if columns exist
                for measure in ["Deficit_mm", "Days_Below"]:
                    for season in ["DJF", "MAM", "JJA", "SON"]:
                        human_col = f"{measure}_{season}_Human"
                        natural_col = f"{measure}_{season}_Natural"

                        if human_col in df.columns and natural_col in df.columns:
                            human_avg = df.loc[df[human_col] > 0, human_col].mean()
                            natural_avg = df.loc[df[natural_col] > 0, natural_col].mean()

                            summary_row[f"{measure}_{season}_Human_Avg"] = human_avg
                            summary_row[f"{measure}_{season}_Natural_Avg"] = natural_avg

                            # Calculate percentage difference
                            if pd.notnull(human_avg) and pd.notnull(natural_avg) and natural_avg > 0:
                                diff_key = f"{season}_{measure.split('_')[0]}_Percent_Diff"
                                summary_row[diff_key] = ((human_avg - natural_avg) / natural_avg) * 100

                summary_data.append(summary_row)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Convert the summary data to a DataFrame
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Save the summary data to a text file
def save_summary_to_txt(summary_df, output_file):
    with open(output_file, "w") as file:
        # Write the header
        header = (
            "Gauge_id\t"
            + "\t".join(
                [
                    f"{measure}_{season}_{type}_Avg"
                    for measure in ["Deficit_mm", "Days_Below"]
                    for season in ["DJF", "MAM", "JJA", "SON"]
                    for type in ["Human", "Natural"]
                ]
                + [
                    f"{season}_Deficit_Percent_Diff\t{season}_Days_Percent_Diff"
                    for season in ["DJF", "MAM", "JJA", "SON"]
                ]
            )
        )
        file.write(header + "\n")

        # Write each row of the summary DataFrame
        for _, row in summary_df.iterrows():
            line = "\t".join(str(row[col]) if pd.notnull(row[col]) else "NA" for col in summary_df.columns)
            file.write(line + "\n")

    print(f"Summary saved to {output_file}")

# Main function
def main():
    summary_df = compute_summary_for_gauges(deficit_seasonal_output_dir)
    save_summary_to_txt(summary_df, summary_output_file)

if __name__ == "__main__":
    main()


#%%  summary for specific stations
import pandas as pd
import os

# Paths
deficit_seasonal_output_dir = r"C:\Users\ddo001\Documents\LoFloMeuse\Discharge\deficit_seasonal"
summary_output_file = os.path.join(deficit_seasonal_output_dir, "seasonal_summary_aggravation_alleviation_selected.txt")

# Selected basins in the specified order
selected_basins = [
    "B315002001", "B460101001", "90211002", "94341002", "B720000002", "71321002", 
    "82211002", "L7319", "59211002", "62281002", "66211002", "Eijsden grens", 
    "6421500", "63400000", "6Q18", "10Q36", "2829100000100"
]

def compute_aggravation_alleviation(deficit_seasonal_output_dir, selected_basins):
    summary_data = []

    for basin_id in selected_basins:
        file_path = os.path.join(deficit_seasonal_output_dir, f"{basin_id}_seasonal_comparison.txt")
        if not os.path.exists(file_path):
            print(f"File not found for {basin_id}. Skipping...")
            continue

        try:
            # Read data
            df = pd.read_csv(file_path, sep="\t")

            # Ensure numeric values
            for col in df.columns:
                if col != "Year":
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Initialize summary row
            summary_row = {"Gauge_id": basin_id}

            # Compute signals for each season
            for season in ["DJF", "MAM", "JJA", "SON"]:
                human_col = f"Deficit_mm_{season}_Human"
                natural_col = f"Deficit_mm_{season}_Natural"

                if human_col in df.columns and natural_col in df.columns:
                    # Compute per-year percentage differences
                    df = df.dropna(subset=[human_col, natural_col])
                    if not df.empty:
                        df["Percent_Diff"] = ((df[human_col] - df[natural_col]) / df[natural_col]) * 100

                        # Filter valid events where the natural deficit is greater than zero
                        valid_diff = df[df[natural_col] > 0]["Percent_Diff"]

                        # Average the percentage differences across years
                        if not valid_diff.empty:
                            avg_percent_diff = valid_diff.mean()

                            # Determine signal based on average percentage difference
                            if avg_percent_diff < -5:
                                signal = "Alleviated"
                            elif avg_percent_diff > 5:
                                signal = "Aggravated"
                            else:
                                signal = "NoChange"
                            summary_row[f"{season}_Signal"] = signal
                        else:
                            summary_row[f"{season}_Signal"] = "NoChange"
                    else:
                        summary_row[f"{season}_Signal"] = "NoChange"
                else:
                    summary_row[f"{season}_Signal"] = "NA"

            summary_data.append(summary_row)

        except Exception as e:
            print(f"Error processing {basin_id}: {e}")

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Reorder by the specified order
    summary_df = summary_df.set_index("Gauge_id").reindex(selected_basins).reset_index()
    return summary_df


def save_summary_to_txt(summary_df, output_file):
    with open(output_file, "w") as file:
        # Write header
        header = "Gauge_id\tDJF_Signal\tMAM_Signal\tJJA_Signal\tSON_Signal"
        file.write(header + "\n")

        # Write each row
        for _, row in summary_df.iterrows():
            line = "\t".join(str(row[col]) if pd.notnull(row[col]) else "NA" for col in summary_df.columns)
            file.write(line + "\n")

    print(f"Summary saved to {output_file}")


def main():
    summary_df = compute_aggravation_alleviation(deficit_seasonal_output_dir, selected_basins)
    save_summary_to_txt(summary_df, summary_output_file)


if __name__ == "__main__":
    main()

#%%

