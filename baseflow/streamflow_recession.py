#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

# Function to calculate mean annual flow
def calculate_mean_annual_flow(data):
    return data.mean()

# Function to identify and merge close dry spells based on Q₀ and persistence threshold
def identify_and_merge_dry_spells(data, threshold, persistence_days=5, merge_gap=7):

    below_threshold = data < threshold
    persistent = below_threshold.rolling(window=persistence_days, center=False).sum() >= persistence_days
    dry_spell_start = persistent & ~persistent.shift(1).fillna(False)
    dry_spell_end = persistent & ~persistent.shift(-1).fillna(False)

    dry_spell_indices = list(zip(data.index[dry_spell_start], data.index[dry_spell_end]))
    merged_dry_spells = []
    current_start, current_end = None, None

    for start, end in dry_spell_indices:
        if current_start is None:
            current_start, current_end = start, end
        elif (start - current_end).days <= merge_gap:
            current_end = end
        else:
            merged_dry_spells.append((current_start, current_end))
            current_start, current_end = start, end

    if current_start is not None:
        merged_dry_spells.append((current_start, current_end))

    return merged_dry_spells

# Function to calculate Q1/2 (time to halve streamflow during recession)
def calculate_recession_period(data, start_date, end_date):
    Q0 = data.loc[start_date]
    Q_half = Q0 / 2

    for date in data.loc[start_date:end_date].index:
        if data.loc[date] <= Q_half:
            return date
    return end_date

# Function to plot dry spell analysis
def plot_dry_spell_analysis_with_recession(data, start_date, end_date, mean_flow, threshold, dry_spell_periods, gauge_id):
    filtered_data = data[(data.index >= start_date) & (data.index <= end_date)].copy()

    plt.figure(figsize=(10, 6))

    # Plot streamflow on log scale
    plt.plot(filtered_data.index, filtered_data['Discharge'], label='Streamflow (Q)', color='black', linewidth=1)

    # Plot mean annual flow
    plt.axhline(y=mean_flow, color='dodgerblue', linestyle='--', label=f'Mean Annual Flow ({mean_flow:.2f} mm/day)')

    # Plot dry spell regions
    for start_idx, end_idx in dry_spell_periods:
        if start_idx in filtered_data.index and end_idx in filtered_data.index:
            # Add shaded dry spell
            plt.axvspan(start_idx, end_idx, color='lightgray', alpha=0.5, label='Dry Spell Period' if 'Dry Spell Period' not in plt.gca().get_legend_handles_labels()[1] else "")

            # Calculate Qmin and Q1/2
            Qmin = filtered_data.loc[start_idx:end_idx, 'Discharge'].min()
            Qmin_idx = filtered_data.loc[start_idx:end_idx, 'Discharge'].idxmin()
            Q_half_date = calculate_recession_period(filtered_data['Discharge'], start_idx, end_idx)

            # Mark Q₀  and Qmin (red)
            Q0 = filtered_data.loc[start_idx, 'Discharge']
            plt.scatter(start_idx, Q0, color='dodgerblue', label='Q₀ (Initial Flow)' if 'Q₀ (Initial Flow)' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.scatter(Qmin_idx, Qmin, color='red', label='Qₘₑₓ (Minimum Flow)' if 'Qₘₑₓ (Minimum Flow)' not in plt.gca().get_legend_handles_labels()[1] else "")

            # Plot Q1/2 line
            plt.plot(
                [start_idx, Qmin_idx],
                [Q0, Qmin],
                color='red', linestyle='-', linewidth=1, label='Q₁/₂ (Recession)' if 'Q₁/₂ (Recession)' not in plt.gca().get_legend_handles_labels()[1] else ""
            )

            # Add half-period annotation
            if start_idx <= Q_half_date <= end_idx:
                num_days = (Q_half_date - start_idx).days
                plt.annotate(
                    f'{num_days} days',
                    xy=((start_idx + (Q_half_date - start_idx) / 2), Q0 / 2),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9
                )
                plt.annotate(
                    '',
                    xy=(start_idx, Q0 / 2),
                    xytext=(Q_half_date, Q0 / 2),
                    arrowprops=dict(arrowstyle='<->', color='black')
                )

    # Plot 
    plt.title(f'Dry Spell Analysis for Station {gauge_id}')
    plt.xlabel('Date')
    plt.ylabel('7-day Q (mm/day)')
    plt.yscale('log')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.show()

# Main function to process the file
def process_dry_spell_data(directory_path, gauge_id, start_date, end_date):
    file_path = os.path.join(directory_path, gauge_id + '.txt')

    # Load data
    data = pd.read_csv(file_path, delimiter=' ', header=0, names=['Date', 'Discharge'])
    data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
    data.set_index('Date', inplace=True)
    data = data.dropna(subset=['Discharge'])

    # Smooth the discharge data (optional, 30-day moving average)
    data['Smoothed_Discharge'] = data['Discharge'].rolling(window=30, center=True).mean()

    # Calculate mean annual flow
    mean_flow = calculate_mean_annual_flow(data['Smoothed_Discharge'])

    # Set 80% non-exceedance threshold
    threshold = np.percentile(data['Smoothed_Discharge'].dropna(), 20)

    # Identify dry spells and recession
    dry_spell_periods = identify_and_merge_dry_spells(data['Smoothed_Discharge'], threshold, persistence_days=5, merge_gap=7)

    # Plot data
    plot_dry_spell_analysis_with_recession(data, start_date, end_date, mean_flow, threshold, dry_spell_periods, gauge_id)

# Directory and parameters
directory_path = r'D:\My Documents\LoFlowMaas\Discharge\All'
gauge_id = '2829100000100'  # Replace with your gauge ID
start_date = '2015-04-01'
end_date = '2015-10-31'

# Run
process_dry_spell_data(directory_path, gauge_id, start_date, end_date)

#%%