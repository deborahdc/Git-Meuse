#%% 

#%%
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets.embed import embed_minimal_html

# === 📂 File paths ===
file_combined = r"C:\Users\ddo001\Documents\LoFloMeuse\Model_results\all_20250317_combined.nc"
file_discharge = r"C:\Users\ddo001\Documents\LoFloMeuse\Model_results\other\discharge_hourlyobs_smoothed.nc"

# === 📦 Load data ===
ds_combined = xr.open_dataset(file_combined)
ds_discharge = xr.open_dataset(file_discharge)
time_combined = pd.to_datetime(ds_combined.time.values)
time_discharge = pd.to_datetime(ds_discharge.time.values)

# === 🔧 Interface widgets ===
all_indices = list(ds_combined.index.values.astype(str))
all_runs = list(ds_combined.runs.values.astype(str))
all_vars = list(ds_combined.data_vars.keys())

index_dropdown = widgets.Dropdown(options=all_indices, description="Index:")
runs_select = widgets.SelectMultiple(options=all_runs, value=("ref",), description="Runs:", layout={'height': '150px'})
vars_select = widgets.SelectMultiple(options=all_vars, value=("Q",), description="Variables:", layout={'height': '200px'})

month_options = [t.strftime('%Y-%m') for t in pd.date_range(time_combined[0], time_combined[-1], freq='MS')]
time_range_slider = widgets.SelectionRangeSlider(
    options=month_options,
    index=(0, len(month_options) - 1),
    description='Time Range:',
    layout={'width': '95%', 'height': '50px'}
)

include_qobs = widgets.Checkbox(value=True, description='Include Q_obs')
include_qhbv = widgets.Checkbox(value=True, description='Include Q_hbv')

plot_btn = widgets.Button(description='Generate Time Series')
heatmap_btn = widgets.Button(description='Generate Heatmap')

output = widgets.Output()

# === 🎨 Plot functions ===
def plot_timeseries(index, runs, variables, t_start, t_end, include_qobs, include_qhbv):
    with output:
        clear_output()
        print("Generating time series plot...")

        i_station = int((ds_combined.index == index).argmax().item())
        t0 = pd.to_datetime(t_start + "-01")
        t1 = pd.to_datetime(t_end + "-01") + pd.offsets.MonthEnd(1)

        fig, ax_left = plt.subplots(figsize=(18, 6))
        ax_right = ax_left.twinx()

        # === Plot discharge obs/hbv if selected ===
        i_q = int((ds_discharge.wflow_id == int(index)).argmax().item())
        df_q = pd.DataFrame({
            'Q_hbv': ds_discharge.Q[:, i_q, 0].values,
            'Q_obs': ds_discharge.Q[:, i_q, 1].values
        }, index=time_discharge).resample("D").mean().loc[t0:t1]

        max_left, max_right = 0, 0

        if include_qobs:
            ax_left.plot(df_q.index, df_q["Q_obs"], label="Q_obs (run=2)", color="black", linewidth=1)
            max_left = max(max_left, df_q["Q_obs"].max())
        if include_qhbv:
            ax_left.plot(df_q.index, df_q["Q_hbv"], label="Q_hbv (run=1)", color="green", linewidth=1)
            max_left = max(max_left, df_q["Q_hbv"].max())

        # === Plot selected variables for selected runs ===
        colors = plt.cm.tab10.colors
        color_map = {
            'Pr': 'steelblue', 'Ea': 'orange', 'Q': 'darkred', 'Qgwr': 'blue', 'Qinf': 'darkgreen',
            'Qof': 'dodgerblue', 'Qssf': 'purple', 'Et': 'darkorange', 'Sr': 'brown', 'St': 'teal'
        }

        for var in variables:
            for i, run in enumerate(runs):
                if run not in ds_combined.runs.values:
                    continue
                i_run = int((ds_combined.runs == run).argmax().item())
                data = ds_combined[var][:, i_station, i_run].to_series().loc[t0:t1]

                color = color_map.get(var, colors[i % len(colors)])
                label = f"{var} ({run})"

                if var == "Pr":
                    ax_right.bar(data.index, data, label=label, color=color, width=1.0)
                    max_right = max(max_right, data.max())
                elif var in ["Ea", "Et", "Qinf", "Qgwr"]:
                    ax_right.plot(data.index, data, label=label, color=color, linestyle="--")
                    max_right = max(max_right, data.max())
                else:
                    ax_left.plot(data.index, data, label=label, color=color, linestyle=":")
                    max_left = max(max_left, data.max())

        ax_left.set_ylabel("Discharge / Flow (m³/s or m³/day)")
        ax_right.set_ylabel("Precipitation / Evaporation / Recharge (mm/day)")
        ax_left.set_ylim(0, 1.5 * max(1, max_left))
        ax_right.set_ylim(1.5 * max(1, max_right), 0)

        fig.suptitle(f"Hydro-Meteo Combined Plot | wflow_id={index} | Runs: {', '.join(runs)} | {t_start} to {t_end}", fontsize=14)
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

        plt.tight_layout()
        plt.show()


def plot_heatmap(index, runs, variables, t_start, t_end):
    with output:
        clear_output()
        print("Generating heatmap...")

        i_station = int((ds_combined.index == index).argmax().item())
        t0 = pd.to_datetime(t_start + "-01")
        t1 = pd.to_datetime(t_end + "-01") + pd.offsets.MonthEnd(1)

        means = {}
        for run in runs:
            i_run = int((ds_combined.runs == run).argmax().item())
            run_means = []
            for var in variables:
                data = ds_combined[var][:, i_station, i_run].to_series().loc[t0:t1]
                run_means.append(data.mean())
            means[run] = run_means

        df_means = pd.DataFrame(means, index=variables).T

        if "ref" in df_means.index:
            df_pct_change = (df_means - df_means.loc["ref"]) / df_means.loc["ref"] * 100
        else:
            df_pct_change = df_means.copy()

        plt.figure(figsize=(14, 6))
        sns.heatmap(df_pct_change, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar_kws={'label': '% Change from ref'})
        plt.title(f"% Change from ref at index {index}\n(mean from {t_start} to {t_end})")
        plt.ylabel("Run")
        plt.xlabel("Variable")
        plt.tight_layout()
        plt.show()


# === Connect buttons ===
def on_plot_clicked(b):
    plot_timeseries(index_dropdown.value, list(runs_select.value), list(vars_select.value), time_range_slider.value[0], time_range_slider.value[1], include_qobs.value, include_qhbv.value)

def on_heatmap_clicked(b):
    plot_heatmap(index_dropdown.value, list(runs_select.value), list(vars_select.value), time_range_slider.value[0], time_range_slider.value[1])

plot_btn.on_click(on_plot_clicked)
heatmap_btn.on_click(on_heatmap_clicked)

# === Show interface ===
display(widgets.HBox([index_dropdown, runs_select, vars_select]))
display(time_range_slider)
display(widgets.HBox([plot_btn, heatmap_btn, include_qobs, include_qhbv]))
display(output)

# === Export as HTML (optional) ===
embed_minimal_html('export.html', views=[index_dropdown, runs_select, vars_select, time_range_slider, plot_btn, heatmap_btn, include_qobs, include_qhbv, output], title='Hydro-Meteo Analysis')
#%%
