#%%














#%%
from result_reader import ResultReader
from result_plotter import plot_table_statistics_combined
import os 
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
from tqdm import tqdm


work_dir = r"."
os.chdir(work_dir)

# Initialize the reader
reader = ResultReader("./run_catalog.yml")
months_dry = list(range(4, 10))  
months_wet = [i for i in range(1, 13) if i not in months_dry]
# specify the scenarios to combine
# scenarios = reader.catalog['call_all']
# today = dt.datetime.now().strftime("%Y%m%d")
# outname = f"all_{today}"
# out_ds = f"{outname}_combined.nc"
# outpath = os.path.join(work_dir, out_ds)

# #trigger the combine function
# combined_datasets = reader.combine_grouped_datasets(keys=scenarios, 
#                                                     Q_combine=['Q_Sall', 'Q_extra'],
#                                                     outpath=outpath,
#                                                     overwrite=False)

#Performance metrics for combine_grouped_datasets:
# Time elapsed: 3382.73 seconds
# Memory usage: 3281.69 MB
# Final memory: 3561.43 MB
def discard_warmup(combined, n_discarded:int=3):
    years = combined.time.dt.year
    warmup_years = years[:n_discarded]
    combined = combined.sel(time=~combined.time.dt.year.isin(warmup_years))
    return combined

combined_ds = xr.open_dataset(os.path.join(work_dir, "all_20250317_combined.nc"))

combined = discard_warmup(combined_ds, n_discarded=3)
#%%

# Example usage
plot_dir = os.path.join(work_dir, "NEW_PLOTS", "heatmaps")
os.makedirs(plot_dir, exist_ok=True)

for index in tqdm(['16'], desc="Generating combined heatmaps", total=1, colour="green"):
    outname = f"heatmap_combined_index_{index}"
    fig, ax = plot_table_statistics_combined(
        combined,
        # variables=variables_subset, #some
        variables=None, #all
        months_dry=months_dry,
        months_wet=months_wet,
        reader=reader,
        reference_run='ref',
        index=str(index),
        vmin=-20,
        vmax=20,
        # outpath=os.path.join(plot_dir, outname), #to save
        outpath=None, #to show
        # invert_cmap_for=['Ea','Ei', 'Eow', 'Es', 'Et']
    )


#%%
print("Cells below are extras/unfinished testing hydrographs, monthly vars etc.")
#%%
# def find_annual_minima(Q):
#     Qmins = Q.groupby('time.year').min()
#     Qminsdf = Qmins.to_dataframe()["Q"]
#     ranked = Qminsdf.sort_values(ascending=True)
#     ranked = pd.DataFrame(ranked)
#     ranked['rank'] = ranked.Q.rank(method='min')
#     return ranked

# Qminsdf = find_annual_minima(combined.Q.sel(index=str(16), runs='ref'))
# top_5_dry_years = Qminsdf.index.values[:5]
# print(top_5_dry_years)

# #%%

# def plot_hydrograph(combined, 
#                     reader, 
#                     years_to_plot:list=None, 
#                     runs:list=None, 
#                     index:str=str(16), 
#                     fig=None, time_tuple:tuple=(None, None)):
    
#     Q = combined.Q.sel(index=index)
    
#     if runs is None:
#         runs = Q.runs.values
    
#     if years_to_plot is None:
#         years = Q.time.dt.year.values
#         random_years = np.random.choice(years, size=5, replace=False)
#         years_to_plot = random_years
#     if fig is None:
#         fig, axs = plt.subplots(1, len(years_to_plot), figsize=(10, 6.18))
    
#     #add a new ax for each year
#     for i, year in enumerate(years_to_plot):
#         ax_year = fig.add_subplot(1, len(years_to_plot), i+1)
#         for run in runs:
#             color = reader.catalog[reader.key_alias()[run]]['color']
#             marker = reader.catalog[reader.key_alias()[run]]['marker']
#             alpha = reader.catalog[reader.key_alias()[run]]['alpha']
#             label = reader.catalog[reader.key_alias()[run]]['meta']['longname']
#         if time_tuple[0] is not None:
#             Q = Q.sel(time=slice(time_tuple[0], time_tuple[1]))
#             ax_year.plot(Q.time, Q.sel(runs=run), label=run, color=color, marker=marker, alpha=alpha)
#     return ax

# def format_hydrograoh(ax, index:str=None, time_tuple:tuple=(None, None)):
#     ax.set_title(f'Hydrograph at {index}')
#     ax.set_ylabel('Q (m³/s)')
#     ax.set_xlabel('Time')
#     ax.grid(True)
#     return ax

# ax = plot_hydrograph(combined, reader, index=str(16))
# format_hydrograoh(ax, index=str(16))



# def find_hydrological_year(ds, var_name=None, ax=None):
#     """
#     Plot climatology starting from the month with lowest average flow (hydrological year).
    
#     Args:
#         ds: xarray DataArray or Dataset
#         var_name: variable name if ds is Dataset
#         ax: matplotlib axis for plotting
#     """
#     # Calculate monthly means
#     monthly_mean = ds.groupby('time.month').mean()
    
#     # Find the month with minimum flow (start of hydrological year)
#     start_month = monthly_mean.month.values[monthly_mean.argmin()]
    
#     return start_month

# def set_month_ticks(ax, month_order:list):
#         # Create month mapping
#     months = {  
#               1: 'Jan', 2: 'Feb', 3: 'Mar', 
#               4: 'Apr', 5: 'May', 6: 'Jun',
#               7: 'Jul', 8: 'Aug', 9: 'Sep',
#               10: 'Oct', 11: 'Nov', 12: 'Dec'
#               }

#     xlabels = [months[m] for m in month_order]
#     ax.set_xticks(range(len(month_order)))
#     ax.set_xticklabels(xlabels)
#     return ax, months

# def get_cat_entry(reader, key):
#     if not key in reader.catalog.keys():
#         print(f"Variable {key} not found in catalog")
#         #check if var_name is in the alias sub keys 
#         for main_key, value in reader.catalog.items():
#             if main_key=='meta':
#                 continue
#             if key == value['alias']:
#                 key = main_key
#                 break
    
#     cat_entry = reader.catalog[key]
#     return cat_entry

# def plotting_components(cat_entry):
#     color=cat_entry['color']
#     marker=cat_entry['marker']
#     alpha=cat_entry['alpha']
#     long_name=cat_entry['meta']['longname']
#     short_name=cat_entry['meta']['shortname']
#     return color, marker, alpha, long_name, short_name
    
# def plot_monthly_climatology(ds, cat_entry=None, var_name=None, ax=None, start_month:int=None, return_data_only:bool=False):
#     """
#     Plot monthly climatology (average for each month across all years).
    
#     Args:
#         ds: xarray DataArray or Dataset
#         var_name: variable name if ds is Dataset
#         ax: matplotlib axis for plotting
#         start_month: start month of the hydrological year
#     """
    
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
#     # Group by month and compute mean across years
#     monthly_mean = ds.groupby('time.month').mean()
    
#     if not start_month:
#         month_order = list(range(1, 13))
    
#     else:
#         month_order = list(range(start_month, 13)) + list(range(1, start_month))
    
#     if return_data_only:
#         return monthly_mean.sel(month=month_order)
    
#     else:
#         ax, months = set_month_ticks(ax, month_order)
        
        
#         # Plot data in correct order
#         values = [monthly_mean.sel(month=m).values for m in month_order]
#         # Plot
#         if cat_entry is not None:
#             color, marker, alpha, long_name, short_name = plotting_components(cat_entry)
#             ax.plot(range(len(month_order)), values, marker=marker, alpha=alpha, color=color, label=short_name)
#         else:
#             ax.plot(range(len(month_order)), values)
#         ax.grid(True)
        
#         return ax

# def diff_monthly_climatology_plot(ds, reader, ref_name='ref',var_name=None, ax=None, start_month:int=None, relative=True):
#     """
#     take the series in ax and in a new ax plot the difference between the
#     """
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
#     data = plot_monthly_climatology(ds, cat_entry=None, var_name=var_name, ax=ax, return_data_only=True)
#     ref_data = data.sel(runs=ref_name)
#     if not start_month:
#         month_order = list(range(1, 13))
#     else:
#         month_order = list(range(start_month, 13)) + list(range(1, start_month))
    
#     ax, months = set_month_ticks(ax, month_order)
    
#     for run in data.runs.values:
#         if run != ref_name:
#             if relative:
#                 diff = (data.sel(runs=run) - ref_data) / ref_data
#             else:
#                 diff = data.sel(runs=run) - ref_data
#             diff = diff.sel(month=month_order)
#             l_month = [months[m] for m in month_order]
#             l_data = [val for val in diff]
#             if reader is not None:
#                 cat_entry = get_cat_entry(reader, run)
#                 color, marker, alpha, long_name, short_name = plotting_components(cat_entry)
#                 ax.plot(l_month, l_data, marker=marker, alpha=alpha, color=color, label=f"{run} - {short_name}")
#             else:
#                 ax.plot(l_month, l_data)
#             for month, val in zip(l_month, l_data):
#                 if relative:
#                     ax.text(month, val, f'{val*100:.2f}%', ha='right', va='bottom', fontsize=8)
#                 else:
#                     ax.text(month, val, f'{val:.2f}', ha='right', va='bottom', fontsize=8)
    
#     return ax


# def plot_monthly_var(combined, reader, var_name=None, index=None, start_month=None, diff_plot=True, relative=True):
#     # Example usage
#     init_style()
#     if diff_plot:
#         fig, axs = plt.subplots(2, 1, figsize=(10, 5))
#     else:
#         fig, axs = plt.subplots(1, 1, figsize=(10, 5))

#     # For your specific data
#     ex = combined[var_name].sel(index=str(index))  # Get data for first index
#     if start_month is None:
#         start_month = find_hydrological_year(ex.sel(runs='ref'))
#         print(f"Hydrological year starts in month: {start_month}")

#     # Plot climatology for each run
#     for run in ex.runs.values:
#         run_data = ex.sel(runs=run)
#         cat_entry = get_cat_entry(reader, run)
#         plot_monthly_climatology(ds=run_data, cat_entry=cat_entry, ax=axs[0], start_month=start_month)

#     axs[0].legend([format_label(run) for run in ex.runs.values])
#     axs[0].set_ylabel('Mean Discharge [m³/s]')
#     axs[0].set_title(f'Monthly Discharge of {ex.index.values}')
#     # axs[0].set_yscale('log')

#     if diff_plot:
#         diff_monthly_climatology_plot(ds=ex, reader=reader, ref_name='ref', ax=axs[1], start_month=start_month, relative=relative)
#         axs[1].legend([format_label(run) for run in ex.runs.values if run != 'ref'])
        
#         if relative:
#             axs[1].set_ylabel('Difference from ref [%]')
#             axs[1].set_yticklabels(['{:,.0f}%'.format(x*100) for x in axs[1].get_yticks()])
#         else:
#             axs[1].set_ylabel('Difference from ref [m³/s]')
#         #get max val in y axs[1]
#         max_val = axs[1].get_ylim()[0]
#         if max_val < 0:
#             axs[1].set_ylim(max_val, 0)
#         axs[1].set_xlabel('Month')
#         axs[1].set_grid(True)
#         fig.add_subplot(axs[1])
#     plt.show()

# plot_monthly_var(combined, 
#                  reader, 
#                  var_name='Q', 
#                  index=str(16), 
#                  start_month=8, 
#                  diff_plot=True,
#                  relative=True)