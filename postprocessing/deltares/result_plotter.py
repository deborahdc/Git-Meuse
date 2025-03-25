import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def init_style():
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (10, 6.18)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['figure.titleweight'] = 'bold'


def calc_Qwet(ds, wet_months):
    """
    Calculate the mean discharge over all wet months in the timeseries.
    """
    # Select wet months directly from time series
    wet_mask = ds['time.month'].isin(wet_months)
    ds_wet = ds.where(wet_mask, drop=True)
    # Mean over all wet season timepoints
    Q_mean_wet = ds_wet.mean(dim='time')
    return Q_mean_wet

def calc_Qdry(ds, dry_months):
    """
    Calculate the mean discharge over all dry months in the timeseries.
    """
    # Select dry months directly from time series
    dry_mask = ds['time.month'].isin(dry_months)
    ds_dry = ds.where(dry_mask, drop=True)
    # Mean over all dry season timepoints
    Q_mean_dry = ds_dry.mean(dim='time')
    return Q_mean_dry

def format_value(x):
    """Custom formatter with compact scientific notation for large numbers"""
    if x <= -1e6:
        return r'-$\infty$'
    elif x >= 1e6:
        return r'+$\infty$'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.2f}k'
    elif abs(x) <= 1e2:
        return f'{x:.2f}'
    return f'{x:.2f}'

def format_label(label):
    """Format label by capitalizing each part and joining them."""
    if '_' in label:
        label_parts = label.split('_')
        return ' '.join(part.capitalize() for part in label_parts)
    return label.capitalize()
    

def plot_table_statistics_combined(
    combined_ds,
    variables=None,
    months_dry=None,
    months_wet=None,
    reader=None,
    reference_run='ref',
    index=None,
    vmin=-20,
    vmax=20,
    cmap="RdBu",
    outpath=None,
    invert_cmap_for=[]
):
    """
    Create a heatmap showing both absolute values and relative changes in each cell.
    """
    if index is None:
        index = combined_ds[variables[0]].index.values[0]
    
    # Prepare data dictionaries
    abs_dict = {}
    
    if isinstance(variables, str):
        variables = [variables]
    elif isinstance(variables, list):
        pass
    elif variables is None:
        variables = list(combined_ds.data_vars) + ["Qwet", "Qdry"]
    
    all_runs = [run for run in combined_ds[variables[0]].runs.values]
    
    # Calculate absolute values for all variables and runs
    # Calculate Q_wet and Q_dry first
    if "Qwet" in variables or "Qdry" in variables:
        Q_data = combined_ds.sel(index=index).Q
        
        if "Qwet" in variables:
            Q_wet = calc_Qwet(Q_data, months_wet)
            abs_dict["Qwet"] = Q_wet.values
            
        if "Qdry" in variables:
            Q_dry = calc_Qdry(Q_data, months_dry)
            abs_dict["Qdry"] = Q_dry.values
    
    # Process other variables
    for var in variables:
        if var not in ["Qwet", "Qdry"]:
            data = combined_ds[var].sel(index=index)
            # Calculate mean over entire timeseries
            abs_mean = data.mean(dim='time')
            abs_dict[var] = abs_mean.values

    # Create DataFrames for absolute values
    # Make sure reference_run is the first row in the DataFrame
    all_runs_ordered = [reference_run] + [run for run in all_runs if run != reference_run]
    labels = [format_label(reader.catalog[reader.key_alias()[run]]['meta']['longname']) for run in all_runs_ordered]
    
    # Create and organize the DataFrame
    abs_df = pd.DataFrame(
        {var: [abs_dict[var][all_runs_ordered.index(run)] for run in all_runs_ordered] for var in abs_dict.keys()},
        index=labels
    )
    
    # Order columns alphabetically
    abs_df = abs_df[sorted(abs_df.columns)]
    abs_df = abs_df.fillna(0.)
    abs_df = abs_df.replace(np.inf, 1e6)
    abs_df = abs_df.replace(-np.inf, -1e6)
    abs_df = abs_df.astype(float).round(3)
    
    # Calculate relative changes
    # Get reference values (first row)
    ref_row = abs_df.iloc[0]
    
    # Create a relative change dataframe initialized with zeros
    rel_df = pd.DataFrame(0, index=abs_df.index, columns=abs_df.columns)
    
    # Calculate relative changes (but leave reference row as zeros)
    for i in range(1, len(abs_df)):  # Skip the reference row
        rel_df.iloc[i] = (abs_df.iloc[i] / ref_row - 1.0) * 100
    
    # Invert colormap for specified variables if needed
    for var in invert_cmap_for:
        if var in rel_df.columns:
            for i in range(1, len(rel_df)):  # Skip reference row
                rel_df.iloc[i, rel_df.columns.get_loc(var)] = -rel_df.iloc[i, rel_df.columns.get_loc(var)]
    
    # Create the plot
    fs = 10
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Clip values for coloring
    rel_df_clipped = rel_df.copy()
    
    # Make sure reference row stays at 0 (white/neutral color)
    for i in range(1, len(rel_df_clipped)):
        rel_df_clipped.iloc[i] = rel_df_clipped.iloc[i].clip(lower=vmin, upper=vmax)
    
    # Create heatmap
    im = sns.heatmap(
        rel_df_clipped,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        annot=True,  # Will replace these annotations
        fmt='.1f',
        linewidths=0.5,
        ax=ax,
        cbar_kws={
            'label': 'Relative Change (%)',
            'orientation': 'horizontal',
            'pad': 0.05,
            'fraction': 0.05
        }
    )
    
    # Now replace annotations to include both absolute and relative values
    for i, text in enumerate(im.texts):
        row_idx = i // len(rel_df.columns)
        col_idx = i % len(rel_df.columns)
        
        abs_val = abs_df.iloc[row_idx, col_idx]
        rel_val = rel_df.iloc[row_idx, col_idx]
        
        # Format values
        if abs(abs_val) >= 1e6:
            abs_text = "±∞"
        elif abs(abs_val) >= 1e3:
            abs_text = f"{abs_val/1e3:.2f}k"
        else:
            abs_text = f"{abs_val:.2f}"
        
        # For reference row, only show absolute value
        if row_idx == 0:  # Reference row
            text.set_text(f"{abs_text}\n(Ref)")
            # Highlight reference cells with a different color
            text.set_color('darkblue')
            text.set_fontweight('bold')
        else:
            if abs(rel_val) >= 1e6:
                rel_text = "±∞%"
            else:
                rel_text = f"{rel_val:.1f}%"
            text.set_text(f"{abs_text}\n({rel_text})")
        
        text.set_fontsize(8)  # Smaller font to fit
    
    # Customize the plot
    ax.set_title(f'Absolute Values and Relative Changes from {format_label(reference_run)} at {index}')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Scenarios')
    
    # Add a horizontal line below the reference row to visually separate it
    ax.axhline(y=1, color='black', linewidth=2)
    
    # Save or show the plot
    plt.tight_layout()
    if outpath is not None:
        plt.savefig(f"{outpath}_combined.png", dpi=500, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig, ax
