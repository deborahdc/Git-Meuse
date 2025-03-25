from pathlib import Path
import yaml
from typing import Union
from hydromt_wflow import WflowModel
import xarray as xr
import numpy as np
import sys

#profiling
import time
import psutil
import os
from functools import wraps

def measure_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the execution
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"\nPerformance metrics for {func.__name__}:")
        print(f"Time elapsed: {end - start:.2f} seconds")
        print(f"Memory usage: {mem_after - mem_before:.2f} MB")
        print(f"Final memory: {mem_after:.2f} MB")
        
        return result
    return wrapper

class ResultReader:
    """Class to read and manage wflow model results based on a run catalog."""
    
    def __init__(self, catalog_path: str = "./models/run_catalog.yml"):
        """
        Initialize the ResultReader with a path to the run catalog.
        
        Args:
            catalog_path (str): Path to the run catalog YAML file
        """
        self.catalog_path = Path(catalog_path)
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog file not found at {catalog_path}")

        if " " in str(self.catalog_path):
            print("WARNING: Catalog path contains spaces, known to cause issues")

        self.catalog_path = Path(str(self.catalog_path).replace("\\", "/"))
        try:
            with open(self.catalog_path, 'r') as f:
                self.catalog = yaml.safe_load(f)
                print(f"Catalog loaded successfully from {self.catalog_path}")
            # Get the base path from meta section
        except Exception as e:
            print(f"Error loading catalog file: {e}")
            raise e
        
        def syscheck():
            if "win" in sys.platform:
                self.base_path = Path(str(self.base_path).format("p:"))
            else:
                self.base_path = Path(str(self.base_path).format("/p"))
            return self.base_path
        
        if "path" in self.catalog.get("meta", {}):
            self.base_path = Path(self.catalog["meta"]["path"])
            if "{}" in str(self.base_path):
                self.base_path = syscheck()
        else:
            self.base_path = Path(self.catalog_path).parent
            
    def key_alias(self) -> dict:
        """
        Get the key alias from the catalog.
        """
        # Create exact matches only between aliases and keys
        aliases = {}
        for k, v in self.catalog.items():
            if "alias" in v:
                aliases[v["alias"]] = k
        if not aliases:
            print("WARNING: No aliases found in catalog")
        return aliases

    def get_toml(self, key: str) -> Path:
        """
        Get the toml path from the catalog.
        """
        if not self.catalog[key]["config"].endswith(".toml"):
            return f"{self.catalog[key]['config']}.toml"
        else:
            return self.catalog[key]["config"]
    
    def get_root_config(self, key: str) -> bool:
        """
        Check if a model configuration exists and is valid.
        
        Args:
            key (str): The model identifier from the catalog or its alias
            
        Returns:
            bool: True if the model exists and is valid, False otherwise
        """
        original_key = key
        if key not in self.catalog.keys():
            try:
                key = self.key_alias()[key]
            except KeyError:
                print(f"No matching key or alias found for {original_key}")
                return False
        
        self.root = Path(self.base_path, self.catalog[key]["root"])
        self.config = self.get_toml(key)
        return True

    def check_model_exists(self, key: str) -> bool:
        """
        Check if a model exists.
        """
        self.get_root_config(key)
    
    def get_wflow_model(self, key: str):
        """
        Get the wflow model for a model.
        """
        self.get_root_config(key)
        self.model = WflowModel(root=self.root, config_fn=self.config, mode="r")
        assert len(self.model.config.keys()) > 0, "Model config is empty"
        return self.model

    def get_results(self, key: str):
        """
        Get the results for a model.
        """
        self.get_wflow_model(key)
        return self.model.results

    def combine_to_Q(self, result:dict=None, vars:list=None, key:str=None):
        """
        Combine the Q variables to a single 'Q' variable.
        
        Args:
            result (dict, optional): Dictionary containing results
            vars (list, optional): List of Q variables to combine
            key (str, optional): Model key to get results if result not provided
            
        Returns:
            dict: Updated results dictionary with combined Q variable
        """
        # Input validation
        if vars is None or len(vars) == 0:
            raise ValueError("vars must be provided and non-empty")
        
        # Get results if not provided
        if result is None:
            if key is None:
                raise ValueError("Either result or key must be provided")
            result = self.get_results(key)
        
        out_result = result.copy()
        joined = None
        
        # Combine Q variables
        for var in vars:
            ds = result['netcdf'][[var]]
            gaugemap = list(ds.coords)[1]
            ds_renamed = ds.rename({gaugemap: 'index', var: 'Q'})
            if joined is None:
                joined = ds_renamed
            else:
                joined = xr.concat([joined, ds_renamed], dim='index')
        
        # Get list of Q variables to drop
        Q_vars = [var for var in out_result['netcdf'].data_vars if 'Q_' in var]
        print(f"dropping {Q_vars}")
        
        # Drop the Q variables and their associated coordinates
        for var in Q_vars:
            coord_to_drop = [coord for coord in out_result['netcdf'].coords 
                            if var in coord and 'gauges' in coord]
            
            # Drop both the variable and its coordinate
            out_result['netcdf'] = out_result['netcdf'].drop_vars(var)
            for coord in coord_to_drop:
                out_result['netcdf'] = out_result['netcdf'].drop_dims(coord)
        
        # Merge the joined dataset
        out_result['netcdf'] = xr.merge([out_result['netcdf'], joined])
        
        return out_result

    def get_discharge_gauges(self, results=None,
                             key: str = None,
                             source: str = "netcdf",
                             gauge_ref: Union[str, list] = "Q_gauges") -> dict:
        """
        Use a gauge reference e.g. Q_Sall_gauges_sall to get the values of gauges on that dimension.
        Collect the values of the desired gauges for the analysis. 
        """
        if results is None:
            results = self.get_results(key)[source]
        
        if isinstance(gauge_ref, str):
            return {
                gauge_ref: list(results[source][gauge_ref].coords[gauge_ref].values)
            }
        elif isinstance(gauge_ref, list):
            return {
                key: list(results[source][key].coords[key].values) for key in gauge_ref
            }
        else:
            raise ValueError(f"Invalid gauge reference type: {type(gauge_ref)}")
    
    def find_ideal_datavars(self, datavars, Qex=None):
        q_vars = []
        other_vars = []

        # split out the Q and other vars
        for var in datavars:
            if var == 'Q':
                q_vars.append(var)
            else:
                split = var.split('_')
                other_vars.append(split[0])
        
        q_vars = list(set(q_vars))
        other_vars = list(set(other_vars))

        ideal_datavars = list(q_vars + other_vars)
        return ideal_datavars
    
    def list_das(self, results, target_var):
        """
        List all data variables that start with the target variable.
        """
        datavars = [var for var in results.data_vars.keys() if var.startswith(target_var)]
        das = results[datavars]
        return das
    
    def reduce_dims(self, ds, target_var):
        """
        Reduce dimensions of a dataset by combining all subcatchment data into a single DataArray
        with time and index coordinates.
        
        Args:
            ds: xarray Dataset with multiple subcatchment dimensions
            target_var: str, prefix of variables to combine (e.g., 'Qgwr')
            
        Returns:
            xr.Dataset: New dataset with simplified dimensions (time, index)
        """
        # Get all variables that start with target_var
        datavars = [var for var in ds.data_vars if var.startswith(target_var)]
        
        # Initialize lists to store indices and values
        indices = []
        values = []
        
        # Collect data from each variable
        for var in datavars:
            coord_name = ds[var].dims[1]  # Get the coordinate name
            indices.extend(ds[coord_name].values)
            values.append(ds[var].values)
        
        # Concatenate all values along the second axis (index)
        combined_values = np.concatenate(values, axis=1)
        
        # Create new dataset
        ds_reduced = xr.Dataset(
            {
                target_var: (['time', 'index'], combined_values)
            },
            coords={
                'time': ds.time,
                'index': indices
            }
        )
        
        return ds_reduced

    def create_grouped_dataset(self, 
                               results=None, 
                               key=None, 
                               source='netcdf', 
                               Q_combine:list=None)->dict:
        """
        Create a new dataset combining all variables, handling both matching and new coordinates.
        If results are provided, key must be providd 
        Args:
            results: xarray Dataset containing the results
            key: str, optional, key to load results from
            source: str, optional, source of results ('netcdf' by default)
            Q_combine: list, optional, list of Q variables to combine
        Returns:
            xr.Dataset: New dataset with combined dimensions
        """
        # Check if either results or key is provided
        if results is None:
            assert key is not None, "Key must be provided if results are None"
            results = self.get_results(key)
        
        if Q_combine is not None:
            results = self.combine_to_Q(results, Q_combine, key)
        
        results = results[source]
        
        # First reduce dimensions for each variable type
        ds_reduced = {}

        ideal_datavars = self.find_ideal_datavars(list(results.data_vars.keys()))
        
        for var in ideal_datavars:
            ds_reduced[var] = self.reduce_dims(
                self.list_das(results, var), 
                var
            )
        
        # Get all unique indices while preserving order of appearance
        all_indices = []
        for ds in ds_reduced.values():
            for idx in ds.index.values:
                if idx not in all_indices:
                    all_indices.append(idx)
        
        # Create a new combined dataset
        time_coords = list(ds_reduced.values())[0].time
        combined_das = xr.Dataset(
            coords={
                'time': time_coords,
                'index': all_indices
            }
        )
        
        # Add each variable's data with proper padding for missing indices
        for var, ds in ds_reduced.items():
            # Create a full-size array filled with NaN
            full_data = np.full((len(time_coords), len(all_indices)), np.nan)
            
            # Get the positions of this variable's indices in the full index list
            idx_map = {idx: i for i, idx in enumerate(all_indices)}
            var_idx_positions = [idx_map[idx] for idx in ds.index.values]
            
            # Fill in the data at the correct positions
            full_data[:, var_idx_positions] = ds[var].values
            
            # Add to combined dataset
            combined_das[var] = xr.DataArray(
                data=full_data,
                dims=['time', 'index'],
                coords={
                    'time': time_coords,
                    'index': all_indices
                }
            )
        dict_out = {key: combined_das}
        print(f"Created grouped dataset for {key}")
        return dict_out

    @measure_performance
    def combine_grouped_datasets(self, keys:list, grouped_datasets: dict=None, Q_combine:list=None, outpath:str=None, overwrite:bool=False) -> xr.Dataset:
        """
        Combine multiple datasets from different runs into a single dataset with a new 'runs' dimension.
        
        Args:
            run_datasets: dict where keys are run names and values are xarray Datasets
            
        Returns:
            xr.Dataset: Combined dataset with new 'runs' dimension
        """
        
        if outpath is not None and Path(outpath).exists() and not overwrite:
            print(f"Loading pre-combined dataset from {outpath}")
            combined = xr.open_dataset(outpath)
            return combined

        for key in keys:
            self.check_model_exists(key)
            
        assert keys is not None, "Keys must be provided"
        assert not(keys is None and grouped_datasets is None), "Either keys or grouped_datasets must be provided"
        
        if grouped_datasets is None:
            grouped_datasets = {}
            for key in keys:
                gd = self.create_grouped_dataset(key=key, Q_combine=Q_combine)
                grouped_datasets.update(gd)

        # Get list of all variables (assuming all datasets have same variables)
        first_ds = next(iter(grouped_datasets.values()))
        variables = list(first_ds.data_vars)
        
        # Create the combined dataset
        combined = xr.Dataset(
            coords={
                'time': first_ds.time,
                'index': first_ds.index,
                'runs': list(grouped_datasets.keys())
            }
        )
        
        # Add each variable with the new runs dimension
        for var in variables:
            # Stack all run data for this variable
            stacked_data = np.stack([
                grouped_datasets[run][var].values 
                for run in grouped_datasets.keys()
            ], axis=-1)  # Add runs as the last dimension
            
            # Add to combined dataset
            combined[var] = xr.DataArray(
                data=stacked_data,
                dims=['time', 'index', 'runs'],
                coords={
                    'time': first_ds.time,
                    'index': first_ds.index,
                    'runs': list(grouped_datasets.keys())
                }
            )
        
        return combined
 
    def key_color(self, keys:list)->dict:
        """
        List the colors for the keys.
        """
        colors = {}
        for key in keys:
            try:
                colors[key] = self.catalog[key]["color"]
            except KeyError:
                alias = self.key_alias()[key]
                colors[key] = self.catalog[alias]["color"]
        return colors
    
    #TODO: attach obs
    #TODO: plot signatures
    #TODO: plot change heatmaps
    def plot_change_heatmap(self, keys:list, ref_key:str='ref', grouped_datasets:dict=None, plotids:list=None)->None:
        """
        Plot a change heatmap for the keys.
        """
        

