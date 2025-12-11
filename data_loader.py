import xarray as xr
import torch
import numpy as np
import pandas as pd

def load_and_process_netcdf_data(
    file_paths: dict,
    var_map: dict,
    target_grid_key: str,
    time_range: tuple = None,
    target_freq: str = 'D'
) -> dict:
    """
    Loads and processes multiple NetCDF files into a model-ready dictionary of tensors.
    This version standardizes the time coordinate name to 'time' for robustness.
    """
    # --- 1. Determine Time Range (Automatic or Manual) ---
    if time_range is None:
        print(f"Time range not provided. Inferring from reference file: '{target_grid_key}'")
        try:
            with xr.open_dataset(file_paths[target_grid_key]) as ref_ds:
                time_coord_name = 'time' if 'time' in ref_ds.coords else 'valid_time'
                start_date = pd.to_datetime(ref_ds[time_coord_name].min().values).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(ref_ds[time_coord_name].max().values).strftime('%Y-%m-%d')
                time_range = (start_date, end_date)
                print(f"  -> Inferred time range: {start_date} to {end_date}")
        except Exception as e:
            print(f"Could not automatically determine time range. Error: {e}")
            raise
    else:
        print(f"Using manually provided time range: {time_range[0]} to {time_range[1]}")

    # --- 2. Load, Standardize, and Pre-process Datasets ---
    print("\n--- Loading and pre-processing datasets ---")
    datasets = {}
    for key, path in file_paths.items():
        print(f"Loading '{key}' from {path}...")
        ds = xr.open_dataset(path)

        # Standardize time coordinate name to 'time'
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})

        # Unit Conversion for Temperature
        if key == 'temperature':
            temp_var_name = var_map.get('temperature', 't2m')
            if temp_var_name in ds.data_vars and ds[temp_var_name].attrs.get('units', '').lower() == 'k':
                print(f"  Converting temperature from Kelvin to Celsius for var '{temp_var_name}'.")
                ds[temp_var_name] = ds[temp_var_name] - 273.15
                ds[temp_var_name].attrs['units'] = 'C'
        
        datasets[key] = ds

    # Combine LAI_HV and LAI_LV
    if 'lai_hv' in datasets and 'lai_lv' in datasets:
        print("Combining LAI_HV and LAI_LV to create total LAI...")
        lai_hv_var = var_map.get('lai_hv', 'lai_hv')
        lai_lv_var = var_map.get('lai_lv', 'lai_lv')
        
        lai_hv_da = datasets['lai_hv'][lai_hv_var]
        lai_lv_da = datasets['lai_lv'][lai_lv_var]

        # Align timestamps and grid before adding
        aligned_hv, aligned_lv = xr.align(lai_hv_da, lai_lv_da, join='inner')
        
        total_lai = aligned_hv + aligned_lv
        total_lai.name = 'lai'
        
        datasets['lai'] = xr.Dataset({'lai': total_lai})
        del datasets['lai_hv'], datasets['lai_lv']
    
    # --- 3. Align to Common Grid and Time ---
    print("\n--- Aligning all data to a common grid and timeframe ---")
    
    reference_ds = datasets[target_grid_key]
    ref_var_name = var_map.get(target_grid_key)
    target_grid = xr.Dataset({
        'lat': reference_ds[ref_var_name].latitude,
        'lon': reference_ds[ref_var_name].longitude,
    })
    
    processed_data_arrays = []
    for key, ds in datasets.items():
        print(f"Aligning '{key}'...")
        da = ds[var_map.get(key, key)]
        
        # Slicing, Resampling, and Regridding (now safely uses 'time')
        da_sliced = da.sel(time=slice(*time_range))
        
        if key in ['precipitation', 'runoff_observed_map', 'pet']:
            da_resampled = da_sliced.resample(time=target_freq).sum()
        else:
            da_resampled = da_sliced.resample(time=target_freq).mean()
        
        da_regridded = da_resampled.interp_like(target_grid, method='linear')
        da_regridded.name = key
        processed_data_arrays.append(da_regridded)

    final_ds = xr.merge(processed_data_arrays)

    # --- 4. Generate Cyclical Temporal Features ---
    print("\n--- Generating cyclical temporal features ---")
    daily_times = pd.to_datetime(final_ds['time'].values)
    day_of_year = daily_times.dayofyear
    month_of_year = daily_times.month
    
    final_ds['day_sin'] = ('time', np.sin(2 * np.pi * day_of_year / 365.25))
    final_ds['day_cos'] = ('time', np.cos(2 * np.pi * day_of_year / 365.25))
    final_ds['month_sin'] = ('time', np.sin(2 * np.pi * month_of_year / 12.0))
    final_ds['month_cos'] = ('time', np.cos(2 * np.pi * month_of_year / 12.0))

    # --- 5. Convert to PyTorch Tensors ---
    print("\n--- Converting to PyTorch tensors ---")
    tensor_dict = {}
    for var in final_ds.data_vars:
        if 'lat' in final_ds[var].dims and 'lon' in final_ds[var].dims:
            tensor_dict[str(var)] = torch.from_numpy(final_ds[var].transpose('time', 'lat', 'lon').values).float()
        else:
            tensor_dict[str(var)] = torch.from_numpy(final_ds[var].values).float()
            
    tensor_dict['daily_dates'] = daily_times
    print("\n--- Data loading and processing complete! ---")
    
    return tensor_dict

if __name__ == '__main__':
    # --- EXAMPLE USAGE ---
    file_paths = {
        'precipitation': 'pre_2023.nc',
        'temperature': 'temp_2023.nc',
        'pet': 'pev_runoff_2023.nc',
        'lai_hv': 'low_high_LAI_2023.nc',
        'lai_lv': 'low_high_LAI_2023.nc',
        'runoff_observed_map': 'pev_runoff_2023.nc'
    }

    var_map = {
        'precipitation': 'tp',
        'temperature': 't2m',
        'pet': 'pev',
        'lai_hv': 'lai_hv',
        'lai_lv': 'lai_lv',
        'runoff_observed_map': 'ro'
    }

    target_grid_key = 'runoff_observed_map'

    # --- Seçenek 1: Zaman aralığını otomatik bul ---
    print("===== Running with AUTOMATIC time range detection =====")
    try:
        data_tensors_auto = load_and_process_netcdf_data(
            file_paths, var_map, target_grid_key, time_range=None
        )
        print("Shapes (auto time):", {k: v.shape for k, v in data_tensors_auto.items() if isinstance(v, torch.Tensor)})
    except Exception as e:
        print(f"AN ERROR OCCURRED: {e}")

    print("\n" + "="*60 + "\n")

    # --- Seçenek 2: Zaman aralığını manuel olarak ver ---
    print("===== Running with MANUAL time range detection =====")
    manual_time = ('2023-03-01', '2023-05-31')
    try:
        data_tensors_manual = load_and_process_netcdf_data(
            file_paths, var_map, target_grid_key, time_range=manual_time
        )
        print("Shapes (manual time):", {k: v.shape for k, v in data_tensors_manual.items() if isinstance(v, torch.Tensor)})
    except Exception as e:
        print(f"AN ERROR OCCURRED: {e}")