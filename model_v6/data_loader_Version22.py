import xarray as xr
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_process_data_from_folders(root_data_dir: str, file_name_map: dict, var_map: dict, time_splits: dict, convert_to_mm: bool = True) -> dict:
    """
    Kök klasör içindeki yıl klasörlerini tarar, verileri birleştirir ve böler.
    GÜNCELLENDİ: Dosya adlarını dinamik olarak oluşturur (örn: 'pre_2023.nc').
    """
    print("\n--- Starting Smart Data Loading and Splitting ---")
    root_path = Path(root_data_dir)
    year_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not year_dirs:
        raise FileNotFoundError(f"No year-named subdirectories found in '{root_data_dir}'")
    
    print(f"Found year directories: {[d.name for d in year_dirs]}")

    all_datasets = {key: [] for key in var_map.keys()}
    
    for year_dir in year_dirs:
        year_str = year_dir.name
        for ds_key, var_name in var_map.items():
            # YENİ: Dosya adını `file_name_map` kullanarak dinamik olarak oluştur
            # Örnek: 'pre' + '_' + '2023' + '.nc' -> 'pre_2023.nc'
            file_prefix = file_name_map.get(ds_key, ds_key)
            file_name = f"{file_prefix}_{year_str}.nc"
            file_path = year_dir / file_name
            
            if file_path.exists():
                ds = xr.open_dataset(file_path)
                all_datasets[ds_key].append(ds)
            else:
                # Orijinal dosyayı bulamazsa, belki de birleşik bir dosyadır.
                # Örn: pev_runoff_2023.nc gibi.
                combined_file_path = year_dir / f"pev_runoff_{year_str}.nc"
                if combined_file_path.exists():
                     ds = xr.open_dataset(combined_file_path)
                     all_datasets[ds_key].append(ds)
                else:
                    print(f"Warning: File '{file_path}' not found. Skipping '{ds_key}' for year {year_str}.")

    # --- (Bu noktadan sonraki kod aynı kalabilir) ---
    # ... Veriyi birleştirme, işleme ve bölme adımları ...
    # ...

    # Referans veri setini ve boyutları belirle
    ref_key = 'runoff_observed_map'
    if not all_datasets[ref_key]:
        raise ValueError(f"Critical data '{ref_key}' could not be loaded from any directory.")
    
    combined_ds = {}
    for ds_key, ds_list in all_datasets.items():
        if ds_list:
            time_dim = next((dim for dim in ds_list[0].dims if 'time' in dim.lower()), 'time')
            combined_ds[ds_key] = xr.concat(ds_list, dim=time_dim)

    ref_ds = combined_ds[ref_key]
    time_dim_name = next(dim for dim in ref_ds.coords if 'time' in dim.lower())
    lat_dim_name = next(dim for dim in ref_ds.coords if 'lat' in dim.lower())
    lon_dim_name = next(dim for dim in ref_ds.coords if 'lon' in dim.lower())
    
    full_start_date = pd.to_datetime(ref_ds[time_dim_name].values.min()).floor('D')
    full_end_date = pd.to_datetime(ref_ds[time_dim_name].values.max()).ceil('D')
    daily_time_index = pd.date_range(start=full_start_date, end=full_end_date, freq='D')
    print(f"Full data range: {full_start_date.date()} to {full_end_date.date()}")
    
    full_data = {'daily_dates': daily_time_index}
    vars_to_convert = ['precipitation', 'pet', 'runoff_observed_map']
    
    for key, ds in combined_ds.items():
        da = ds[var_map[key]]
        resampling_method = 'sum' if key == 'precipitation' else 'mean'
        daily_da = da.resample({time_dim_name: '1D'}).reduce(getattr(np, resampling_method, np.mean))
        aligned_da = daily_da.reindex({time_dim_name: daily_time_index}, method='nearest').rename({time_dim_name: 'time'})
        
        numpy_array = aligned_da.fillna(0).values
        if convert_to_mm and key in vars_to_convert:
            numpy_array *= 1000.0
        full_data[key] = torch.from_numpy(numpy_array.astype(np.float32))

    print("\nSplitting data into periods:")
    start_train = pd.to_datetime(time_splits['train_start'])
    end_train = pd.to_datetime(time_splits['train_end'])
    start_test = pd.to_datetime(time_splits['test_start'])
    end_test = pd.to_datetime(time_splits['test_end'])
    warmup_days = time_splits['warmup_days']
    start_warmup = start_train - pd.Timedelta(days=warmup_days)

    split_data = {'train': {}, 'test': {}, 'attributes': {}}
    
    train_slice = slice(start_warmup, end_train)
    train_dates = daily_time_index[daily_time_index.slice_indexer(start_train, end_train)]
    split_data['train']['dates'] = train_dates
    split_data['train']['warmup_steps'] = len(daily_time_index[daily_time_index.slice_indexer(start_warmup, start_train-pd.Timedelta(days=1))])
    
    test_warmup_start = start_test - pd.Timedelta(days=warmup_days)
    test_slice = slice(test_warmup_start, end_test)
    test_dates = daily_time_index[daily_time_index.slice_indexer(start_test, end_test)]
    split_data['test']['dates'] = test_dates
    split_data['test']['warmup_steps'] = len(daily_time_index[daily_time_index.slice_indexer(test_warmup_start, start_test-pd.Timedelta(days=1))])
    
    print(f"  Training period: {start_train.date()} - {end_train.date()} (with warmup from {start_warmup.date()})")
    print(f"  Testing period:  {start_test.date()} - {end_test.date()} (with warmup from {test_warmup_start.date()})")
    
    for key, tensor in full_data.items():
        if key != 'daily_dates':
            split_data['train'][key] = tensor[daily_time_index.slice_indexer(train_slice.start, train_slice.stop)]
            split_data['test'][key] = tensor[daily_time_index.slice_indexer(test_slice.start, test_slice.stop)]
    
    split_data['attributes']['grid_shape'] = full_data['precipitation'].shape[1:]
    return split_data