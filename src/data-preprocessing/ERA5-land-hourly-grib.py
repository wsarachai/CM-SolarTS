# 📊 Dataset Variables
# These are the actual data fields stored in the dataset:
#  ssrd: Surface solar radiation downwards (in W/m²).
#  t2m: 2-meter air temperature (in Kelvin).
#  u10: 10-meter U-component of wind (in m/s).
#  v10: 10-meter V-component of wind (in m/s).
#  tp: Total precipitation (in meters).
#

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_grib_data(grib_path):
    """
    Visualize the data in a GRIB file.

    Args:
        grib_path (str): Path to the GRIB file
    """
    print(f"Visualizing GRIB file: {grib_path}")

    try:
        # Open the GRIB file
        with xr.open_dataset(grib_path, engine='cfgrib') as ds:
            print(f"\nDataset dimensions: {list(ds.dims.keys())}")
            print(f"Dataset variables: {list(ds.data_vars.keys())}")
            print(f"Dataset coordinates: {list(ds.coords.keys())}")

            # Extract the data
            try:
                time = ds['time'].values
                longitude = ds['longitude'].values
                latitude = ds['latitude'].values
            except Exception as e:
                print(f"Error extracting coordinates: {e}")
                return

            # Variables we want to visualize (only those present in dataset)
            vars_to_plot = [v for v in ['ssrd', 't2m', 'u10', 'v10', 'tp'] if v in ds.data_vars]
            lon_len = np.asarray(longitude).size
            lat_len = np.asarray(latitude).size
            print(f"Longitude length: {lon_len}, Latitude length: {lat_len}")

            # Helper: try to decode time coordinate to datetimes
            try:
                time_vals = xr.conventions.times.decode_cf_datetime(ds['time'], ds['time'].attrs.get('units'))
            except Exception:
                time_vals = ds['time'].values

            for var in vars_to_plot:
                a = ds[var]
                print(f"\nVariable '{var}' shape: {a.shape}, dtype: {a.dtype}")
                print(f"Variable '{var}' attributes: {a.attrs}")

                print(f"{a}")

                try:
                    # Avoid loading full data for very large DataArrays; sample a small slice instead
                    if hasattr(a, 'size') and a.size > 50_000_000:
                        print(f"  DataArray too large ({a.size} elements), sampling a small slice for stats")
                        # build small slice for each dim
                        sel = {}
                        for d, s in zip(a.dims, a.shape):
                            sel[d] = slice(0, min(3, s))
                        try:
                            sample = a.isel(sel)
                            arr = sample.values
                        except Exception:
                            # fallback to converting a small numpy slice
                            arr = np.asarray(a.isel({a.dims[0]: slice(0, min(3, a.shape[0]))}).values).ravel()
                    else:
                        arr = np.asarray(a)
                except Exception as e:
                    print(f"  all extraction methods failed for {var}")
                    print(f"  array type: {type(a)}")
                    print(f"  array shape: {a.shape}")
                    print(f"  array dtype: {a.dtype}")
                    print(f"  error details: {str(e)}")
                    continue

                if arr is None:
                    print(f"  unable to extract array data from variable")
                    continue

                # Check if array is numeric and not empty
                if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    # Handle large arrays that might cause memory issues
                    try:
                        # For very large arrays, compute stats in chunks or use sampling
                        if arr.size > 50_000_000:  # ~200MB for float32
                            print(f"  array too large ({arr.size} elements), using sampling method")
                            # Sample a subset of the data for statistics
                            flat_arr = arr.flatten()
                            sample_size = min(1_000_000, arr.size // 10)  # Sample 1M or 10% of data
                            sample_indices = np.random.choice(arr.size, size=sample_size, replace=False)
                            sample_data = flat_arr[sample_indices]

                            print(f"  min={np.nanmin(sample_data)}, max={np.nanmax(sample_data)}, mean={np.nanmean(sample_data)} (sampled)")
                        else:
                            print(f"  min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")

                        # Also show data type and shape info
                        print(f"  dtype: {arr.dtype}, shape: {arr.shape}")

                    except MemoryError:
                        print(f"  memory error with large array ({arr.size} elements), trying minimal stats")
                        # Try to get basic info without full computation
                        print(f"  dtype: {arr.dtype}, shape: {arr.shape}, size: {arr.size}")
                        # Try to get a small slice for basic stats
                        if arr.ndim >= 2:
                            small_slice = arr.flat[:1000] if arr.size > 1000 else arr.flatten()
                            print(f"  estimated min={np.nanmin(small_slice)}, max={np.nanmax(small_slice)}, mean={np.nanmean(small_slice)} (from slice)")
                    except Exception as e:
                        print(f"  error computing stats: {str(e)}")
                else:
                    print(f"  unable to compute numeric stats: non-numeric or empty data (dtype: {arr.dtype}, size: {arr.size})")
                    continue


                if lat_len > 1 and lon_len > 1:
                    # spatial grid: reduce over time if present
                    try:
                        if a.ndim == 3:
                            # For large 3D arrays, compute mean in chunks to avoid memory issues
                            total_elems = int(a.shape[0]) * int(a.shape[1]) * int(a.shape[2])
                            if total_elems > 50_000_000:
                                print(f"  large 3D array, computing mean in chunks")
                                # Compute mean along time axis in chunks using .isel to avoid loading full array
                                chunk_size = min(int(a.shape[0]), 100)  # Process up to 100 time steps at a time
                                means = []
                                for i in range(0, int(a.shape[0]), chunk_size):
                                    sel = {a.dims[0]: slice(i, i+chunk_size)}
                                    try:
                                        chunk = a.isel(sel).values
                                    except Exception:
                                        # try smaller chunk or fallback
                                        chunk = np.asarray(a.isel({a.dims[0]: slice(i, i+1)}).values)
                                    chunk_mean = np.nanmean(chunk, axis=0)
                                    means.append(chunk_mean)
                                data2d = np.nanmean(np.array(means), axis=0)
                            else:
                                # small enough to load
                                data2d = np.nanmean(a.values, axis=0)
                        elif a.ndim == 2:
                            data2d = a.values
                        else:
                            print(f"Skipping {var}: unexpected ndim={a.ndim}")
                            continue
                    except Exception as e:
                        print(f"  error processing spatial data: {str(e)}")
                        print(f"Skipping {var}: unexpected ndim={a.ndim}")
                        continue

                    print(f"  plotting spatial map for {var}, shape={data2d.shape}")
                    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
                    plt.figure(figsize=(12, 6))
                    cs = plt.contourf(lon_grid, lat_grid, data2d, cmap='viridis', levels=20)
                    plt.colorbar(cs, label=var)
                    plt.title(f'{var} (mean over time)')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.grid(True, alpha=0.25)
                    out_map = f'dataset/{var}_spatial_map.png'
                    plt.tight_layout()
                    plt.savefig(out_map, dpi=300, bbox_inches='tight')
                    print(f"  saved {out_map}")
                    plt.show()

                else:
                    # single-point: extract time series
                    try:
                        if a.ndim == 3:
                            ts = a.values[:, 0, 0].squeeze()
                        elif a.ndim == 2:
                            ts = a.values[:, 0].squeeze()
                        elif a.ndim == 1:
                            ts = a.values.squeeze()
                        else:
                            print(f"Skipping {var}: unexpected ndim={a.ndim}")
                            continue
                    except Exception as e:
                        print(f"  error extracting time series: {str(e)}")
                        print(f"Skipping {var}: unexpected ndim={a.ndim}")
                        continue

                    print(f"  plotting time series for {var}, length={ts.shape[0]}")
                    plt.figure(figsize=(12, 4))
                    plt.plot(time_vals, ts, '-', lw=1.2)
                    plt.title(f'{var} time series at single location')
                    plt.xlabel('Time')
                    plt.ylabel(var)
                    plt.grid(alpha=0.3)
                    out_ts = f'dataset/{var}_timeseries.png'
                    plt.tight_layout()
                    plt.savefig(out_ts, dpi=300, bbox_inches='tight')
                    print(f"  saved {out_ts}")
                    plt.show()

                    # For single-point datasets we only create the time-series plot above.
    
    except Exception as e:
        print(f"Error visualizing GRIB file: {e}")

if __name__ == "__main__":
    # Path to the GRIB file
    grib_file_path = "dataset/ERA5-land-hourly-2019.grib"

    # Check if the file exists
    if os.path.exists(grib_file_path):
        visualize_grib_data(grib_file_path)
    else:
        print(f"Error: GRIB file not found at {grib_file_path}")