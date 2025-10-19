# 📊 Dataset Variables
# These are the actual data fields stored in the dataset:
#  cc: Cloud cover (likely fractional, from 0 to 1).
#  r: Relative humidity (percentage, from 0 to 100).
#  q: Specific humidity (mass of water vapor per mass of air, in kg/kg).
#  t: Air temperature (usually in Kelvin or Celsius).
#
# 🧭 Dataset Coordinates
# These provide metadata or additional axes for interpreting the data:
#
#    number: May refer to ensemble member number (if using ensemble forecasts).
#    time: Timestamp of the data point.
#    step: Forecast lead time (e.g., 0h, 1h, 2h).
#    isobaricInhPa: Pressure level in hectopascals (e.g., 850 hPa, 500 hPa).
#    latitude and longitude: Geospatial coordinates.
#    valid_time: The actual time the forecast is valid for (i.e., time + step).

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

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
            time = ds['time'].values
            longitude = ds['longitude'].values
            latitude = ds['latitude'].values

            # Variables we want to visualize (only those present in dataset)
            vars_to_plot = [v for v in ['cc', 'r', 'q', 't'] if v in ds.data_vars]
            lon_len = np.asarray(longitude).size
            lat_len = np.asarray(latitude).size
            print(f"Longitude length: {lon_len}, Latitude length: {lat_len}")

            # Helper: try to decode time coordinate to datetimes
            try:
                time_vals = xr.conventions.times.decode_cf_datetime(ds['time'], ds['time'].attrs.get('units'))
            except Exception:
                time_vals = ds['time'].values

            # Track if we've plotted the single-point location map already
            single_point_map_created = False

            for var in vars_to_plot:
                a = ds[var]
                print(f"\nVariable '{var}' shape: {a.shape}, dtype: {a.dtype}")

                # Compute basic stats
                try:
                    arr = a.values
                    print(f"  min={np.nanmin(arr)}, max={np.nanmax(arr)}, mean={np.nanmean(arr)}")
                except Exception:
                    print("  unable to compute numeric stats for this variable")

                if lat_len > 1 and lon_len > 1:
                    # spatial grid: reduce over time if present
                    if a.ndim == 3:
                        data2d = np.nanmean(a.values, axis=0)
                    elif a.ndim == 2:
                        data2d = a.values
                    else:
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
                    if a.ndim == 3:
                        ts = a.values[:, 0, 0].squeeze()
                    elif a.ndim == 2:
                        ts = a.values[:, 0].squeeze()
                    elif a.ndim == 1:
                        ts = a.values.squeeze()
                    else:
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
    grib_file_path = "dataset/derived-utci-historical.grib"

    # Check if the file exists
    if os.path.exists(grib_file_path):
        visualize_grib_data(grib_file_path)
    else:
        print(f"Error: GRIB file not found at {grib_file_path}")