#!/usr/bin/env python3
"""
ERA5-Land meteorological data extraction for specific coordinates
Coordinates: 18.899741434351892, 99.01248957594561

Output: CSV with merged datetime column (YYYY-MM-DD HH:MM:SS)
One row per datetime, continuous time series with spatial averaging
"""

import os
import glob
import numpy as np
import platform
import datetime as _dt
import location
try:
    import xarray as xr  # Used primarily on Windows (cfgrib engine)
except Exception:
    xr = None
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def parse_step_value(step):
    """Parse step value to hours"""
    if isinstance(step, np.timedelta64):
        # Convert nanoseconds to hours
        return int(step.astype('timedelta64[h]').astype(int))
    elif hasattr(step, 'total_seconds'):
        return int(step.total_seconds() / 3600)
    else:
        return int(step)
    
def merge_time_step_to_datetime(time_vals, step_vals):
    """Merge time and step arrays into single datetime array"""
    datetimes = []
    
    for base_time in time_vals:
        if isinstance(base_time, np.datetime64):
            # Convert to pandas datetime
            base_dt = pd.to_datetime(base_time)
        else:
            base_dt = pd.to_datetime(base_time)

        for step in step_vals:
            # Parse step to hours
            step_hours = parse_step_value(step)
            
            # Add step hours to base time
            final_dt = base_dt + timedelta(hours=step_hours)
            datetimes.append(final_dt)
    
    return datetimes

def extract_from_file(grib_path, lat_point, lon_point):
    if os.path.exists(grib_path):                    
        print(f"Extracting ERA5-Land data for coordinates ({lat_point}, {lon_point})")
        print("=" * 60)
        
        # Remove potentially corrupted index file
        idx_file = grib_path + '.*.idx'
        for idx in glob.glob(idx_file):
            try:
                os.remove(idx)
                print(f"  Removed old index: {os.path.basename(idx)}")
            except:
                pass
    else:
        print(f"Warning: listed file not found: {grib_path}")

    sys_platform = platform.system()
    print(f"Detected platform: {sys_platform}")

    dataset_opened = False
    latitude = None
    longitude = None
    available_vars = []
    df_rows = []

    print("Attempting to read with xarray/cfgrib ...")
    # Disable index caching to avoid corrupted index errors
    backend_kwargs = {'indexpath': ''}
    with xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=backend_kwargs) as ds:
        print(f"  Dimensions: {list(ds.dims.keys())}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        
        latitude = ds['latitude'].values
        longitude = ds['longitude'].values
        print(f"  Lat range: {latitude.min():.3f} to {latitude.max():.3f}")
        print(f"  Lon range: {longitude.min():.3f} to {longitude.max():.3f}")

        available_vars = [v for v in ds.data_vars]
        print(f"  Available variables (xarray): {available_vars}")

        # Basic extraction for available vars (single point)
        if available_vars:
            try:
                lat_indices, lon_indices, distances = location.find_nearest_point(latitude, longitude, lat_point, lon_point)
                print(f"  Spatial average computed over {len(lat_indices)} points")
                print(f"  Grid points used: {len(lat_indices)}")
                print(f"  Resolution: ~{np.mean(distances):.3f} km")

            except ValueError as e:
                print(f"  WARNING: Spatial averaging failed: {e}")
                print("  Using single point data instead")
                return None
            
            # Calculate inverse distance weights for spatial averaging
            # Points closer to center get higher weights
            max_dist = np.max(distances)
            inverse_distances = np.exp(max_dist - distances)
            weights = inverse_distances / np.sum(inverse_distances)

            print(f"  Weights sum to: {np.sum(weights):.6f}")

            time_vals = ds['time'].values
            step_vals = ds['step'].values

            # Create datetime array
            all_datetimes = merge_time_step_to_datetime(time_vals, step_vals)
            
            # Store data - extract actual float values
            for i, dt in enumerate(all_datetimes):
                time_idx = i // len(step_vals)
                step_idx = i % len(step_vals)

                row = { 'datetime': dt }

                for var in available_vars:
                    data_array = ds[var][time_idx, step_idx, lat_indices, lon_indices]
                    weighted_data = data_array * weights
                    averaged_data = weighted_data.sum(axis=(0, 1))
                    row[var] = float(averaged_data)

                df_rows.append(row)

        dataset_opened = True

    if not dataset_opened:
        print("  ERROR: Unable to open GRIB file with any backend.")
        return None

    # Coordinate range safety check (if latitude/longitude captured)
    if latitude is not None and longitude is not None:
        if not (latitude.min() <= lat_point <= latitude.max()) or not (longitude.min() <= lon_point <= longitude.max()):
            print(f"  WARNING: Coordinates ({lat_point}, {lon_point}) outside data range")
            return None

    if not df_rows:
        print("  WARNING: No data rows extracted; returning None")
        return None

    # Merge rows by datetime (combine variable columns)
    df = pd.DataFrame(df_rows)
    df = df.groupby('datetime').first().reset_index()
    return df

def extract_all_coordinates(files, lat_point, lon_point, output_csv):
    # Process all files
    all_dataframes = []

    for grib_file in sorted(files):
        if os.path.exists(grib_file):
            print(f"Processing {grib_file}...")
                
            df = extract_from_file(grib_file, lat_point, lon_point)
            if df is not None:
                all_dataframes.append(df)
                print(f"  Added {len(df)} records from {os.path.basename(grib_file)}")
            else:
                print(f"  Failed to extract from {os.path.basename(grib_file)}")
        else:
            print(f"Warning: listed file not found: {grib_file}")

    if not all_dataframes:
        print("ERROR: No data extracted from any files")
        return
    
    # Combine all dataframes
    print("=" * 60)
    print(f"Combining data from {len(all_dataframes)} files...")
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Comprehensive merge strategy for duplicate rows
    # print("Applying comprehensive merge strategy for duplicate rows...")
    # combined_df = _intelligent_merge_duplicates(combined_df)
    
    # Sort by datetime
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    # Format datetime column
    combined_df['datetime'] = combined_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV
    print("=" * 60)
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except:
        pass
    
    combined_df.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"SUCCESS: Data saved to {output_csv}")
    print(f"Total records: {len(combined_df)}")
    print(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(combined_df.head())
    
    # Show data statistics
    print("\nData statistics:")
    for var in combined_df.columns:
        non_null = combined_df[var].count()
        total = len(combined_df)
        if non_null > 0:
            min_val = combined_df[var].min()
            max_val = combined_df[var].max()
            print(f"  {var}: {non_null}/{total} values, range [{min_val:.3f}, {max_val:.3f}]")
    
    return combined_df

if __name__ == "__main__":
    # Check for a list of files to process
    # files_list_path = 'dataset/ERA5-land-hourly/era5-land-files.txt'
    netcdf_dir = '/Volumes/Seagate/_datasets/weather-dataset/ERA5-Land-data/land'
    aggregated_out = 'dataset/era5-land_timeseries.csv'

    lat_point=18.899741434351892 
    lon_point=99.01248957594561
    
    # Remove existing aggregated output so we start fresh
    if os.path.exists(aggregated_out):
        try:
            os.remove(aggregated_out)
        except Exception:
            pass

    netcdf_pattern = os.path.join(netcdf_dir, "*.grib")
    netcdf_files = glob.glob(netcdf_pattern)
    
    print(f"Found {len(netcdf_files)} GRIB files")
    # Process each file and collect data
    all_files = []

    for netcdf_file in netcdf_files:
        all_files.append(netcdf_file)

    extract_all_coordinates(all_files, lat_point=lat_point, lon_point=lon_point, output_csv=aggregated_out)
