#!/usr/bin/env python3
"""
ERA5-Land meteorological data extraction for specific coordinates
Coordinates: 18.899741434351892, 99.01248957594561

Output: CSV with merged datetime column (YYYY-MM-DD HH:MM:SS)
One row per datetime, continuous time series
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

# Target meteorological variables to extract
TARGET_VARS = ['t2m', 'fal', 'slhf', 'ssr', 'str', 'sshf', 'ssrd', 'strd', 'u10', 'v10', 'sp', 'tp']

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
        for step in step_vals:
            if isinstance(base_time, np.datetime64):
                # Convert to pandas datetime
                base_dt = pd.to_datetime(base_time)
            else:
                base_dt = pd.to_datetime(base_time)
            
            # Parse step to hours
            step_hours = parse_step_value(step)
            
            # Add step hours to base time
            final_dt = base_dt + timedelta(hours=step_hours)
            datetimes.append(final_dt)
    
    return datetimes

def extract_era5_land_coordinates(grib_path, lat_point, lon_point, target_vars=TARGET_VARS):
    """
    Extract ERA5-Land data for specific coordinates from a single GRIB file
    
    Args:
        grib_path (str): Path to GRIB file
        lat_point (float): Target latitude
        lon_point (float): Target longitude
        target_vars (list): List of variable names to extract
    
    Returns:
        pandas.DataFrame: Extracted data with merged datetime column
    """
    print(f"Processing: {os.path.basename(grib_path)}")
    
    try:
        # Open GRIB file
        with xr.open_dataset(grib_path, engine='cfgrib') as ds:
            print(f"  Dimensions: {list(ds.dims.keys())}")
            print(f"  Variables: {list(ds.data_vars.keys())}")
            
            # Check if requested variables are available
            available_vars = [v for v in target_vars if v in ds.data_vars]
            missing_vars = [v for v in target_vars if v not in ds.data_vars]
            
            if missing_vars:
                print(f"  Missing variables: {missing_vars}")
            
            print(f"  Available variables: {available_vars}")
            
            # Get coordinate ranges
            latitude = ds['latitude'].values
            longitude = ds['longitude'].values
            
            print(f"  Lat range: {latitude.min():.3f} to {latitude.max():.3f}")
            print(f"  Lon range: {longitude.min():.3f} to {longitude.max():.3f}")
            
            # Check if coordinates are within range
            if not (latitude.min() <= lat_point <= latitude.max() and 
                   longitude.min() <= lon_point <= longitude.max()):
                print(f"  WARNING: Coordinates ({lat_point}, {lon_point}) outside data range")
                return None
            
            # Find nearest grid point
            lat_idx = np.abs(latitude - lat_point).argmin()
            lon_idx = np.abs(longitude - lon_point).argmin()
            nearest_lat = latitude[lat_idx]
            nearest_lon = longitude[lon_idx]
            
            print(f"  Nearest grid point: ({nearest_lat:.3f}, {nearest_lon:.3f})")
            print(f"  Distance: {abs(nearest_lat - lat_point):.3f} deg lat, {abs(nearest_lon - lon_point):.3f} deg lon")
            
            # Extract data for each variable
            all_data = []
            
            for var in available_vars:
                var_data = ds[var]
                print(f"  Processing {var}: shape={var_data.shape}, dims={var_data.dims}")
                
                if var_data.ndim == 4:
                    # 4D data (time, step, latitude, longitude)
                    time_vals = var_data['time'].values
                    step_vals = var_data['step'].values
                    
                    # Get data for specific location and all time-steps
                    location_data = var_data[:, :, lat_idx, lon_idx].values
                    
                    # Create datetime array
                    all_datetimes = merge_time_step_to_datetime(time_vals, step_vals)
                    
                    # Store data
                    for i, dt in enumerate(all_datetimes):
                        time_idx = i // len(step_vals)
                        step_idx = i % len(step_vals)
                        
                        all_data.append({
                            'datetime': dt,
                            'variable': var,
                            'value': location_data[time_idx, step_idx]
                        })
                
                elif var_data.ndim == 3:
                    # 3D data (time, latitude, longitude)
                    time_vals = var_data['time'].values
                    
                    # Get data for specific location
                    location_data = var_data[:, lat_idx, lon_idx].values
                    
                    # Create datetime array (assume 00:00 for each time step)
                    all_datetimes = [pd.to_datetime(t) for t in time_vals]
                    
                    # Store data
                    for i, dt in enumerate(all_datetimes):
                        all_data.append({
                            'datetime': dt,
                            'variable': var,
                            'value': location_data[i]
                        })
                
                elif var_data.ndim == 2:
                    # 2D data (latitude, longitude) - single timestep
                    location_data = var_data[lat_idx, lon_idx].values
                    
                    # Use file modification time or current time as reference
                    ref_time = pd.to_datetime(os.path.getmtime(grib_path), unit='s', utc=True)
                    
                    all_data.append({
                        'datetime': ref_time,
                        'variable': var,
                        'value': location_data
                    })
                
                else:
                    print(f"  WARNING: Unexpected dimensions for {var}: {var_data.ndim}")
                    continue
            
            # Convert to DataFrame and pivot to wide format
            if all_data:
                df_long = pd.DataFrame(all_data)
                
                # Pivot to wide format (datetime as index, variables as columns)
                df_wide = df_long.pivot(index='datetime', columns='variable', values='value')
                df_wide = df_wide.reset_index()
                
                # Add coordinate information
                df_wide['latitude'] = nearest_lat
                df_wide['longitude'] = nearest_lon
                df_wide['target_lat'] = lat_point
                df_wide['target_lon'] = lon_point
                df_wide['file'] = os.path.basename(grib_path)
                
                # Sort by datetime
                df_wide = df_wide.sort_values('datetime').reset_index(drop=True)
                
                return df_wide
            else:
                print(f"  WARNING: No data extracted")
                return None
                
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_all_coordinates(grib_dir="dataset/ERA5-land-hourly/", 
                          lat_point=18.899741434351892, 
                          lon_point=99.01248957594561,
                          output_csv="dataset/era5_land_extracted.csv"):
    """
    Extract data from all GRIB files for specific coordinates
    
    Args:
        grib_dir (str): Directory containing GRIB files
        lat_point (float): Target latitude
        lon_point (float): Target longitude  
        output_csv (str): Output CSV file path
    """
    print(f"Extracting ERA5-Land data for coordinates ({lat_point}, {lon_point})")
    print(f"From directory: {grib_dir}")
    print("=" * 60)
    
    # Find all GRIB files
    grib_pattern = os.path.join(grib_dir, "*.grib")
    grib_files = glob.glob(grib_pattern)
    
    if not grib_files:
        print(f"ERROR: No GRIB files found in {grib_dir}")
        return
    
    print(f"Found {len(grib_files)} GRIB files")
    print("=" * 60)
    
    # Process all files
    all_dataframes = []
    
    for grib_file in sorted(grib_files):
        df = extract_era5_land_coordinates(grib_file, lat_point, lon_point)
        if df is not None:
            all_dataframes.append(df)
            print(f"  Added {len(df)} records from {os.path.basename(grib_file)}")
        else:
            print(f"  Failed to extract from {os.path.basename(grib_file)}")
    
    if not all_dataframes:
        print("ERROR: No data extracted from any files")
        return
    
    # Combine all dataframes
    print("=" * 60)
    print(f"Combining data from {len(all_dataframes)} files...")
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Remove rows where all target variables are NaN
    print("Removing rows with all NaN values...")
    combined_df = combined_df.dropna(subset=TARGET_VARS, how='all')
    
    # Remove duplicates and sort
    print("Removing duplicates and sorting...")
    combined_df = combined_df.drop_duplicates(subset=['datetime'] + TARGET_VARS)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
    # Format datetime column
    combined_df['datetime'] = combined_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Ensure all target variables are present
    for var in TARGET_VARS:
        if var not in combined_df.columns:
            combined_df[var] = np.nan
    
    # Reorder columns
    column_order = ['datetime'] + TARGET_VARS + ['latitude', 'longitude', 'target_lat', 'target_lon', 'file']
    available_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
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
    print(f"Variables extracted: {TARGET_VARS}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(combined_df.head())
    
    # Show data statistics
    print("\nData statistics:")
    for var in TARGET_VARS:
        if var in combined_df.columns:
            non_null = combined_df[var].count()
            total = len(combined_df)
            if non_null > 0:
                min_val = combined_df[var].min()
                max_val = combined_df[var].max()
                print(f"  {var}: {non_null}/{total} values, range [{min_val:.3f}, {max_val:.3f}]")
    
    return combined_df

if __name__ == "__main__":
    print("ERA5-Land Data Extraction")
    print("Specific coordinates: 18.899741434351892, 99.01248957594561")
    print("Output format: CSV with merged datetime column")
    print("Variables: time,t2m,fal,slhf,ssr,str,sshf,ssrd,strd,u10,v10,sp,tp")
    print("=" * 80)
    
    # Extract data
    result_df = extract_all_coordinates()
    
    if result_df is not None:
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"Output file: dataset/era5_land_extracted.csv")
        print(f"Records: {len(result_df)}")
        print(f"Time range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
    else:
        print("\n" + "=" * 80)
        print("EXTRACTION FAILED!")