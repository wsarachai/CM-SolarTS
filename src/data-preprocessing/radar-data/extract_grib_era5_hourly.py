# 📊 GRIB Data Extractor
# This script extracts actual data values from GRIB files and saves them as CSV files.
#
# Variables in ERA5 data:
# cc: *Cloud cover* — typically expressed as a fraction (0 to 1) indicating the proportion of the sky covered by clouds.
# r: *Relative humidity* — the ratio of the actual water vapor pressure to the saturation vapor pressure, expressed as a percentage.
# q: *Specific humidity* — the mass of water vapor per unit mass of moist air (usually in kg/kg).
# t: *Temperature* — air temperature, usually in Kelvin.
# u: *Zonal wind component* — the east-west component of wind speed (positive values indicate wind from west to east).
# v: *Meridional wind component* — the north-south component of wind speed (positive values indicate wind from south to north).
#

import xarray as xr
import pandas as pd
import numpy as np
import os
import glob

# Variables we're interested in
dataVars = ['cc', 'r', 'q', 't', 'u', 'v']

def extract_grib_data(grib_path, lat_point=None, lon_point=None):
    """
    Extract data values from a single GRIB file.
    
    Args:
        grib_path (str): Path to the GRIB file
        lat_point (float): Latitude of point to extract (optional, defaults to center point)
        lon_point (float): Longitude of point to extract (optional, defaults to center point)
    
    Returns:
        pandas.DataFrame: DataFrame containing extracted data
    """
    print(f"Extracting data from GRIB file: {grib_path}")
    
    try:
        # Open the GRIB file
        with xr.open_dataset(grib_path, engine='cfgrib') as ds:
            print(f"\nDataset dimensions: {list(ds.dims.keys())}")
            print(f"Dataset variables: {list(ds.data_vars.keys())}")
            
            # Extract time coordinate
            time = ds['time'].values
            
            # Variables we want to extract (only those present in dataset)
            vars_to_extract = [v for v in dataVars if v in ds.data_vars]
            print(f"Variables to extract: {vars_to_extract}")
            
            # Determine spatial dimensions
            latitude = ds['latitude'].values
            longitude = ds['longitude'].values
            print(f"Latitude range: {latitude.min()} to {latitude.max()}")
            print(f"Longitude range: {longitude.min()} to {longitude.max()}")
            
            # If no specific point is provided, use the center point
            if lat_point is None:
                lat_point = (latitude.min() + latitude.max()) / 2
            if lon_point is None:
                lon_point = (longitude.min() + longitude.max()) / 2
            
            print(f"Extracting data for point: lat={lat_point}, lon={lon_point}")
            
            # Find nearest grid points
            lat_idx = np.abs(latitude - lat_point).argmin()
            lon_idx = np.abs(longitude - lon_point).argmin()
            
            print(f"Nearest grid point indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
            print(f"Grid point coordinates: lat={latitude[lat_idx]}, lon={longitude[lon_idx]}")
            
            # Extract data for each variable at the selected point
            data_dict = {'time': time}
            
            for var in vars_to_extract:
                var_data = ds[var]
                print(f"Processing variable '{var}' with shape {var_data.shape}")
                
                # Extract data at the specific point
                # Assuming 3D data (time, lat, lon)
                if var_data.ndim == 3:
                    point_data = var_data[:, lat_idx, lon_idx].values
                elif var_data.ndim == 2:
                    # If only 2D, assume it's (lat, lon) and create a time series with same value
                    point_data = np.full(len(time), var_data[lat_idx, lon_idx].values)
                else:
                    print(f"Unexpected dimensionality for {var}: {var_data.ndim}")
                    continue
                
                data_dict[var] = point_data
            
            # Create DataFrame
            df = pd.DataFrame(data_dict)
            
            # Convert time to readable format if needed
            if df['time'].dtype == 'datetime64[ns]':
                df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            return df
                
    except Exception as e:
        print(f"Error extracting data from GRIB file {grib_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_all_grib_data(grib_dir="dataset/ERA5-hourly-data/", output_csv="dataset/era5_hourly_data.csv"):
    """
    Extract data from all GRIB files in a directory and save as a single CSV.
    
    Args:
        grib_dir (str): Directory containing GRIB files
        output_csv (str): Path to output CSV file
    """
    print(f"Processing all GRIB files in directory: {grib_dir}")
    
    # Find all GRIB files
    grib_pattern = os.path.join(grib_dir, "*.grib")
    grib_files = glob.glob(grib_pattern)
    
    if not grib_files:
        print(f"No GRIB files found in {grib_dir}")
        return
    
    print(f"Found {len(grib_files)} GRIB files")
    
    # Process each file and collect data
    all_data = []
    
    for grib_file in grib_files:
        print(f"\nProcessing file: {os.path.basename(grib_file)}")
        df = extract_grib_data(grib_file)
        if df is not None:
            all_data.append(df)
            print(f"Extracted {len(df)} records from {os.path.basename(grib_file)}")
        else:
            print(f"Failed to extract data from {os.path.basename(grib_file)}")
    
    if not all_data:
        print("No data extracted from any files")
        return
    
    # Combine all data
    print(f"\nCombining data from {len(all_data)} files")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by time if it exists
    if 'time' in combined_df.columns:
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    # Remove duplicates if any
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    final_count = len(combined_df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate records")
    
    # Save to CSV
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except:
        # If we can't create the directory, just use the current directory
        output_csv = os.path.basename(output_csv)
    
    combined_df.to_csv(output_csv, index=False)
    print(f"\nCombined data saved to: {output_csv}")
    print(f"Total records: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Show first few rows
    print("\nFirst few rows:")
    print(combined_df.head())

if __name__ == "__main__":
    # Process all GRIB files in the directory
    extract_all_grib_data()