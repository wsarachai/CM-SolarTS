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
import platform
import xarray as xr
import pandas as pd
import numpy as np
import os
import glob
import location

def extract_grib_data(grib_path, lat_point=None, lon_point=None):
    if os.path.exists(grib_path):                    
        print(f"Extracting ERA5-hourly data for coordinates ({lat_point}, {lon_point})")
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
    
    df_rows = []

    try:
        # Open the GRIB file
        backend_kwargs = {'indexpath': ''}
        with xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=backend_kwargs) as ds:
            print(f"\nDataset dimensions: {list(ds.dims.keys())}")
            print(f"Dataset variables: {list(ds.data_vars.keys())}")
        
            latitude = ds['latitude'].values
            longitude = ds['longitude'].values
            print(f"  Lat range: {latitude.min():.3f} to {latitude.max():.3f}")
            print(f"  Lon range: {longitude.min():.3f} to {longitude.max():.3f}")
            
            available_vars = [v for v in ds.data_vars]
            print(f"  Available variables (xarray): {available_vars}")

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
                
                # Extract time coordinate
                time = ds['time'].values
            
                for t_i, base_time in enumerate(time):
                    if isinstance(base_time, np.datetime64):
                        # Convert to pandas datetime
                        base_dt = pd.to_datetime(base_time)
                    else:
                        base_dt = pd.to_datetime(base_time)

                    data_dict = {
                        'datetime': base_dt
                    }

                    for var in available_vars:
                        var_data = ds[var]
                        print(f"Processing variable '{var}' with shape {var_data.shape}")
                        
                        data_array = ds[var][t_i, lat_indices, lon_indices]
                        weighted_data = data_array * weights
                        averaged_data = weighted_data.sum(axis=(0, 1))
                        data_dict[var] = float(averaged_data)
                    
                    df_rows.append(data_dict)
        
        # Merge rows by datetime (combine variable columns)
        df = pd.DataFrame(df_rows)
        df = df.groupby('datetime').first().reset_index()
        return df

    except Exception as e:
        print(f"Error extracting data from GRIB file {grib_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_all_grib_data(all_files, lat_point, lon_point, output_csv):
    # Process each file and collect data
    all_data = []
    
    for grib_file in all_files:
        print(f"\nProcessing file: {os.path.basename(grib_file)}")
        df = extract_grib_data(grib_file, lat_point=lat_point, lon_point=lon_point)
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
    if 'datetime' in combined_df.columns:
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    
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
    netcdf_dir = 'D:\\_datasets\\weather-dataset\\ERA5-hourly-data\\hourly'
    aggregated_out = 'dataset/era5-hourly_timeseries.csv'

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

    # Process all GRIB files in the directory
    extract_all_grib_data(all_files, lat_point=lat_point, lon_point=lon_point, output_csv=aggregated_out)