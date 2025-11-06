#!/usr/bin/env python3
"""
ERA5-Land meteorological data extraction for specific coordinates
Coordinates: 18.899741434351892, 99.01248957594561

Output: CSV with merged datetime column (YYYY-MM-DD HH:MM:SS)
One row per datetime, continuous time series with spatial averaging
"""

import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Target meteorological variables to extract
TARGET_VARS = ['t2m', 'fal', 'slhf', 'ssr', 'str', 'sshf', 'ssrd', 'strd', 'u10', 'v10', 'sp', 'tp']

def calculate_spatial_bounds(lat_center, lon_center, area_km2=1.0):
    """
    Calculate degree-based bounding box for a specified area in km².
    
    Uses accurate coordinate conversions accounting for latitude dependence
    of longitude distance (cosine effect).
    
    Args:
        lat_center (float): Center latitude in decimal degrees
        lon_center (float): Center longitude in decimal degrees
        area_km2 (float): Area in square kilometers (default: 1.0 km²)
        
    Returns:
        tuple: (lat_min, lat_max, lon_min, lon_max) bounds in decimal degrees
    """
    # Convert to radians for calculations
    lat_rad = np.radians(lat_center)
    
    # Earth's radius in kilometers
    # earth_radius_km = 6371.0
    
    # Calculate side length for square area
    side_length_km = np.sqrt(area_km2)
    
    # Calculate latitude bounds (±distance in km / 111 km/degree)
    lat_half_deg = side_length_km / 111.0
    lat_min = lat_center - lat_half_deg
    lat_max = lat_center + lat_half_deg
    
    # Calculate longitude bounds (account for latitude cos factor)
    # At equator: 1° longitude ≈ 111 km
    # At latitude φ: 1° longitude ≈ 111 * cos(φ) km
    lon_half_deg = side_length_km / (111.0 * np.cos(lat_rad))
    lon_min = lon_center - lon_half_deg
    lon_max = lon_center + lon_half_deg
    
    # Handle polar regions and numerical precision
    lat_min = max(-90.0, min(90.0, lat_min))
    lat_max = max(-90.0, min(90.0, lat_max))
    
    # Handle longitude wraparound
    while lon_min < -180.0:
        lon_min += 360.0
        lon_max += 360.0
    while lon_max > 180.0:
        lon_min -= 360.0
        lon_max -= 360.0
    
    return lat_min, lat_max, lon_min, lon_max

def extract_spatial_average_data(ds, lat_center, lon_center, area_km2=1.0):
    """
    Extract spatial average data from xarray dataset for specified area.
    
    Uses efficient xarray slicing and coordinate interpolation to compute
    spatial averages over a bounding box.
    
    Args:
        ds (xarray.Dataset): Input dataset with dimensions ['time', 'step', 'latitude', 'longitude']
        lat_center (float): Center latitude
        lon_center (float): Center longitude
        area_km2 (float): Area in square kilometers (default: 1.0 km²)
        
    Returns:
        tuple: (lat_bounds, lon_bounds, area_info_dict)
    """
    # Calculate spatial bounds
    lat_min, lat_max, lon_min, lon_max = calculate_spatial_bounds(lat_center, lon_center, area_km2)
    
    # Get coordinate arrays
    latitude = ds['latitude'].values
    longitude = ds['longitude'].values

    lat_min_idx, lon_min_idx, distance_min_km, min_is_within_1km2 = find_nearest_point_in_1km2(latitude, longitude, lat_min, lon_min)
    lat_max_idx, lon_max_idx, distance_min_km, max_is_within_1km2 = find_nearest_point_in_1km2(latitude, longitude, lat_max, lon_max)
    
    # Find the best point within 1km² area
    nearest_min_lat = latitude[lat_min_idx]
    nearest_min_lon = longitude[lon_min_idx]
    nearest_max_lat = latitude[lat_max_idx]
    nearest_max_lon = longitude[lon_max_idx]
    
    # Print detailed location information
    print(f"  Target coordinates: ({lat_min:.6f}, {lon_min:.6f}, {lat_max:.6f}, {lon_max:.6f})")
    print(f"  Target coordinates: ({nearest_min_lat:.6f}, {nearest_min_lon:.6f}, {nearest_max_lat:.6f}, {nearest_max_lon:.6f})")
    print(f"  Min distance from target: {distance_min_km:.3f} km")
    print(f"  Max distance from target: {distance_min_km:.3f} km")
    
    if min_is_within_1km2:
        print(f"  Found min grid point within 1km² area")
    else:
        print(f"  Warning: Nearest min point is outside 1km² area")

    if max_is_within_1km2:
        print(f"  Found max grid point within 1km² area")
    else:
        print(f"  Warning: Nearest max point is outside 1km² area")
            
            
    # Find indices for spatial selection (with buffer for interpolation)
    lat_indices = np.where((latitude >= nearest_min_lat) & (latitude <= nearest_max_lat))[0]
    lon_indices = np.where((longitude >= nearest_min_lon) & (longitude <= nearest_max_lon))[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        raise ValueError(f"No data points found within bounds: lat[{lat_min:.6f}, {lat_max:.6f}], lon[{lon_min:.6f}, {lon_max:.6f}]")
    
    # Extend selection slightly for better interpolation
    lat_buffer = max(1, min(3, len(lat_indices)))
    lon_buffer = max(1, min(3, len(lon_indices)))
    
    lat_start = max(0, lat_indices[0] - lat_buffer)
    lat_end = min(len(latitude), lat_indices[-1] + lat_buffer + 1)
    lon_start = max(0, lon_indices[0] - lon_buffer)
    lon_end = min(len(longitude), lon_indices[-1] + lon_buffer + 1)
    
    # Extract subset for spatial processing
    lat_subset = latitude[lat_start:lat_end]
    lon_subset = longitude[lon_start:lon_end]
    
    # Calculate area information
    lat_span_km = (lat_max - lat_min) * 111.0
    lon_span_km = (lon_max - lon_min) * 111.0 * np.cos(np.radians(lat_center))
    actual_area_km2 = lat_span_km * lon_span_km
    
    area_info = {
        'lat_center': lat_center,
        'lon_center': lon_center,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'requested_area_km2': area_km2,
        'actual_area_km2': actual_area_km2,
        'lat_span_km': lat_span_km,
        'lon_span_km': lon_span_km,
        'grid_points_lat': len(lat_indices),
        'grid_points_lon': len(lon_indices)
    }
    
    return lat_start, lat_end, lon_start, lon_end, lat_subset, lon_subset, area_info

def extract_and_average_area(ds, lat_point, lon_point, target_vars=None, area_km2=1.0):
    """
    Extract and compute spatial averages for a specified area around given coordinates.
    
    Args:
        ds (xarray.Dataset): Input dataset with dimensions ['time', 'step', 'latitude', 'longitude']
        lat_point (float): Center latitude
        lon_point (float): Center longitude
        target_vars (list): List of variable names to process (optional, processes all if None)
        area_km2 (float): Area in square kilometers (default: 1.0 km²)
        
    Returns:
        dict: Dictionary containing 'metadata' and 'data' keys
              - 'metadata': Dictionary with area information
              - 'data': xarray.Dataset with spatially averaged data
    """
    
    # Use all available variables if none specified
    if target_vars is None:
        target_vars = [v for v in ds.data_vars if v in TARGET_VARS]
    
    # Calculate spatial bounds and get subset indices
    lat_start_idx, lat_end_idx, lon_start_idx, lon_end_idx, lat_subset, lon_subset, area_info = extract_spatial_average_data(ds, lat_point, lon_point, area_km2)

    # Extract subset data for processing
    spatial_subset = ds.isel(
        latitude=slice(lat_start_idx, lat_end_idx), 
        longitude=slice(lon_start_idx, lon_end_idx)
    )

    # Create coordinate meshes for area-weighted averaging
    lat_mesh, lon_mesh = np.meshgrid(lat_subset, lon_subset, indexing='ij')

    lat_mesh_rad = np.radians(lat_mesh)
    lat_distances = (lat_mesh - lat_point) * 111.0  # km
    lon_distances = (lon_mesh - lon_point) * 111.0 * np.cos(lat_mesh_rad)  # km
    distances = np.sqrt(lat_distances**2 + lon_distances**2)

    # Inverse distance weighting
    inverse_distances = np.abs(distances - np.max(distances))
    # Normalize inverse distances
    weights = inverse_distances / np.sum(inverse_distances) if np.sum(inverse_distances) > 0 else 1.0

    # Store processed variables
    processed_data = {}
    processed_vars = []
    
    for var in target_vars:
        if var in spatial_subset.data_vars:
            var_data = spatial_subset[var]
            
            if var_data.ndim == 4:
                # 4D data: (time, step, latitude, longitude)
                # Compute weighted average over spatial dimensions
                # Reshape for broadcasting
                weights_expanded = weights[np.newaxis, np.newaxis, :, :]
                weighted_data = var_data * weights_expanded
                
                # Sum over spatial dimensions (latitude, longitude)
                averaged_data = weighted_data.sum(dim=['latitude', 'longitude'])
                processed_data[var] = averaged_data
                processed_vars.append(var)
                
            elif var_data.ndim == 3:
                # 3D data: (time, latitude, longitude)
                weights_expanded = weights[np.newaxis, :, :]
                weighted_data = var_data * weights_expanded
                averaged_data = weighted_data.sum(dim=['latitude', 'longitude'])
                processed_data[var] = averaged_data
                processed_vars.append(var)
                
            elif var_data.ndim == 2:
                # 2D data: (latitude, longitude) - single timestep
                weighted_data = var_data * weights
                averaged_data = weighted_data.sum()
                processed_data[var] = averaged_data
                processed_vars.append(var)
            else:
                print(f"  WARNING: Skipping {var} - unsupported dimensions: {var_data.ndim}")
    
    # Create result dataset with processed variables
    if processed_data:
        result_data = xr.Dataset(processed_data)
    else:
        print("  WARNING: No variables were successfully processed")
        result_data = xr.Dataset()
    
    # Add metadata information
    result_metadata = {
        'actual_area_km2': area_info['actual_area_km2'],
        'grid_points': weights.size,
        'resolution_km': np.sqrt(area_info['actual_area_km2'] / np.pi),
        'center_lat': lat_point,
        'center_lon': lon_point,
        'processed_variables': processed_vars,
        'area_bounds': {
            'lat_min': area_info['lat_min'],
            'lat_max': area_info['lat_max'],
            'lon_min': area_info['lon_min'],
            'lon_max': area_info['lon_max']
        },
        'weighted_average': True,
        'area_weighting': 'cosine_latitude'
    }
    
    return {
        'metadata': result_metadata,
        'data': result_data
    }

def _intelligent_merge_duplicates(df):
    """
    Comprehensive merge strategy for duplicate rows based on datetime.
    
    Args:
        df (pandas.DataFrame): Input dataframe with potential duplicates
        
    Returns:
        pandas.DataFrame: Dataframe with intelligently merged duplicates
    """
    print(f"  Analyzing duplicates for {len(df)} rows...")
    
    # Find duplicate datetime entries
    duplicate_groups = df.groupby('datetime')
    unique_datetimes = []
    
    for datetime_val, group in duplicate_groups:
        if len(group) == 1:
            # No duplicates for this datetime
            # Convert Series to dict for consistency
            row_dict = group.iloc[0].to_dict()
            unique_datetimes.append(row_dict)
        else:
            # Handle duplicates for this datetime
            print(f"  Found {len(group)} duplicates for datetime: {datetime_val}")
            merged_row = _merge_duplicate_group(group)
            unique_datetimes.append(merged_row)
    
    # Create merged dataframe
    if not unique_datetimes:
        print("  No duplicates found, returning original dataframe")
        return df
    
    print(f"  Merged duplicate groups")
    merged_df = pd.DataFrame(unique_datetimes).reset_index(drop=True)
    
    return merged_df

def _merge_duplicate_group(group):
    """
    Merge a group of duplicate rows sharing the same datetime.
    
    Args:
        group (pandas.DataFrame): Group of duplicate rows
        
    Returns:
        dict: Merged row data
    """
    # Start with the first row as base
    merged = {}
    
    # Preserve datetime from first row
    merged['datetime'] = group.iloc[0]['datetime']
    
    # Handle target variables with intelligent aggregation
    for var in TARGET_VARS:
        if var in group.columns:
            values = group[var].values
            # Remove NaN values for aggregation
            non_nan_values = values[~pd.isna(values)]
            
            if len(non_nan_values) == 0:
                # All values are NaN, preserve NaN
                merged[var] = np.nan
            elif len(non_nan_values) == 1:
                # Only one non-NaN value, use it
                merged[var] = non_nan_values[0]
            else:
                # Multiple non-NaN values, apply aggregation rules
                merged[var] = _aggregate_variable(var, non_nan_values)
    
    # Handle coordinate columns (should be same for duplicates)
    coord_columns = ['latitude', 'longitude', 'target_lat', 'target_lon', 'file']
    for col in coord_columns:
        if col in group.columns:
            # Take first non-NaN value or first value
            merged[col] = group[col].dropna().iloc[0] if not group[col].dropna().empty else group[col].iloc[0]
    
    return merged

def _aggregate_variable(var_name, values):
    """
    Apply intelligent aggregation rules for different variable types.
    
    Args:
        var_name (str): Variable name
        values (array): Array of non-NaN values
        
    Returns:
        float: Aggregated value
    """
    # Define cumulative variables (should be summed)
    cumulative_vars = ['tp', 'ssr', 'str', 'ssrd', 'strd']
    
    # Define continuous variables (should be averaged)
    continuous_vars = ['t2m', 'u10', 'v10', 'sp', 'fal']
    
    # Define flux variables (special handling)
    flux_vars = ['slhf', 'sshf']
    
    if var_name in cumulative_vars:
        # For cumulative variables, sum the values
        return float(np.sum(values))
    elif var_name in continuous_vars:
        # For continuous variables, use mean
        return float(np.mean(values))
    elif var_name in flux_vars:
        # For flux variables, use mean but preserve units
        return float(np.mean(values))
    else:
        # Default to mean for unknown variables
        return float(np.mean(values))

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

def find_nearest_point_in_1km2(latitude, longitude, target_lat, target_lon):
    # Get grid resolution
    lat_res = np.abs(np.median(np.diff(latitude)))
    lon_res = np.abs(np.median(np.diff(longitude)))
    print(f"  Grid resolution: {lat_res:.6f}° lat, {lon_res:.6f}° lon")
    
    # Convert 1km to degrees at this latitude (approximate)
    km_to_lat = 1.0 / 111.0  # 1 degree ≈ 111 km
    km_to_lon = 1.0 / (111.0 * np.cos(np.radians(target_lat)))
    
    # Calculate search radius in grid points
    lat_radius = int(np.ceil(km_to_lat / lat_res))
    lon_radius = int(np.ceil(km_to_lon / lon_res))
    
    # Find nearest point first
    lat_idx = np.abs(latitude - target_lat).argmin()
    lon_idx = np.abs(longitude - target_lon).argmin()
    
    # Get window around nearest point
    lat_start = max(0, lat_idx - lat_radius)
    lat_end = min(len(latitude), lat_idx + lat_radius + 1)
    lon_start = max(0, lon_idx - lon_radius)
    lon_end = min(len(longitude), lon_idx + lon_radius + 1)
    
    # Create mask for points within window
    lat_points = latitude[lat_start:lat_end]
    lon_points = longitude[lon_start:lon_end]
    
    # Find all distances within window
    min_distance = float('inf')
    best_lat_idx = lat_idx
    best_lon_idx = lon_idx
    is_within_1km2 = False
    
    for i, lat in enumerate(lat_points):
        for j, lon in enumerate(lon_points):
            # Calculate true distance in km
            lat_diff_km = abs(lat - target_lat) * 111.0
            lon_diff_km = abs(lon - target_lon) * (111.0 * np.cos(np.radians(target_lat)))
            distance = np.sqrt(lat_diff_km**2 + lon_diff_km**2)
            
            if distance < min_distance:
                min_distance = distance
                best_lat_idx = lat_start + i
                best_lon_idx = lon_start + j
                # Due to ERA5-Land's 0.1° resolution (~11km), consider points within
                # half a grid cell (~5.5km) as "close enough" for the target area
                is_within_1km2 = distance <= 5.5  # Half grid cell
    
    print(f"  Search window: {lat_radius} points lat, {lon_radius} points lon")
    print(f"  Points checked: {len(lat_points) * len(lon_points)}")
    
    return best_lat_idx, best_lon_idx, min_distance, is_within_1km2
    
def extract_from_file(grib_path, lat_point, lon_point, target_vars=TARGET_VARS):
    """
    Extract ERA5-Land data for specific coordinates from a single GRIB file.
    Can either extract single point data or compute 1km² spatial averages.
    
    Args:
        grib_path (str): Path to GRIB file
        lat_point (float): Target latitude
        lon_point (float): Target longitude
        target_vars (list): List of variable names to extract
    
    Returns:
        pandas.DataFrame: Extracted data with merged datetime column
    """
    
    if not os.path.isabs(grib_path):
        grib_path = os.path.join(os.getcwd(), "dataset", "ERA5-land-hourly", grib_path)
    if os.path.exists(grib_path):                    
        print(f"Extracting ERA5-Land data for coordinates ({lat_point}, {lon_point})")
        print(f"Processing {grib_path}...")
        print("=" * 60)
    else:
        print(f"Warning: listed file not found: {grib_path}")

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
            
            # Compute 1km² spatial average centered on the found point
            try:
                results = extract_and_average_area(ds, lat_point, lon_point)
                metadata = results['metadata']
                print(f"  Spatial average computed over {metadata['actual_area_km2']:.3f} km²")
                print(f"  Grid points used: {metadata['grid_points']}")
                print(f"  Resolution: ~{metadata['resolution_km']:.3f} km")
            except ValueError as e:
                print(f"  WARNING: Spatial averaging failed: {e}")
                print("  Using single point data instead")
                return None
            
            # Extract data for each variable
            all_data = []
            average_data = results['data']
            
            for var in available_vars:
                var_data = average_data[var]
                print(f"  Processing {var}: shape={var_data.shape}, dims={var_data.dims}")
                
                if var_data.ndim == 2:
                    # 2D data (time, step), var_data was spatially averaged
                    time_vals = ds['time'].values
                    step_vals = ds['step'].values
                    
                    # Get data for nearest point - we'll handle the spatial average differently
                    print(f"  Processing {var} values shape: {var_data.shape}")

                    if np.isnan(var_data).all():
                        print(f"  WARNING: All values for {var} are NaN")

                    # Create datetime array
                    all_datetimes = merge_time_step_to_datetime(time_vals, step_vals)

                    # Store data
                    for i, dt in enumerate(all_datetimes):
                        time_idx = i // len(step_vals)
                        step_idx = i % len(step_vals)
                        
                        all_data.append({
                            'datetime': dt,
                            'variable': var,
                            'value': var_data[time_idx, step_idx]
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

def extract_all_coordinates(files, lat_point, lon_point, output_csv):
    # Process all files
    all_dataframes = []

    for grib_file in sorted(files):
        df = extract_from_file(grib_file, lat_point, lon_point)
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
    
    # Comprehensive merge strategy for duplicate rows
    print("Applying comprehensive merge strategy for duplicate rows...")
    combined_df = _intelligent_merge_duplicates(combined_df)
    
    # Sort by datetime
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
    # Check for a list of files to process
    files_list_path = 'dataset/era5-land-files.txt'
    aggregated_out = 'dataset/era5-land_timeseries.csv'

    lat_point=18.899741434351892 
    lon_point=99.01248957594561
    
    # Enable spatial averaging by default (1km² area)
    use_spatial_average = True

    print("=" * 80)         
    print("ERA5-Land Data Extraction")
    print("Specific coordinates: 18.899741434351892, 99.01248957594561")
    print("Output format: CSV with merged datetime column")
    print("Variables: time,t2m,fal,slhf,ssr,str,sshf,ssrd,strd,u10,v10,sp,tp")
    print("=" * 80)
     
    # Remove existing aggregated output so we start fresh
    if os.path.exists(aggregated_out):
        try:
            os.remove(aggregated_out)
        except Exception:
            pass

    if os.path.exists(files_list_path):
        with open(files_list_path, 'r', encoding='utf-8') as fh:
            files = [line.strip() for line in fh if line.strip()]
            print(f"Found {len(files)} GRIB files")
            print("=" * 60)

            # Extract data with spatial averaging
            result_df = extract_all_coordinates(files, lat_point, lon_point, output_csv=aggregated_out)
            
            if result_df is not None:
                print("\n" + "=" * 80)
                print("EXTRACTION COMPLETED SUCCESSFULLY!")
                print(f"Output file: dataset/era5_land_extracted.csv")
                print(f"Records: {len(result_df)}")
                print(f"Time range: {result_df['datetime'].min()} to {result_df['datetime'].max()}")
            else:
                print("\n" + "=" * 80)
                print("EXTRACTION FAILED!")
                    
        print(f"All done. Aggregated results in {aggregated_out}")
