# 📊 GRIB Attribute Extractor
# This script extracts all attributes from GRIB files and saves them as CSV files.
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
import json
from collections import defaultdict

# Variables we're interested in
dataVars = ['cc', 'r', 'q', 't', 'u', 'v']

def extract_grib_attributes(grib_path, output_csv=None):
    """
    Extract all attributes from a GRIB file and save them as CSV.
    
    Args:
        grib_path (str): Path to the GRIB file
        output_csv (str): Path to output CSV file (optional)
    
    Returns:
        pandas.DataFrame: DataFrame containing all extracted attributes
    """
    print(f"Extracting attributes from GRIB file: {grib_path}")
    
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(grib_path))[0]
        output_csv = f"dataset/{base_name}_attributes.csv"
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    except:
        # If we can't create the directory, just use the current directory
        output_csv = os.path.basename(output_csv)
    
    try:
        # Open the GRIB file
        with xr.open_dataset(grib_path, engine='cfgrib') as ds:
            print(f"\nDataset dimensions: {list(ds.dims.keys())}")
            print(f"Dataset variables: {list(ds.data_vars.keys())}")
            print(f"Dataset coordinates: {list(ds.coords.keys())}")
            
            # Collect all attributes
            attributes_data = []
            
            # 1. Dataset-level attributes
            print("\n--- Dataset Attributes ---")
            dataset_attrs = {}
            dataset_attrs['level'] = 'dataset'
            dataset_attrs['variable'] = 'global'
            for attr_name, attr_value in ds.attrs.items():
                dataset_attrs[attr_name] = str(attr_value)
                print(f"  {attr_name}: {attr_value}")
            attributes_data.append(dataset_attrs)
            
            # 2. Coordinate-level attributes
            print("\n--- Coordinate Attributes ---")
            for coord_name in ds.coords.keys():
                coord = ds.coords[coord_name]
                coord_attrs = {}
                coord_attrs['level'] = 'coordinate'
                coord_attrs['variable'] = coord_name
                coord_attrs['shape'] = str(coord.shape)
                coord_attrs['dtype'] = str(coord.dtype)
                
                for attr_name, attr_value in coord.attrs.items():
                    coord_attrs[attr_name] = str(attr_value)
                    print(f"  {coord_name}.{attr_name}: {attr_value}")
                
                attributes_data.append(coord_attrs)
            
            # 3. Variable-level attributes
            print("\n--- Variable Attributes ---")
            for var_name in ds.data_vars.keys():
                var = ds[var_name]
                var_attrs = {}
                var_attrs['level'] = 'variable'
                var_attrs['variable'] = var_name
                var_attrs['shape'] = str(var.shape)
                var_attrs['dtype'] = str(var.dtype)
                
                for attr_name, attr_value in var.attrs.items():
                    var_attrs[attr_name] = str(attr_value)
                    print(f"  {var_name}.{attr_name}: {attr_value}")
                
                attributes_data.append(var_attrs)
            
            # Convert to DataFrame
            if attributes_data:
                df = pd.DataFrame(attributes_data)
                # Reorder columns to put key identifiers first
                cols = ['level', 'variable', 'shape', 'dtype'] + [c for c in df.columns if c not in ['level', 'variable', 'shape', 'dtype']]
                df = df[cols]
                
                # Save to CSV
                df.to_csv(output_csv, index=False)
                print(f"\nAttributes saved to: {output_csv}")
                print(f"Total records: {len(df)}")
                return df
            else:
                print("No attributes found to save.")
                return None
                
    except Exception as e:
        print(f"Error extracting attributes from GRIB file: {e}")
        return None

def extract_grib_attributes_detailed(grib_path, output_dir=None):
    """
    Extract detailed attributes from a GRIB file and save them as multiple CSV files.
    
    Args:
        grib_path (str): Path to the GRIB file
        output_dir (str): Directory to save output CSV files (optional)
    
    Returns:
        dict: Dictionary containing paths to created CSV files
    """
    print(f"Extracting detailed attributes from GRIB file: {grib_path}")
    
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(grib_path))[0]
        output_dir = f"dataset/{base_name}_attributes"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except:
        output_dir = "."
    
    try:
        # Open the GRIB file
        with xr.open_dataset(grib_path, engine='cfgrib') as ds:
            output_files = {}
            
            # 1. Dataset attributes
            dataset_attrs = []
            for attr_name, attr_value in ds.attrs.items():
                dataset_attrs.append({
                    'attribute': attr_name,
                    'value': str(attr_value),
                    'type': type(attr_value).__name__
                })
            
            if dataset_attrs:
                df_dataset = pd.DataFrame(dataset_attrs)
                dataset_csv = os.path.join(output_dir, "dataset_attributes.csv")
                df_dataset.to_csv(dataset_csv, index=False)
                output_files['dataset'] = dataset_csv
                print(f"Dataset attributes saved to: {dataset_csv}")
            
            # 2. Coordinate attributes
            coord_attrs_list = []
            for coord_name in ds.coords.keys():
                coord = ds.coords[coord_name]
                coord_info = {
                    'coordinate': coord_name,
                    'shape': str(coord.shape),
                    'dtype': str(coord.dtype),
                    'dimensions': str(coord.dims)
                }
                
                # Add coordinate-specific attributes
                for attr_name, attr_value in coord.attrs.items():
                    coord_info[f'attr_{attr_name}'] = str(attr_value)
                
                coord_attrs_list.append(coord_info)
            
            if coord_attrs_list:
                df_coords = pd.DataFrame(coord_attrs_list)
                coords_csv = os.path.join(output_dir, "coordinate_attributes.csv")
                df_coords.to_csv(coords_csv, index=False)
                output_files['coordinates'] = coords_csv
                print(f"Coordinate attributes saved to: {coords_csv}")
            
            # 3. Variable attributes
            var_attrs_list = []
            for var_name in ds.data_vars.keys():
                var = ds[var_name]
                var_info = {
                    'variable': var_name,
                    'shape': str(var.shape),
                    'dtype': str(var.dtype),
                    'dimensions': str(var.dims)
                }
                
                # Add variable-specific attributes
                for attr_name, attr_value in var.attrs.items():
                    var_info[f'attr_{attr_name}'] = str(attr_value)
                
                var_attrs_list.append(var_info)
            
            if var_attrs_list:
                df_vars = pd.DataFrame(var_attrs_list)
                vars_csv = os.path.join(output_dir, "variable_attributes.csv")
                df_vars.to_csv(vars_csv, index=False)
                output_files['variables'] = vars_csv
                print(f"Variable attributes saved to: {vars_csv}")
            
            # 4. Summary file
            summary_data = [
                {'category': 'dataset', 'count': 1, 'file': dataset_csv if dataset_attrs else 'N/A'},
                {'category': 'coordinates', 'count': len(coord_attrs_list), 'file': coords_csv if coord_attrs_list else 'N/A'},
                {'category': 'variables', 'count': len(var_attrs_list), 'file': vars_csv if var_attrs_list else 'N/A'}
            ]
            df_summary = pd.DataFrame(summary_data)
            summary_csv = os.path.join(output_dir, "summary.csv")
            df_summary.to_csv(summary_csv, index=False)
            output_files['summary'] = summary_csv
            print(f"Summary saved to: {summary_csv}")
            
            return output_files
                
    except Exception as e:
        print(f"Error extracting detailed attributes from GRIB file: {e}")
        return {}

if __name__ == "__main__":
    # Path to the GRIB file
    grib_file_path = "dataset/ERA5-hourly-data/5a732b807ec056db47aace313c25a9ac.grib"
    
    # Check if the file exists
    if os.path.exists(grib_file_path):
        # Extract basic attributes
        df = extract_grib_attributes(grib_file_path)
        
        # Extract detailed attributes
        output_files = extract_grib_attributes_detailed(grib_file_path)
        
        if df is not None:
            print("\nFirst few rows of extracted attributes:")
            print(df.head())
    else:
        print(f"Error: GRIB file not found at {grib_file_path}")
        # Try to find any GRIB file in the directory
        grib_dir = "dataset/ERA5-hourly-data/"
        if os.path.exists(grib_dir):
            grib_files = [f for f in os.listdir(grib_dir) if f.endswith('.grib')]
            if grib_files:
                first_grib = os.path.join(grib_dir, grib_files[0])
                print(f"Trying first available GRIB file: {first_grib}")
                df = extract_grib_attributes(first_grib)
                output_files = extract_grib_attributes_detailed(first_grib)
            else:
                print(f"No GRIB files found in {grib_dir}")
        else:
            print(f"GRIB directory not found: {grib_dir}")