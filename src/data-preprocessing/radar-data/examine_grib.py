#!/usr/bin/env python3
"""
Script to examine GRIB file contents
"""

import os
import sys

def examine_grib_file(grib_path):
    """Examine the structure and contents of a GRIB file"""
    print(f"Examining GRIB file: {grib_path}")
    print(f"File size: {os.path.getsize(grib_path) / (1024*1024):.2f} MB")
    
    print("\npygrib not available. Trying with cfgrib...")
    try:
        import xarray as xr
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        print(f"\nDataset dimensions: {list(ds.dims.keys())}")
        print(f"Dataset variables: {list(ds.data_vars.keys())}")
        print(f"Dataset coordinates: {list(ds.coords.keys())}")

        vars_to_plot = [v for v in ['ssrd', 't2m', 'u10', 'v10', 'tp'] if v in ds.data_vars]
        
        # Show some basic info about the first variable
        if len(vars_to_plot) > 0:
            for first_var in vars_to_plot:
                print(f"\nFirst variable '{first_var}' info:")
                print(f"  Shape: {ds[first_var].shape}")
                print(f"  Dimensions: {ds[first_var].dims}")
                if hasattr(ds[first_var], 'units'):
                    print(f"  Units: {ds[first_var].units}")
                if hasattr(ds[first_var], 'long_name'):
                    print(f"  Long name: {ds[first_var].long_name}")
    except ImportError:
        print("Neither pygrib nor cfgrib/xarray available. Cannot examine GRIB file.")

if __name__ == "__main__":
    grib_file_path = "dataset/ERA5-land-hourly-2019.grib"
    if os.path.exists(grib_file_path):
        examine_grib_file(grib_file_path)
    else:
        print(f"GRIB file not found: {grib_file_path}")