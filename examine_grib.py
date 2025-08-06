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
    
    try:
        # Try using pygrib
        import pygrib
        grbs = pygrib.open(grib_path)
        
        # Get all messages
        messages = grbs.messages
        print(f"\nNumber of GRIB messages: {messages}")
        
        # Get keys for the first message
        if messages > 0:
            grb = grbs.message(1)
            keys = grb.keys()
            print(f"\nAvailable keys ({len(keys)}):")
            for i, key in enumerate(keys[:20]):  # Show first 20 keys
                print(f"  {key}")
            
            if len(keys) > 20:
                print(f"  ... and {len(keys) - 20} more")
            
            # Show some key values
            print("\nKey values from first message:")
            try:
                print(f"  Name: {grb.name}")
                print(f"  Parameter: {grb.parameterName}")
                print(f"  Level: {grb.level}")
                print(f"  Type of level: {grb.typeOfLevel}")
                print(f"  Data date: {grb.dataDate}")
                print(f"  Data time: {grb.dataTime}")
                print(f"  Forecast time: {grb.forecastTime}")
                print(f"  Units: {grb.units}")
            except Exception as e:
                print(f"  Error getting key values: {e}")
        
        grbs.close()
        
    except ImportError:
        print("\npygrib not available. Trying with cfgrib...")
        try:
            import xarray as xr
            ds = xr.open_dataset(grib_path, engine='cfgrib')
            print(f"\nDataset dimensions: {list(ds.dims.keys())}")
            print(f"Dataset variables: {list(ds.data_vars.keys())}")
            print(f"Dataset coordinates: {list(ds.coords.keys())}")
            
            # Show some basic info about the first variable
            if len(ds.data_vars) > 0:
                first_var = list(ds.data_vars.keys())[0]
                print(f"\nFirst variable '{first_var}' info:")
                print(f"  Shape: {ds[first_var].shape}")
                print(f"  Dimensions: {ds[first_var].dims}")
                if hasattr(ds[first_var], 'units'):
                    print(f"  Units: {ds[first_var].units}")
                if hasattr(ds[first_var], 'long_name'):
                    print(f"  Long name: {ds[first_var].long_name}")
        except ImportError:
            print("Neither pygrib nor cfgrib/xarray available. Cannot examine GRIB file.")
        except Exception as e:
            print(f"Error examining GRIB file with cfgrib: {e}")
    
    except Exception as e:
        print(f"Error examining GRIB file with pygrib: {e}")

if __name__ == "__main__":
    grib_file = "data/download.grib"
    if os.path.exists(grib_file):
        examine_grib_file(grib_file)
    else:
        print(f"GRIB file not found: {grib_file}")