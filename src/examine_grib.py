import os
import sys
import xarray as xr
import pandas as pd

# Check if a file path was provided
if len(sys.argv) < 2:
    print("Usage: python examine_grib.py <grib_file_path>")
    exit(1)

grib_file = sys.argv[1]

# Check if the file exists
if not os.path.exists(grib_file):
    print(f"Error: {grib_file} not found")
    exit(1)

# Try to open with xarray and cfgrib
try:
    print(f"Opening {grib_file} with xarray and cfgrib...")
    ds = xr.open_dataset(grib_file, engine='cfgrib')
    print("\nDataset info:")
    print(ds)
    
    print("\nVariables:")
    for var in ds.variables:
        print(f"- {var}: {ds[var].attrs}")
    
    print("\nDimensions:")
    for dim in ds.dims:
        print(f"- {dim}: {len(ds[dim])}")
    
    print("\nAttributes:")
    for attr in ds.attrs:
        print(f"- {attr}: {ds.attrs[attr]}")
        
except Exception as e:
    print(f"Error opening with cfgrib: {e}")
    
    # Try to get basic file information
    file_size = os.path.getsize(grib_file)
    print(f"\nFile size: {file_size} bytes")
    
    # Try to read first few bytes to check file type
    with open(grib_file, 'rb') as f:
        header = f.read(8)
        print(f"File header: {header}")