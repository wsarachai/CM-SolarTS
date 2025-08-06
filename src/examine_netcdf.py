import netCDF4 as nc
import numpy as np

# Open the NetCDF file
file_path = "z_cams_l_uor_201806_v2_rf_co2_srf_lw.nc"
try:
    print(f"Opening {file_path}...")
    ds = nc.Dataset(file_path, 'r')
    
    print("\n=== NetCDF File Information ===")
    print(f"File: {ds.filepath()}")
    print(f"Format: {ds.data_model}")
    
    print("\n=== Global Attributes ===")
    for attr_name in ds.ncattrs():
        print(f"{attr_name}: {getattr(ds, attr_name)}")
    
    print("\n=== Dimensions ===")
    for dim_name, dim in ds.dimensions.items():
        print(f"{dim_name}: size={len(dim)} (unlimited={dim.isunlimited()})")
    
    print("\n=== Variables ===")
    for var_name, variable in ds.variables.items():
        print(f"\nVariable: {var_name}")
        print(f"  Dimensions: {variable.dimensions}")
        print(f"  Shape: {variable.shape}")
        print(f"  Size: {variable.size}")
        print(f"  Datatype: {variable.dtype}")
        
        # Print variable attributes
        for attr_name in variable.ncattrs():
            print(f"  {attr_name}: {getattr(variable, attr_name)}")
        
        # For small arrays, print actual values
        if variable.size < 20:
            print(f"  Values: {variable[:]}")
        else:
            print(f"  Sample values: {variable[:5]}")
    
    # Close the dataset
    ds.close()
    print("\nNetCDF file examination completed successfully!")
    
except Exception as e:
    print(f"Error examining NetCDF file: {e}")