import netCDF4 as nc
import os

def examine_netcdf_file(nc_path):
    """
    Examine the contents of a NetCDF file.
    
    Args:
        nc_path (str): Path to the NetCDF file
    """
    print(f"Examining NetCDF file: {nc_path}")
    
    try:
        # Open the NetCDF file
        with nc.Dataset(nc_path, 'r') as ds:
            print("\n=== NetCDF File Information ===")
            print(f"File path: {ds.filepath()}")
            print(f"File format: {ds.file_format}")
            print(f"Disk format: {ds.disk_format}")
            
            # Print global attributes
            print("\n=== Global Attributes ===")
            for attr_name in ds.ncattrs():
                print(f"{attr_name}: {getattr(ds, attr_name)}")
            
            # Print dimensions
            print("\n=== Dimensions ===")
            for dim_name, dim in ds.dimensions.items():
                print(f"{dim_name}: {len(dim)}")
            
            # Print variables
            print("\n=== Variables ===")
            for var_name, var in ds.variables.items():
                print(f"\nVariable: {var_name}")
                print(f"  Dimensions: {var.dimensions}")
                print(f"  Shape: {var.shape}")
                print(f"  Data type: {var.dtype}")
                
                # Print variable attributes
                if var.ncattrs():
                    print("  Attributes:")
                    for attr_name in var.ncattrs():
                        print(f"    {attr_name}: {getattr(var, attr_name)}")
                
                # Print a sample of the data (if it's not too large)
                if var.size > 0 and var.size < 100:
                    print(f"  Data: {var[:]}")
                elif var.size > 0:
                    print(f"  Data sample: {var[:min(10, var.size)]}...")
    
    except Exception as e:
        print(f"Error examining NetCDF file: {e}")

if __name__ == "__main__":
    # Path to the NetCDF file
    netcdf_file_path = "data/z_cams_l_uor_201806_v2_rf_co2_srf_lw.nc"
    
    # Check if the file exists
    if os.path.exists(netcdf_file_path):
        examine_netcdf_file(netcdf_file_path)
    else:
        print(f"Error: NetCDF file not found at {netcdf_file_path}")