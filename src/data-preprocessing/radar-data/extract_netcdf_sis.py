import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
import glob
import datetime as _dt
import location

# Convert possible cftime objects to ISO strings where possible
def _to_iso(t):
    if isinstance(t, str):
        return t
    if isinstance(t, _dt.datetime):
        return t.isoformat()
    if hasattr(t, 'year') and hasattr(t, 'month') and hasattr(t, 'day'):
        try:
            hour = int(getattr(t, 'hour', 0))
            minute = int(getattr(t, 'minute', 0))
            second = int(getattr(t, 'second', 0))
            micro = int(getattr(t, 'microsecond', 0))
            return _dt.datetime(int(t.year), int(t.month), int(t.day), hour, minute, second, micro).isoformat()
        except Exception:
            return str(t)
    return str(t)

def extract_netcdf_data(nc_path, out_txt='dataset/sis_selected_timeseries.csv'):
    print(f"Extracting data from NetCDF file: {nc_path}")

    lat_point=18.899741434351892 
    lon_point=99.01248957594561
    
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

            # Check if 'utci' variable exists in the file
            for var_name in ['lon', 'lat', 'time', 'SIS']:
                if var_name not in ds.variables:
                    print(f"  WARNING: 'utci' variable not found in file. Skipping...")
                    print(f"  Available variables: {list(ds.variables.keys())}")
                    return None

            sis_var = ds.variables['SIS'] if 'SIS' in ds.variables else None
            lowacc_sis_var = ds.variables['LowAcc_SIS'] if 'LowAcc_SIS' in ds.variables else None
            siscls_var = ds.variables['SISCLS'] if 'SISCLS' in ds.variables else None
            sis_stdv_var = ds.variables['SIS_stdv'] if 'SIS_stdv' in ds.variables else None
            sis_nobs_var = ds.variables['SIS_nobs'] if 'SIS_nobs' in ds.variables else None

            time = ds.variables['time'][:]
            # Get coordinate ranges
            latitude = ds.variables['lat'][:]
            longitude = ds.variables['lon'][:]
        
            print(f"  Lat range: {latitude.min():.3f} to {latitude.max():.3f}")
            print(f"  Lon range: {longitude.min():.3f} to {longitude.max():.3f}")
            
            # Check if coordinates are within range
            if not (latitude.min() <= lat_point <= latitude.max() and 
                   longitude.min() <= lon_point <= longitude.max()):
                print(f"  WARNING: Coordinates ({lat_point}, {lon_point}) outside data range")
                return None
            
            # Compute 1km² spatial average centered on the found point
            try:
                lat_indices, lon_indices, distances = location.find_nearest_point(latitude, longitude, lat_point, lon_point)
                print(f"  Spatial average computed over {len(lat_indices)} points")
                print(f"  Grid points used: {len(lat_indices)}")
                print(f"  Resolution: ~{np.mean(distances):.3f} km")

                # Extract UTCI data for each point in the circle
                n_times = sis_var.shape[0]
                n_points = len(lat_indices)
                
                # Pre-allocate array for sis data at selected points
                sis_data = np.zeros((n_times, n_points))
                lowAccSis_data = np.zeros((n_times, n_points))
                siscls_data = np.zeros((n_times, n_points))
                sis_stdv_data = np.zeros((n_times, n_points))
                sis_nobs_data = np.zeros((n_times, n_points))
                
                # Extract data for each point
                for i, (lat_idx, lon_idx) in enumerate(zip(lat_indices, lon_indices)):
                    sis_data[:, i] = sis_var[:, lat_idx, lon_idx] if sis_var is not None else sis_data[:, i]
                    lowAccSis_data[:, i] = lowacc_sis_var[:, lat_idx, lon_idx] if lowacc_sis_var is not None else lowAccSis_data[:, i]
                    siscls_data[:, i] = siscls_var[:, lat_idx, lon_idx] if siscls_var is not None else siscls_data[:, i]
                    sis_stdv_data[:, i] = sis_stdv_var[:, lat_idx, lon_idx] if sis_stdv_var is not None else sis_stdv_data[:, i]
                    sis_nobs_data[:, i] = sis_nobs_var[:, lat_idx, lon_idx] if sis_nobs_var is not None else sis_nobs_data[:, i]
                
                # Calculate inverse distance weights for spatial averaging
                # Points closer to center get higher weights
                max_dist = np.max(distances)
                inverse_distances = max_dist - distances + 1e-10  # Add small value to avoid division by zero
                weights = inverse_distances / np.sum(inverse_distances)
                
                # Apply weighted average across all points
                weighted = np.dot(sis_data, weights)
                print(f"  Extracted SIS data shape: {weighted.shape}")

                # Try decode time units if numeric -> may return cftime objects
                try:
                    times_raw = nc.num2date(time, units=ds.variables['time'].units)
                except Exception:
                    times_raw = time

                times_iso = [_to_iso(t) for t in np.atleast_1d(times_raw)]

                # Prepare output file. Write header if file does not exist yet.
                header = 'time,sis,lowAccSis,siscls,sis_stdv,sis_nobs\n'
                write_header = not os.path.exists(out_txt)
                with open(out_txt, 'a', encoding='utf-8') as fh:
                    if write_header:
                        fh.write(header)
                    for ti, sis_val, lowAccSis_val, siscls_val, sis_stdv_val, sis_nobs_val in zip(times_iso, weighted, weighted, weighted, weighted, weighted):
                        if np.isnan(sis_val):
                            fh.write(f"{ti},nan,nan,nan,nan,nan\n")
                        else:
                            fh.write(f"{ti},{float(sis_val):.6f},{float(lowAccSis_val):.6f},{float(siscls_val):.6f},{float(sis_stdv_val):.6f},{float(sis_nobs_val):.6f}\n")

                print(f"Appended timeseries to {out_txt}")

            except ValueError as e:
                print(f"  WARNING: Spatial averaging failed: {e}")
                print("  Using single point data instead")
                return None
            
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check for a list of files to process
    #files_list_path = 'dataset/sis-files.txt'
    netcdf_dir = '/Volumes/Seagate/_datasets/weather-dataset/SIS-Surface-radiation-budget/SISdm'
    aggregated_out = 'dataset/sis_selected_timeseries.csv'

    # Remove existing aggregated output so we start fresh
    if os.path.exists(aggregated_out):
        try:
            os.remove(aggregated_out)
        except Exception:
            pass

    netcdf_pattern = os.path.join(netcdf_dir, "*.nc")
    netcdf_files = glob.glob(netcdf_pattern)
    
    print(f"Found {len(netcdf_files)} NetCDF files")

    # Process each file and collect data
    all_files = []

    for netcdf_file in netcdf_files:
        print(f"\nProcessing file: {os.path.basename(netcdf_file)}")
        all_files.append(netcdf_file)

    for f in all_files:
        if os.path.exists(f):
            print(f"Processing {f}...")
            extract_netcdf_data(f, out_txt=aggregated_out)
        else:
            print(f"Warning: listed file not found: {f}")

        print(f"All done. Aggregated results in {aggregated_out}")
        