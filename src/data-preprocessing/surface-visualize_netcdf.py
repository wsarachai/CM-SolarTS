import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_netcdf_data(nc_path, out_txt='dataset/utci_selected_timeseries.txt'):
    """
    Visualize the data in a NetCDF file.
    
    Args:
        nc_path (str): Path to the NetCDF file
    """
    print(f"Visualizing NetCDF file: {nc_path}")
    
    try:
        # Open the NetCDF file
        with nc.Dataset(nc_path, 'r') as ds:
            # Extract the data
            time = ds.variables['time'][:]
            longitude = ds.variables['lon'][:]
            latitude = ds.variables['lat'][:]
            sisData = ds.variables['SIS']  # Surface Incoming Solar Radiation
            sisclrData = ds.variables.get('SISCLS')  # Clear-sky Surface Incoming Solar Radiation (optional)
            sisStdvData = ds.variables.get('SIS_stdv')  # Standard deviation (optional)

            print(f"SIS variable shape: {sisData.shape if sisData is not None else 'N/A'}")
            print(f"SISCLS variable shape: {sisclrData.shape if sisclrData is not None else 'N/A'}")
            print(f"SIS_stdv variable shape: {sisStdvData.shape if sisStdvData is not None else 'N/A'}")

            area = [float(latitude.min()), float(longitude.min()), float(latitude.max()), float(longitude.max())]
            print(f"Area of data: {area}")

            # Desired bounding box (lat_min, lon_min, lat_max, lon_max)
            lat_min = 18.000
            lat_max = 19.000
            lon_min = 98.000
            lon_max = 99.000

            # Find indices within the bounding box
            lat_inds = np.where((latitude >= lat_min) & (latitude <= lat_max))[0]
            lon_inds = np.where((longitude >= lon_min) & (longitude <= lon_max))[0]

            print(f"Found lat indices: {lat_inds}, lon indices: {lon_inds}")

            if lat_inds.size == 0 or lon_inds.size == 0:
                print("No grid points found inside the requested bounding box. Check coordinates or widen the box.")
                return

            # Convert sisData to a float numpy array and slice to (time, lat, lon)
            # sisData may be a masked array or contain non-numeric fill values (e.g. '?').
            try:
                sisRaw = sisData[:] # materializes the whole variable into memory (may be masked array)
                sisclrRaw = sisclrData[:] # materializes the whole variable into memory (may be masked array)
                sisStdvRaw = sisStdvData[:] # materializes the whole variable into memory (may be masked array)
            except Exception as e:
                print(f"Warning reading sis variable: {e}")
                # fallback: read into object array
                sisRaw = np.array([sisData[i] for i in range(len(sisData))], dtype=object) if sisData is not None else None
                sisclrRaw = np.array([sisclrData[i] for i in range(len(sisclrData))], dtype=object) if sisclrData is not None else None
                sisStdvRaw = np.array([sisStdvData[i] for i in range(len(sisStdvData))], dtype=object) if sisStdvData is not None else None

            # Robust coercion to float: try a direct astype, otherwise convert elementwise
            def robust_to_float(arr):
                try:
                    return np.asarray(arr, dtype=float)
                except Exception:
                    flat = np.asarray(arr).ravel()
                    out = np.empty(flat.shape, dtype=float)
                    for i, v in enumerate(flat):
                        try:
                            out[i] = float(v)
                        except Exception:
                            out[i] = np.nan
                    return out.reshape(np.asarray(arr).shape)

            sis_arr = robust_to_float(sisRaw)
            sisclr_arr = robust_to_float(sisclrRaw)
            sisStdv_arr = robust_to_float(sisStdvRaw)

            # Slice to the bounding box
            # Attempt to interpret sis_arr dimensions: assume (time, lat, lon)
            sis_time_lat_lon = sis_arr[:, lat_inds[:, None], lon_inds]
            sisclr_time_lat_lon = sisclr_arr[:, lat_inds[:, None], lon_inds]
            sisStdv_time_lat_lon = sisStdv_arr[:, lat_inds[:, None], lon_inds]

            # Ensure result is float and shaped (time, lat, lon)
            sis_time_lat_lon = sis_time_lat_lon.astype(float)
            sisclr_time_lat_lon = sisclr_time_lat_lon.astype(float)
            sisStdv_time_lat_lon = sisStdv_time_lat_lon.astype(float)

            # Quick visualization: plot mean over the small spatial box as a time series
            mean_sis = np.nanmean(sis_time_lat_lon, axis=(1, 2))
            mean_sisclr = np.nanmean(sisclr_time_lat_lon, axis=(1, 2))
            mean_sisStdv = np.nanmean(sisStdv_time_lat_lon, axis=(1, 2))
            print(f"Mean timeseries shape: {mean_sis.shape}")

            # Instead of plotting, write time,value pairs to a text file
            import datetime as _dt

            # Try decode time units if numeric -> may return cftime objects
            try:
                times_raw = nc.num2date(time, units=ds.variables['time'].units)
            except Exception:
                times_raw = time

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

            times_iso = [_to_iso(t) for t in np.atleast_1d(times_raw)]

            # Prepare output file. Write header if file does not exist yet.
            header = 'time,sis_mean,sisclr_mean,sisStdv_mean\n'
            write_header = not os.path.exists(out_txt)
            with open(out_txt, 'a', encoding='utf-8') as fh:
                if write_header:
                    fh.write(header)
                for ti, val, val2, val3 in zip(times_iso, mean_sis, mean_sisclr, mean_sisStdv):
                    fh.write(f"{ti},{float(val):.6f},{float(val2):.6f},{float(val3):.6f}\n")

            print(f"Appended timeseries to {out_txt}")

    except Exception as e:
        print(f"Error visualizing NetCDF file: {e}")

if __name__ == "__main__":
    # Check for a list of files to process
    files_list_path = 'dataset/sis-files.txt'
    aggregated_out = 'dataset/sis_selected_timeseries.txt'

    # Remove existing aggregated output so we start fresh
    if os.path.exists(aggregated_out):
        try:
            os.remove(aggregated_out)
        except Exception:
            pass

    if os.path.exists(files_list_path):
        with open(files_list_path, 'r', encoding='utf-8') as fh:
            files = [line.strip() for line in fh if line.strip()]
        for f in files:
            if not os.path.isabs(f):
                f = os.path.join(os.getcwd(), "dataset", "surface-radiation", f)
            if os.path.exists(f):
                print(f"Processing {f}...")
                visualize_netcdf_data(f, out_txt=aggregated_out)
            else:
                print(f"Warning: listed file not found: {f}")
        print(f"All done. Aggregated results in {aggregated_out}")
    else:
        # fallback single file example
        netcdf_file_path = 'dataset/surface-radiation/SISdm20190101000040019AVPOS01GL.nc'
        if os.path.exists(netcdf_file_path):
            visualize_netcdf_data(netcdf_file_path, out_txt=aggregated_out)
        else:
            print(f"Error: NetCDF file not found at {netcdf_file_path} and no {files_list_path} present.")