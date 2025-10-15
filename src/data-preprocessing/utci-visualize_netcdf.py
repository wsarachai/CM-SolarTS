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
            # Try to disable automatic masking/scaling which can error if _FillValue is non-numeric
            try:
                nc.set_auto_mask(False)
            except Exception:
                pass
            try:
                nc.set_auto_maskandscale(False)
            except Exception:
                pass
            try:
                nc.set_auto_scale(False)
            except Exception:
                pass
            # Extract the data
            time = ds.variables['time'][:]
            longitude = ds.variables['lon'][:]
            latitude = ds.variables['lat'][:]
            utciData = ds.variables['utci']  # Universal Thermal Climate Index

            print(f"utci variable shape: {utciData.shape}")

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

            # Convert utciData to a float numpy array and slice to (time, lat, lon)
            # utciData may be a masked array or contain non-numeric fill values (e.g. '?').
            try:
                raw = utciData[:]
            except Exception as e:
                print(f"Warning reading utci variable: {e}")
                # fallback: read into object array
                raw = np.array([utciData[i] for i in range(len(utciData))], dtype=object)
            # Print any common missing-value attributes for diagnostics
            for attr in ['_FillValue', 'missing_value', 'fill_value']:
                if hasattr(utciData, 'getncattr'):
                    try:
                        val = utciData.getncattr(attr)
                        print(f"Attribute {attr} = {val}")
                    except Exception:
                        pass

            # If it's a masked array, fill masked entries with NaN
            try:
                import numpy.ma as ma
                if isinstance(raw, ma.MaskedArray):
                    raw_filled = ma.filled(raw, np.nan)
                else:
                    raw_filled = raw
            except Exception:
                raw_filled = raw

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

            # If utciData has a known fill value attribute and it's non-numeric (e.g. '?'), replace occurrences
            fv = None
            for attr in ['_FillValue', 'missing_value', 'fill_value']:
                try:
                    fv = utciData.getncattr(attr)
                    if isinstance(fv, str):
                        print(f"Non-numeric fill value detected: {fv}; will replace occurrences with NaN")
                        # Replace string fill values in raw with np.nan
                        raw_array = np.asarray(raw, dtype=object)
                        raw_array[raw_array == fv] = np.nan
                        raw = raw_array
                        break
                except Exception:
                    fv = None

            utci_arr = robust_to_float(raw_filled)
            print(f"Original utci array shape: {utci_arr.shape}")

            # Try to detect order: if utci_arr.shape[0] == len(time) assume (time, lat, lon)
            if utci_arr.ndim == 3:
                if utci_arr.shape[0] == len(time):
                    utci_time_lat_lon = utci_arr[:, lat_inds[:, None], lon_inds]
                elif utci_arr.shape[-1] == len(time):
                    # (lat, lon, time) -> transpose to (time, lat, lon)
                    utci_time_lat_lon = np.transpose(utci_arr, (2, 0, 1))[:, lat_inds[:, None], lon_inds]
                else:
                    # fallback: try to interpret middle axis as time
                    utci_time_lat_lon = utci_arr.transpose(1, 2, 0)[:, lat_inds[:, None], lon_inds]
            elif utci_arr.ndim == 2:
                # Maybe (lat, lon) single time
                utci_time_lat_lon = utci_arr[lat_inds[:, None], lon_inds][None, ...]
            else:
                raise ValueError(f"Unexpected utci array dimensions: {utci_arr.ndim}")

            # Ensure result is float and shaped (time, lat, lon)
            utci_time_lat_lon = utci_time_lat_lon.astype(float)
            print(f"Filtered utci shape (time, lat, lon): {utci_time_lat_lon.shape}")

            # Quick visualization: plot mean over the small spatial box as a time series
            mean_ts = np.nanmean(utci_time_lat_lon, axis=(1, 2))
            print(f"Mean timeseries shape: {mean_ts.shape}")

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
            header = 'time,utci_mean\n'
            write_header = not os.path.exists(out_txt)
            with open(out_txt, 'a', encoding='utf-8') as fh:
                if write_header:
                    fh.write(header)
                for ti, val in zip(times_iso, mean_ts):
                    if np.isnan(val):
                        fh.write(f"{ti},nan\n")
                    else:
                        fh.write(f"{ti},{float(val):.6f}\n")

            print(f"Appended timeseries to {out_txt}")

    except Exception as e:
        print(f"Error visualizing NetCDF file: {e}")

if __name__ == "__main__":
    # Check for a list of files to process
    files_list_path = 'dataset/files.txt'
    aggregated_out = 'dataset/utci_selected_timeseries.txt'

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
                f = os.path.join(os.getcwd(), "dataset", "utci-daily", f)
            if os.path.exists(f):
                print(f"Processing {f}...")
                visualize_netcdf_data(f, out_txt=aggregated_out)
            else:
                print(f"Warning: listed file not found: {f}")
        print(f"All done. Aggregated results in {aggregated_out}")
    else:
        # fallback single file example
        netcdf_file_path = 'dataset/utci-daily/ECMWF_utci_20250102_v1.1_con.nc'
        if os.path.exists(netcdf_file_path):
            visualize_netcdf_data(netcdf_file_path, out_txt=aggregated_out)
        else:
            print(f"Error: NetCDF file not found at {netcdf_file_path} and no {files_list_path} present.")