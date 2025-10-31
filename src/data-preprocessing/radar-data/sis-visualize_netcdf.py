#
# This file contains AVHRR-based Interim Climate Data Records (ICDR)
# produced by the Satellite Application Facility on Climate Monitoring (CM SAF).
# Conventions: CF-1.6,ACDD-1.3
#
# The amount of solar radiation (shortwave) reaching the Earth's surface from the sun.
#
# File format: NETCDF4_CLASSIC
# references: https://wui.cmsaf.eu/safira/action/viewICDRDetails?acronym=CLARA_AVHRR_V002_ICDR
# project: Satellite Application Facility on Climate Monitoring (CM SAF)
# === Variables ===
# lon         Longitude [degrees_east]
# lat         Latitude [degrees_north]
# time        units: [days since 1970-01-01 00:00:00]
# SIS         Surface Incoming Shortwave Radiation [W m**-2]
#             standard_name: surface_downwelling_shortwave_flux
#             long_name: Surface Downwelling Shortwave Radiation
#             units: W m-2
# LowAcc_SIS  surface_downwelling_shortwave_flux
#             long_name: Surface Downwelling Shortwave Radiation, not compliant with CM SAF target accuracy
#             units: W m-2
# SISCLS      surface_downwelling_shortwave_flux_assuming_clear_sky
#             long_name: Surface Downwelling Shortwave Radiation assuming clear-sky conditions
#             units: W m-2
# SIS_stdv    surface_downwelling_shortwave_flux_standard_deviation
#             long_name: Standard Deviation of Surface Downwelling Shortwave Radiation
#             units: W m-2
#
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os

# common missing-value attributes for diagnostics
missing_value_attrs = ['_FillValue', 'missing_value', 'fill_value']

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
            sisData = ds.variables['SIS']
            sisClsData = ds.variables.get('SISCLS', None)
            sisStdvData = ds.variables.get('SIS_stdv', None)

            print(f"sis variable shape: {sisData.shape if sisData is not None else 'N/A'}")
            print(f"sis cls variable shape: {sisClsData.shape if sisClsData is not None else 'N/A'}")
            print(f"sis stdv variable shape: {sisStdvData.shape if sisStdvData is not None else 'N/A'}")

            area = [float(latitude.min()), float(longitude.min()), float(latitude.max()), float(longitude.max())]
            print(f"Area of data: {area}")

            # Desired bounding box (lat_min, lon_min, lat_max, lon_max)
            lat_min = 18.000
            lat_max = 20.000
            lon_min = 98.000
            lon_max = 100.000

            # Find indices within the bounding box
            lat_inds = np.where((latitude >= lat_min) & (latitude <= lat_max))[0]
            lon_inds = np.where((longitude >= lon_min) & (longitude <= lon_max))[0]

            print(f"Found lat indices: {lat_inds}, lon indices: {lon_inds}")

            if lat_inds.size == 0 or lon_inds.size == 0:
                print("No grid points found inside the requested bounding box. Check coordinates or widen the box.")
                return

            # Convert sis data to a float numpy array and slice to (time, lat, lon)
            # sisData may be a masked array or contain non-numeric fill values (e.g. '?').
            try:
                sisRaw = sisData[:]
            except Exception as e:
                print(f"Warning reading sis variable: {e}")
                # fallback: read into object array
                sisRaw = np.array([sisData[i] for i in range(len(sisData))], dtype=object)
            # Print any common missing-value attributes for diagnostics
            for attr in missing_value_attrs:
                if hasattr(sisData, 'getncattr'):
                    try:
                        val = sisData.getncattr(attr)
                        print(f"Attribute {attr} = {val}")
                    except Exception:
                        pass

            # If it's a masked array, fill masked entries with NaN
            try:
                if isinstance(sisRaw, ma.MaskedArray):
                    sisRaw_filled = ma.filled(sisRaw, np.nan)
                else:
                    sisRaw_filled = sisRaw
            except Exception:
                sisRaw_filled = sisRaw

            try:
                if isinstance(sisClsData, ma.MaskedArray):
                    sisClsRaw_filled = ma.filled(sisClsData, np.nan)
                else:
                    sisClsRaw_filled = sisClsData
            except Exception:
                sisClsRaw_filled = sisClsData

            try:
                if isinstance(sisStdvData, ma.MaskedArray):
                    sisStdvRaw_filled = ma.filled(sisStdvData, np.nan)
                else:
                    sisStdvRaw_filled = sisStdvData
            except Exception:
                sisStdvRaw_filled = sisStdvData

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

            # Check for non-numeric fill values and replace with NaN
            fv = None
            for attr in missing_value_attrs:
                try:
                    fv = sisData.getncattr(attr)
                    if isinstance(fv, str):
                        print(f"Non-numeric fill value detected: {fv}; will replace occurrences with NaN")
                        # Replace string fill values in raw with np.nan
                        raw_array = np.asarray(sisRaw_filled, dtype=object)
                        raw_array[raw_array == fv] = np.nan
                        sisRaw_filled = raw_array
                        break
                except Exception:
                    fv = None
            for attr in missing_value_attrs:
                try:
                    fv = sisClsData.getncattr(attr)
                    if isinstance(fv, str):
                        print(f"Non-numeric fill value detected: {fv}; will replace occurrences with NaN")
                        # Replace string fill values in raw with np.nan
                        raw_array = np.asarray(sisClsRaw_filled, dtype=object)
                        raw_array[raw_array == fv] = np.nan
                        sisClsRaw_filled = raw_array
                        break
                except Exception:
                    fv = None
            for attr in missing_value_attrs:
                try:
                    fv = sisClsData.getncattr(attr)
                    if isinstance(fv, str):
                        print(f"Non-numeric fill value detected: {fv}; will replace occurrences with NaN")
                        # Replace string fill values in raw with np.nan
                        raw_array = np.asarray(sisStdvRaw_filled, dtype=object)
                        raw_array[raw_array == fv] = np.nan
                        sisStdvRaw_filled = raw_array
                        break
                except Exception:
                    fv = None

            sis_arr = robust_to_float(sisRaw_filled) if sisRaw_filled is not None else None
            sis_cls_arr = robust_to_float(sisClsRaw_filled) if sisClsRaw_filled is not None else None
            sis_stdv_arr = robust_to_float(sisStdvRaw_filled) if sisStdvRaw_filled is not None else None
            print(f"Original sis array shape: {sis_arr.shape}")

            # Try to detect order: if sis_arr.shape[0] == len(time) assume (time, lat, lon)
            if sis_arr.ndim == 3:
                if sis_arr.shape[0] == len(time):
                    sis_time_lat_lon = sis_arr[:, lat_inds[:, None], lon_inds] if sis_arr is not None else None
                    sis_cls_time_lat_lon = sis_cls_arr[:, lat_inds[:, None], lon_inds] if sis_cls_arr is not None else None
                    sis_stdv_time_lat_lon = sis_stdv_arr[:, lat_inds[:, None], lon_inds] if sis_stdv_arr is not None else None
                elif sis_arr.shape[-1] == len(time):
                    # (lat, lon, time) -> transpose to (time, lat, lon)
                    sis_time_lat_lon = np.transpose(sis_arr, (2, 0, 1))[:, lat_inds[:, None], lon_inds] if sis_arr is not None else None
                    sis_cls_time_lat_lon = np.transpose(sis_cls_arr, (2, 0, 1))[:, lat_inds[:, None], lon_inds] if sis_cls_arr is not None else None
                    sis_stdv_time_lat_lon = np.transpose(sis_stdv_arr, (2, 0, 1))[:, lat_inds[:, None], lon_inds] if sis_stdv_arr is not None else None
                else:
                    # fallback: try to interpret middle axis as time
                    sis_time_lat_lon = sis_arr.transpose(1, 2, 0)[:, lat_inds[:, None], lon_inds] if sis_arr is not None else None
                    sis_cls_time_lat_lon = sis_cls_arr.transpose(1, 2, 0)[:, lat_inds[:, None], lon_inds] if sis_cls_arr is not None else None
                    sis_stdv_time_lat_lon = sis_stdv_arr.transpose(1, 2, 0)[:, lat_inds[:, None], lon_inds] if sis_stdv_arr is not None else None
            elif sis_arr.ndim == 2:
                # Maybe (lat, lon) single time
                sis_time_lat_lon = sis_arr[lat_inds[:, None], lon_inds][None, ...] if sis_arr is not None else None
                sis_cls_time_lat_lon = sis_cls_arr[lat_inds[:, None], lon_inds][None, ...] if sis_cls_arr is not None else None
                sis_stdv_time_lat_lon = sis_stdv_arr[lat_inds[:, None], lon_inds][None, ...] if sis_stdv_arr is not None else None
            else:
                raise ValueError(f"Unexpected sis array dimensions: {sis_arr.ndim}")

            # Ensure result is float and shaped (time, lat, lon)
            sis_time_lat_lon = sis_time_lat_lon.astype(float)
            print(f"Filtered sis shape (time, lat, lon): {sis_time_lat_lon.shape}")

            # Quick visualization: plot mean over the small spatial box as a time series
            mean_sis = np.nanmean(sis_time_lat_lon, axis=(1, 2)) if sis_time_lat_lon is not None else None
            mean_sis_cls = np.nanmean(sis_cls_time_lat_lon, axis=(1, 2)) if sis_cls_time_lat_lon is not None else None
            mean_sis_stdv = np.nanmean(sis_stdv_time_lat_lon, axis=(1, 2)) if sis_stdv_time_lat_lon is not None else None
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
            header = 'time,sis,sis_cls,sis_stdv\n'
            write_header = not os.path.exists(out_txt)
            with open(out_txt, 'a', encoding='utf-8') as fh:
                if write_header:
                    fh.write(header)
                for ti, val, cls_val, stdv_val in zip(times_iso, mean_sis, mean_sis_cls, mean_sis_stdv):
                    if np.isnan(val):
                        fh.write(f"{ti},nan,{float(cls_val):.6f},{float(stdv_val):.6f}\n")
                    else:
                        fh.write(f"{ti},{float(val):.6f},{float(cls_val):.6f},{float(stdv_val):.6f}\n")

            print(f"Appended timeseries to {out_txt}")

    except Exception as e:
        print(f"Error visualizing NetCDF file: {e}")

if __name__ == "__main__":
    # Check for a list of files to process
    files_list_path = 'dataset/sis-files.txt'
    aggregated_out = 'dataset/sis_selected_timeseries.csv'

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
                f = os.path.join(os.getcwd(), "dataset", "sis-data", f)
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