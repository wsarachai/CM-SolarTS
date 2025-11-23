#!/usr/bin/env python3
"""Utility to examine GRIB file contents.

Features:
  - Detect and use `pygrib` if available for low-level enumeration.
  - Fallback to `xarray` + `cfgrib` engine.
  - Handles multi-message GRIB files by iterating over distinct shortNames.
  - Prints dimensions, variables, coordinates and basic metadata.
  - Graceful error handling: continues even if one variable group fails.

Usage:
  python examine_grib.py /path/to/file.grib
  (If no path given, uses hard-coded sample path.)
"""

import os
import sys
import argparse

COMMON_VARS = ['t2m', 'fal', 'slhf', 'ssr', 'str', 'sshf', 'ssrd', 'strd', 'u10', 'v10', 'sp', 'tp']

def _print_header(title):
    print("\n" + title)
    print("=" * len(title))

def examine_with_pygrib(grib_path):
    import pygrib
    _print_header("pygrib summary")
    msgs = pygrib.open(grib_path)
    short_names = set()
    total = 0
    for m in msgs:
        total += 1
        short_names.add(getattr(m, 'shortName', 'unknown'))
    print(f"Total GRIB messages: {total}")
    print(f"Distinct shortNames: {sorted(short_names)}")
    msgs.seek(0)
    # Show first 3 messages details
    for i, m in enumerate(msgs):
        if i >= 3:
            break
        print(f"Message {i+1}: shortName={getattr(m,'shortName','?')} typeOfLevel={getattr(m,'typeOfLevel','?')} level={getattr(m,'level','?')}")
    msgs.close()

def examine_with_cfgrib(grib_path):
    import xarray as xr
    _print_header("cfgrib/xarray summary")
    try:
        # Basic open attempt. Some ERA5 / ERA5-Land files contain multiple parameter groups.
        ds = xr.open_dataset(grib_path, engine='cfgrib')
        print("Opened combined dataset successfully.")
        _describe_dataset(ds)
    except Exception as e:
        print(f"Initial open failed: {e}")
        print("Attempting per-shortName opening (multi-message handling)...")
        # Fallback strategy: try opening by filtering common variables.
        for short_name in COMMON_VARS:
            try:
                backend_kwargs = {"filter_by_keys": {"shortName": short_name}, "indexpath": ""}
                ds_var = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs=backend_kwargs)
                print(f"\nOpened subset for shortName='{short_name}'")
                _describe_dataset(ds_var, limit_vars=[short_name])
            except Exception:
                # Silently skip missing variables.
                pass

def _describe_dataset(ds, limit_vars=None):
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords.keys())}")
    vars_list = list(ds.data_vars.keys())
    if limit_vars:
        vars_list = [v for v in vars_list if v in limit_vars]
    print(f"Variables: {vars_list}")
    # Show details for up to first 5 variables
    for var_name in vars_list[:5]:
        da = ds[var_name]
        print(f"  Variable '{var_name}': shape={da.shape} dims={da.dims}")
        for attr in ['units', 'long_name', 'shortName', 'typeOfLevel']:
            if attr in da.attrs:
                print(f"    {attr}: {da.attrs[attr]}")

def examine_grib_file(grib_path, backend_preference=None):
    if not os.path.isfile(grib_path):
        print(f"File does not exist: {grib_path}")
        return
    print(f"Examining GRIB file: {grib_path}")
    print(f"File size: {os.path.getsize(grib_path) / (1024*1024):.2f} MB")

    used_pygrib = False
    # If user forces pygrib
    if backend_preference == 'pygrib':
        try:
            import pygrib  # noqa: F401
            examine_with_pygrib(grib_path)
            used_pygrib = True
        except Exception as e:
            print(f"Forced pygrib failed ({e}).")
    elif backend_preference == 'xarray':
        # Force xarray only
        try:
            examine_with_cfgrib(grib_path)
        except Exception as e:
            print(f"Forced xarray failed: {e}")
        return
    else:
        # Auto mode: try pygrib then xarray
        try:
            import pygrib  # noqa: F401
            examine_with_pygrib(grib_path)
            used_pygrib = True
        except Exception as e:
            print(f"pygrib unavailable or failed ({e}); will try cfgrib/xarray.")

    if backend_preference != 'pygrib' and not used_pygrib:
        try:
            examine_with_cfgrib(grib_path)
        except Exception as e:
            print(f"cfgrib/xarray examination failed: {e}")
            if not used_pygrib:
                print("No GRIB backend succeeded.")

def main():
    examine_grib_file("/Volumes/Seagate/_datasets/weather-dataset/ERA5-Land-data/land/3b88b9c39cc18d6d98b7650ff27f27e8.grib", 'xarray')

if __name__ == "__main__":
    main()