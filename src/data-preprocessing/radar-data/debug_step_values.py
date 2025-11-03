#!/usr/bin/env python3
"""
Debug script to understand step values in GRIB files
"""

import xarray as xr
import numpy as np

def debug_step_values(grib_path):
    print(f"Debugging step values in: {grib_path}")
    
    try:
        with xr.open_dataset(grib_path, engine='cfgrib') as ds:
            print(f"Dimensions: {list(ds.dims.keys())}")
            
            # Check step values for first available variable
            for var in ['t2m', 'fal', 'slhf']:
                if var in ds.data_vars:
                    var_data = ds[var]
                    if var_data.ndim == 4:
                        step_vals = var_data['step'].values
                        print(f"\nVariable: {var}")
                        print(f"Step values: {step_vals}")
                        print(f"Step value types: {[type(s) for s in step_vals]}")
                        print(f"Step shape: {step_vals.shape}")
                        
                        # Test parsing each step value
                        for i, step in enumerate(step_vals):
                            try:
                                if isinstance(step, np.timedelta64):
                                    hours = int(step.astype('timedelta64[h]'))
                                    print(f"  Step {i}: {step} -> {hours} hours (timedelta)")
                                else:
                                    hours = int(step)
                                    print(f"  Step {i}: {step} -> {hours} hours (int)")
                            except Exception as e:
                                print(f"  Step {i}: {step} -> ERROR: {e}")
                        break
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with first GRIB file
    grib_file = "dataset/ERA5-land-hourly/3b88b9c39cc18d6d98b7650ff27f27e8.grib"
    debug_step_values(grib_file)