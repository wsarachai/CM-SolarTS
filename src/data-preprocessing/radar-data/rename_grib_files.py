import os
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime

def get_time_range_from_grib(grib_path):
    """
    Extract time range from GRIB file
    
    Args:
        grib_path (str): Path to GRIB file
        
    Returns:
        tuple: (start_date, end_date) as datetime objects, or (None, None) if failed
    """
    try:
        with xr.open_dataset(grib_path, engine='cfgrib') as ds:
            if 'time' in ds.coords:
                time_values = ds['time'].values
                
                # Handle single or multiple time values
                if isinstance(time_values, np.ndarray):
                    if time_values.size == 0:
                        return None, None
                    start_time = np.min(time_values)
                    end_time = np.max(time_values)
                else:
                    start_time = time_values
                    end_time = time_values
                
                # Convert to datetime
                start_dt = np.datetime64(start_time).astype('datetime64[D]').astype(datetime)
                end_dt = np.datetime64(end_time).astype('datetime64[D]').astype(datetime)
                
                return start_dt, end_dt
            else:
                print(f"  WARNING: No 'time' coordinate found in {os.path.basename(grib_path)}")
                return None, None
                
    except Exception as e:
        print(f"  ERROR reading {os.path.basename(grib_path)}: {e}")
        return None, None

def format_date_range(start_dt, end_dt):
    """
    Format date range as YYYYMMDD_YYYYMMDD
    
    Args:
        start_dt (datetime): Start date
        end_dt (datetime): End date
        
    Returns:
        str: Formatted date range
    """
    start_str = start_dt.strftime('%Y%m%d')
    end_str = end_dt.strftime('%Y%m%d')
    
    if start_str == end_str:
        # Single day
        return start_str
    else:
        # Date range
        return f"{start_str}_{end_str}"

def rename_grib_files(input_dir, dry_run=True):
    """
    Rename all GRIB files in directory based on their time range
    
    Args:
        input_dir (str): Directory containing GRIB files
        dry_run (bool): If True, only show what would be renamed without actually renaming
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        return
    
    # Find all .grib files
    grib_files = sorted(input_path.glob('*.grib'))
    
    if not grib_files:
        print(f"No .grib files found in {input_dir}")
        return
    
    print(f"Found {len(grib_files)} GRIB files")
    print("=" * 80)
    
    rename_count = 0
    skip_count = 0
    error_count = 0
    
    for grib_file in grib_files:
        original_name = grib_file.name
        print(f"\nProcessing: {original_name}")
        
        # Get time range
        start_dt, end_dt = get_time_range_from_grib(str(grib_file))
        
        if start_dt is None or end_dt is None:
            print(f"  SKIP: Could not extract time range")
            error_count += 1
            continue
        
        # Format new name
        date_range = format_date_range(start_dt, end_dt)
        new_name = f"{date_range}.grib"
        
        print(f"  Time range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
        print(f"  New name: {new_name}")
        
        # Check if already has correct name
        if original_name == new_name:
            print(f"  SKIP: Already has correct name")
            skip_count += 1
            continue
        
        # Check if target name already exists
        new_path = grib_file.parent / new_name
        if new_path.exists() and new_path != grib_file:
            print(f"  WARNING: Target file already exists: {new_name}")
            print(f"  SKIP: To avoid overwriting")
            skip_count += 1
            continue
        
        # Rename file
        if dry_run:
            print(f"  DRY RUN: Would rename to {new_name}")
            rename_count += 1
        else:
            try:
                grib_file.rename(new_path)
                print(f"  SUCCESS: Renamed to {new_name}")
                rename_count += 1
                
                # Also rename associated .idx file if exists
                idx_file = grib_file.with_suffix(grib_file.suffix + '.5b7b6.idx')
                if idx_file.exists():
                    new_idx_path = new_path.with_suffix(new_path.suffix + '.5b7b6.idx')
                    idx_file.rename(new_idx_path)
                    print(f"  Also renamed index file")
                    
            except Exception as e:
                print(f"  ERROR: Failed to rename: {e}")
                error_count += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(grib_files)}")
    print(f"Renamed: {rename_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    
    if dry_run and rename_count > 0:
        print("\nThis was a DRY RUN. Run with dry_run=False to actually rename files.")

if __name__ == "__main__":
    grib_directory = 'D:\\_datasets\\weather-dataset\\ERA5-Land-data\\land'
    
    # First do a dry run to see what would be renamed
    print("=" * 80)
    print("GRIB FILE RENAMING TOOL")
    print("=" * 80)
    print(f"Directory: {grib_directory}")
    print("\nDRY RUN MODE - No files will be renamed")
    print("=" * 80)
    
    rename_grib_files(grib_directory, dry_run=False)
    
    # Ask user to confirm
    print("\n" + "=" * 80)
    response = input("\nDo you want to proceed with renaming? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\n" + "=" * 80)
        print("RENAMING FILES...")
        print("=" * 80)
        rename_grib_files(grib_directory, dry_run=False)
    else:
        print("\nRenaming cancelled.")
