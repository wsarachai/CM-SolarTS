import os
import re
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

def is_gzip_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'
    
csv_main = tf.keras.utils.get_file(
    origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/data_1_filled.csv.tar.gz')
    #origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/utci_selected_timeseries_filled_15min.csv.zip')
    #origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/era5-hourly_timeseries_filled_15min.csv.zip')
    #origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/era5-land_timeseries_filled_15min.csv.zip')
print(csv_main)

csv_ext = tf.keras.utils.get_file(
    #origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/data_1_filled.csv.tar.gz')
    origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/utci_selected_timeseries_filled_15min.csv.zip')
    #origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/era5-hourly_timeseries_filled_15min.csv.zip')
    #origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/era5-land_timeseries_filled_15min.csv.zip')
print(csv_ext)

date_col = 'datetime' 
if is_gzip_file(csv_main):
  # Read the data directly into a pandas DataFrame
  df_main = pd.read_csv(csv_main, compression='gzip')
else:
  # Read the data without compression
  df_main = pd.read_csv(csv_main)

if is_gzip_file(csv_ext):
  # Read the data directly into a pandas DataFrame
  df_ext = pd.read_csv(csv_ext, compression='gzip')
else:
  # Read the data without compression
  df_ext = pd.read_csv(csv_ext)

df_main = df_main.rename(columns={df_main.columns[0]: date_col})
df_ext = df_ext.rename(columns={df_ext.columns[0]: date_col})

# Ensure timestamp column is datetime type
if df_main[date_col].dtype != 'datetime64[ns]':
    df_main[date_col] = pd.to_datetime(df_main[date_col])
if df_ext[date_col].dtype != 'datetime64[ns]':
    df_ext[date_col] = pd.to_datetime(df_ext[date_col])

print(f"Main total rows: {df_main.shape[0]}")
print(f"Ext total rows: {df_ext.shape[0]}")

print(df_main[date_col].dt.year.unique())
print(df_main[date_col].dt.year.value_counts())
print(df_main.shape)

# Sort by time if it exists
df_main.index = df_main[date_col]
df_main.pop(date_col)
df_main = df_main.sort_index()
print(df_main.shape)

# Upsample from hourly to 15-minute resolution
# Create 15-minute index spanning the full range
start = df_main.index.min()
end = df_main.index.max()
freq_15min = pd.date_range(start=start, end=end, freq='15min')
print(freq_15min)

# Diagnose and fix index issues
print(f"Index dtype: {df_main.index.dtype}")
print(f"Has NaT: {df_main.index.isna().sum()}")
print(f"Has duplicates: {df_main.index.duplicated().sum()}")
print(f"Is monotonic: {df_main.index.is_monotonic_increasing}")

# Prepare ext: set index and sort
df_ext.index = df_ext[date_col]
df_ext.pop(date_col)
df_ext = df_ext.sort_index()

# Join only difference: add ext columns that don't exist in main, aligned by time
unique_ext_cols = df_ext.columns.difference(df_main.columns)
print(f"Unique ext columns to add: {list(unique_ext_cols)}")

df_merged = df_main.join(df_ext[unique_ext_cols], how='left')

print(f"Merged shape: {df_merged.shape}")

# Optional: write output next to main file
out_path = Path(csv_main).with_name(Path(csv_main).stem + "_merged.csv")
df_merged.to_csv(out_path)
print(f"Saved merged dataset to: {out_path}")
