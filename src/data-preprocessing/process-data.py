import os
import datetime
import pytz

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from scipy.signal import find_peaks

def is_gzip_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

csv_path = tf.keras.utils.get_file(
    origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/merge_15min_filled.csv.zip')

if is_gzip_file(csv_path):
  # Read the data directly into a pandas DataFrame
  df = pd.read_csv(csv_path, compression='gzip')
else:
  # Read the data without compression
  df = pd.read_csv(csv_path)

row_count = df.shape[0]
print(f"Total rows: {row_count}")

print("\n=== Adding Derived Features ===\n")

# 1. Wind Speed and Direction
if 'u' in df.columns and 'v' in df.columns:
    u_col, v_col = 'u', 'v'
elif 'u10' in df.columns and 'v10' in df.columns:
    u_col, v_col = 'u10', 'v10'
else:
    u_col, v_col = None, None

if u_col and v_col:
    df['wind_speed'] = np.sqrt(df[u_col]**2 + df[v_col]**2)
    df['wind_direction'] = (270 - np.arctan2(df[v_col], df[u_col]) * 180 / np.pi) % 360
    print(f"✓ Added 'wind_speed' and 'wind_direction' from '{u_col}' and '{v_col}'")
else:
    print("✗ No wind components found (u/v or u10/v10)")

# 2. Net Radiation (short-wave + long-wave)
if 'ssr' in df.columns and 'str' in df.columns:
    df['net_radiation'] = df['ssr'] + df['str']
    print("✓ Added 'net_radiation' (ssr + str)")

# 3. Total Downward Radiation
if 'ssrd' in df.columns and 'strd' in df.columns:
    df['total_downward_radiation'] = df['ssrd'] + df['strd']
    print("✓ Added 'total_downward_radiation' (ssrd + strd)")

# 4. Net Heat Flux
if 'slhf' in df.columns and 'sshf' in df.columns:
    df['net_heat_flux'] = df['slhf'] + df['sshf']
    print("✓ Added 'net_heat_flux' (slhf + sshf)")

# 5. Dewpoint Temperature (from temperature and relative humidity)
if 't' in df.columns and 'r' in df.columns:
    a, b = 17.27, 237.7
    T_celsius = df['t'] - 273.15
    alpha = ((a * T_celsius) / (b + T_celsius)) + np.log(df['r'] / 100)
    df['dewpoint'] = (b * alpha) / (a - alpha) + 273.15  # Kelvin
    print("✓ Added 'dewpoint' from 't' and 'r'")
elif 't2m' in df.columns and 'r' in df.columns:
    a, b = 17.27, 237.7
    T_celsius = df['t2m'] - 273.15
    alpha = ((a * T_celsius) / (b + T_celsius)) + np.log(df['r'] / 100)
    df['dewpoint'] = (b * alpha) / (a - alpha) + 273.15  # Kelvin
    print("✓ Added 'dewpoint' from 't2m' and 'r'")

# 6. Vapor Pressure Deficit (VPD)
if 't' in df.columns and 'r' in df.columns:
    T_celsius = df['t'] - 273.15
    es = 6.112 * np.exp((17.67 * T_celsius) / (T_celsius + 243.5))  # hPa
    ea = es * (df['r'] / 100)
    df['vpd'] = es - ea
    print("✓ Added 'vpd' (vapor pressure deficit)")
elif 't2m' in df.columns and 'r' in df.columns:
    T_celsius = df['t2m'] - 273.15
    es = 6.112 * np.exp((17.67 * T_celsius) / (T_celsius + 243.5))  # hPa
    ea = es * (df['r'] / 100)
    df['vpd'] = es - ea
    print("✓ Added 'vpd' (vapor pressure deficit)")

# 7. Air Density
if 'sp' in df.columns and 't' in df.columns:
    R_specific = 287.05  # J/(kg·K) for dry air
    df['air_density'] = df['sp'] / (R_specific * df['t'])  # kg/m³
    print("✓ Added 'air_density' from 'sp' and 't'")
elif 'sp' in df.columns and 't2m' in df.columns:
    R_specific = 287.05
    df['air_density'] = df['sp'] / (R_specific * df['t2m'])
    print("✓ Added 'air_density' from 'sp' and 't2m'")

# 8. Precipitation Rate (convert meters to mm)
if 'tp' in df.columns:
    df['precip_rate_mm'] = df['tp'] * 1000  # mm per time step
    print("✓ Added 'precip_rate_mm' from 'tp'")

# 9. Temperature in Celsius
if 't2m' in df.columns:
    df['t2m_celsius'] = df['t2m'] - 273.15
    print("✓ Added 't2m_celsius'")
if 't' in df.columns:
    df['t_celsius'] = df['t'] - 273.15
    print("✓ Added 't_celsius'")

# 10. Heating/Cooling Degree Days (base 18°C)
if 't2m' in df.columns:
    T_base_K = 291.15  # 18°C in Kelvin
    df['heating_degree'] = np.maximum(T_base_K - df['t2m'], 0)
    df['cooling_degree'] = np.maximum(df['t2m'] - T_base_K, 0)
    print("✓ Added 'heating_degree' and 'cooling_degree'")

print(f"\nFinal shape: {df.shape}")
print(f"New columns added: {df.shape[1] - row_count}")

