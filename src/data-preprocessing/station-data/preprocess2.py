import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

def is_gzip_file(filepath):
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'
    
def display_all(df):
    plot_cols = df.columns
    plot_features = df[plot_cols]
    plot_features.index = df.index
    _ = plot_features.plot(subplots=True)
    plt.show()

    # plot_features = df[plot_cols][:960]
    # plot_features.index = df.index[:960]
    # _ = plot_features.plot(subplots=True)
    # plt.show()
    
def fill_with_lagged_data(df, periods_back=None):
    """
    Fill missing/zero values with data from specified periods back
    For 15-min data, one year = 365 * 24 * 4 = 35,040 periods
    """
    
    if periods_back is None:
        # Estimate periods for one year based on frequency
        freq = pd.infer_freq(df.index)
        if freq and '15T' in freq:
            periods_back = 365 * 24 * 4  # One year of 15-min data
        elif freq and 'H' in freq:
            periods_back = 365 * 24  # One year of hourly data
        else:
            periods_back = 365  # Default to daily
    
    filled_df = df.copy()
    
    for col in filled_df.columns:
        # Create a shifted version (one year ago)
        lagged_series = filled_df[col].shift(periods_back)
        
        # Create condition: where current value is 0 or NaN
        mask = (filled_df[col].isna()) | (filled_df[col] == 0)
        
        # Fill using the lagged data where condition is met
        filled_df[col] = filled_df[col].mask(mask, lagged_series)
    
    return filled_df

def display_lagged_comparison(df, col_name, periods_back=35040):
    """
    Display original data vs lagged data for comparison

    Args:
        df: DataFrame with the data
        col_name: Name of the column to compare
        periods_back: Number of periods to shift back (default: 1 year for 15-min data)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Original vs Lagged data
    ax1.plot(df.index, df[col_name], label='Original', alpha=0.7)
    ax1.plot(df.index, df[col_name].shift(periods_back), label=f'Lagged ({periods_back} periods)', alpha=0.7)
    ax1.set_title(f'Original vs Lagged Data Comparison - {col_name}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(col_name)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Recent data zoom (last 960 points)
    recent_data = df[col_name][-960:]
    recent_lagged = df[col_name].shift(periods_back)[-960:]
    ax2.plot(recent_data.index, recent_data, label='Original (Recent)', alpha=0.7)
    ax2.plot(recent_lagged.index, recent_lagged, label=f'Lagged ({periods_back} periods)', alpha=0.7)
    ax2.set_title(f'Recent Data Comparison - {col_name} (Last 960 points)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel(col_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot 3: Show where lagged data is used for filling
    filled_series = df[col_name].copy()
    mask = (df[col_name].isna()) | (df[col_name] == 0)
    filled_series = filled_series.mask(mask, df[col_name].shift(periods_back))

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[col_name], label='Original', alpha=0.5, color='red')
    plt.plot(df.index, filled_series, label='After Lagged Filling', alpha=0.7, color='blue')
    plt.scatter(df.index[mask], df[col_name].shift(periods_back)[mask],
               color='green', s=10, label='Lagged Values Used', alpha=0.6)
    plt.title(f'Lagged Data Filling Effect - {col_name}')
    plt.xlabel('Time')
    plt.ylabel(col_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['axes.grid'] = False

    csv_path = tf.keras.utils.get_file(origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/device-1.csv.tar.gz')

    if is_gzip_file(csv_path):
      # Read the data directly into a pandas DataFrame
      df = pd.read_csv(csv_path, compression='gzip')
    else:
      # Read the data without compression
      df = pd.read_csv(csv_path)

    row_count = df.shape[0]
    print(f"Total rows: {row_count}")

    # Rename the first column to 'new_name'
    df = df.rename(columns={df.columns[0]: 'datetime'})

    try:
        df['datetime'] = pd.to_datetime(df['datetime'])
        print("✅ แปลง Datetime สำเร็จด้วย pd.to_datetime() แบบอัตโนมัติ")
    except Exception as e:
        print(f"❌ การแปลงแบบอัตโนมัติล้มเหลว: {e}")
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
            print("✅ แปลง Datetime สำเร็จด้วย format '%Y-%m-%d %H:%M:%S'")
        except Exception as e2:
            print(f"❌ การแปลงด้วย format ล้มเหลว: {e2}")
            # ใช้วิธีสุดท้าย - แปลงแบบ errors='coerce'
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            print("⚠️  แปลง Datetime ด้วย errors='coerce' (ค่าที่แปลงไม่ได้จะเป็น NaT)")

    df = df.sort_values('datetime').reset_index(drop=True)

    # Convert 'timestamp' column to datetime
    df = df.set_index('datetime')

    print(df.index.year.unique())
    print(df.index.year.value_counts())

    filled_df = fill_with_lagged_data(df)

    # Display comparison between original and lagged data for the first column
    first_col = df.columns[0] if len(df.columns) > 0 else None
    if first_col:
        print(f"\nDisplaying lagged data comparison for column: {first_col}")
        display_lagged_comparison(df, first_col)