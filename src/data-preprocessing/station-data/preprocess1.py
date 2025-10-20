from itertools import count
import os
import numpy as np
import pandas as pd

def mean_positive_above_threshold(group_values):
    """
    Calculate mean of values > 0 only if more than 50% of values are > 0.
    Otherwise return NaN.
    """
    # Remove NaN values first
    valid_values = group_values.dropna()
    
    if len(valid_values) == 0:
        return np.nan
    
    # Count values greater than 0
    positive_count = (valid_values > 0).sum()
    total_count = len(valid_values)
    
    # Check if more than 50% are positive
    if positive_count > total_count * 0.5:
        # Return mean of positive values only
        positive_values = valid_values[valid_values > 0]
        return positive_values.mean() if len(positive_values) > 0 else np.nan
    else:
        # Less than or equal to 50% are positive, return NaN
        return np.nan

def preprocess_data(file, device1_out, device2_out, device3_out):
    """
    Preprocess the station data CSV file.
    
    Args:
        file (str): Path to the CSV file
    """
    try:
        df = pd.read_csv(file)
        df['datetime'] = df['datetime'].astype(str).str.strip("'\" ")
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['datetime'] = df['datetime'].dt.round('15min')

        device_1_df = df[df['device_id'] == 1].drop(columns=['device_id'])
        device_2_df = df[df['device_id'] == 2].drop(columns=['device_id'])
        device_3_df = df[df['device_id'] == 3].drop(columns=['device_id'])

        # Sort the DataFrame by 'Start Time'
        df1 = device_1_df.sort_values('datetime')
        df2 = device_2_df.sort_values('datetime')
        df3 = device_3_df.sort_values('datetime')

        # Define the start and end times
        start_time1 = df1['datetime'].min()
        end_time1 = df1['datetime'].max()
        start_time2 = df2['datetime'].min()
        end_time2 = df2['datetime'].max()
        start_time3 = df3['datetime'].min()
        end_time3 = df3['datetime'].max()

        print(f"device_1: {start_time1} - {end_time1}")
        print(f"device_2: {start_time2} - {end_time2}")
        print(f"device_3: {start_time3} - {end_time3}")

        # Create a complete time series with 15-minute intervals
        complete_time_series1 = pd.date_range(start=start_time1, end=end_time1, freq='15min')
        count = complete_time_series1.size
        print(f"Number of rows: {count}")
        
        complete_df1 = pd.DataFrame({'datetime': complete_time_series1})
        complete_time_series2 = pd.date_range(start=start_time2, end=end_time2, freq='15min')
        complete_df2 = pd.DataFrame({'datetime': complete_time_series2})
        complete_time_series3 = pd.date_range(start=start_time3, end=end_time3, freq='15min')
        complete_df3 = pd.DataFrame({'datetime': complete_time_series3})

        # Merge with complete time series
        merged_df1 = pd.merge(complete_df1, df1, on='datetime', how='left')
        merged_df2 = pd.merge(complete_df2, df2, on='datetime', how='left')
        merged_df3 = pd.merge(complete_df3, df3, on='datetime', how='left')

        # Group by datetime and calculate mean for all numeric columns
        df1_clean = merged_df1.pivot_table(
            index='datetime',
            values=[col for col in merged_df1.columns if col != 'datetime'],
            aggfunc=mean_positive_above_threshold
        ).reset_index()
        df2_clean = merged_df2.pivot_table(
            index='datetime',
            values=[col for col in merged_df2.columns if col != 'datetime'],
            aggfunc=mean_positive_above_threshold
        ).reset_index()
        df3_clean = merged_df3.pivot_table(
            index='datetime',
            values=[col for col in merged_df3.columns if col != 'datetime'],
            aggfunc=mean_positive_above_threshold
        ).reset_index()

        print(f"Original shape df1: {merged_df1.shape}")
        print(f"Original shape df2: {merged_df2.shape}")
        print(f"Original shape df3: {merged_df3.shape}")
        print(f"After deduplication df1: {df1_clean.shape}")
        print(f"After deduplication df2: {df2_clean.shape}")
        print(f"After deduplication df3: {df3_clean.shape}")

        # Append to output CSV files
        df1_clean.to_csv(device1_out, mode='a', index=False, header=not os.path.exists(device1_out))
        df2_clean.to_csv(device2_out, mode='a', index=False, header=not os.path.exists(device2_out))
        df3_clean.to_csv(device3_out, mode='a', index=False, header=not os.path.exists(device3_out))
    
    except Exception as e:
        print(f"Error preprocessing data in {file_path}: {e}")

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), 'dataset', 'pvdb', 'org', 'files.txt')
    device1_out = os.path.join(os.getcwd(), 'dataset', 'pvdb', 'org', 'device-1.csv')
    device2_out = os.path.join(os.getcwd(), 'dataset', 'pvdb', 'org', 'device-2.csv')
    device3_out = os.path.join(os.getcwd(), 'dataset', 'pvdb', 'org', 'device-3.csv')

    # Remove existing aggregated output so we start fresh
    if os.path.exists(device1_out):
        try:
            os.remove(device1_out)
        except Exception:
            pass

    if os.path.exists(device2_out):
        try:
            os.remove(device2_out)
        except Exception:
            pass

    if os.path.exists(device3_out):
        try:
            os.remove(device3_out)
        except Exception:
            pass

    line_count = 0
    try:
        with open(file_path, 'r') as file:
            files = [line.strip() for line in file if line.strip()]

            for line in files:
                line_count += 1
                # Each 'line' includes the newline character '\n' at the end
                # Use .strip() to remove leading/trailing whitespace, including '\n'
                filename = line.strip()

                if not os.path.isabs(filename):
                    filename = os.path.join(os.getcwd(), "dataset", "pvdb", "org", filename)

                print(f"Processing {filename}...")
                preprocess_data(filename, device1_out, device2_out, device3_out)

        print(f"\nSuccessfully read {line_count} lines.")

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")