import pandas as pd
import os

# *** File paths ***
file1 = 'C:\\Users\\ASUS\\.keras\\datasets\\merge_15min_filled.csv.tar.gz'
file2 = 'C:\\Users\\ASUS\\.keras\\datasets\\solar_angles_2021_2025_15min_chiangmai.csv.tar.gz'

# *** Read CSV files ***
print("Reading file1...")
df1 = pd.read_csv(file1, compression='tar')
print(f"File1 shape: {df1.shape}")
print(f"File1 columns: {list(df1.columns)}")
print(f"File1 head:\n{df1.head()}\n")

print("Reading file2...")
df2 = pd.read_csv(file2, compression='tar')
print(f"File2 shape: {df2.shape}")
print(f"File2 columns: {list(df2.columns)}")
print(f"File2 head:\n{df2.head()}\n")

# *** Identify time column ***
# หาชื่อ column ที่เป็นเวลา (อาจเป็น 'datetime', 'time', 'timestamp' เป็นต้น)
time_col_df1 = None
time_col_df2 = None

for col in df1.columns:
    if col.lower() in ['datetime', 'time', 'timestamp', 'date']:
        time_col_df1 = col
        break

for col in df2.columns:
    if col.lower() in ['datetime', 'time', 'timestamp', 'date']:
        time_col_df2 = col
        break

if time_col_df1 is None:
    print("ERROR: ไม่พบ time column ใน file1")
    print("โปรดเลือกชื่อ column เองหรือตรวจสอบชื่อ column")
    exit(1)

if time_col_df2 is None:
    print("ERROR: ไม่พบ time column ใน file2")
    print("โปรดเลือกชื่อ column เองหรือตรวจสอบชื่อ column")
    exit(1)

print(f"Time column in file1: {time_col_df1}")
print(f"Time column in file2: {time_col_df2}\n")

# *** Convert time columns to datetime ***
df1[time_col_df1] = pd.to_datetime(df1[time_col_df1])
df2[time_col_df2] = pd.to_datetime(df2[time_col_df2])

# *** Remove timezone to make them compatible ***
# หากมี timezone ให้เอาออก (naive datetime)
if df1[time_col_df1].dt.tz is not None:
    df1[time_col_df1] = df1[time_col_df1].dt.tz_localize(None)
    print(f"Removed timezone from file1")

if df2[time_col_df2].dt.tz is not None:
    df2[time_col_df2] = df2[time_col_df2].dt.tz_localize(None)
    print(f"Removed timezone from file2")

# *** Sort by time ***
df1 = df1.sort_values(by=time_col_df1).reset_index(drop=True)
df2 = df2.sort_values(by=time_col_df2).reset_index(drop=True)

# *** Merge โดยใช้ left join (คงจำนวน row ตาม file1) ***
print(f"Merging... (left join based on {time_col_df1})")
merged = pd.merge(
    df1,
    df2,
    left_on=time_col_df1,
    right_on=time_col_df2,
    how='left'
)

print(f"Merged shape: {merged.shape}")
print(f"Merged head:\n{merged.head()}\n")

# *** Check for duplicate time columns ***
# ถ้ามีชื่อ column เหมือนกันใน file1 และ file2 pandas จะเพิ่ม _x และ _y
print(f"Merged columns: {list(merged.columns)}\n")

# *** Remove redundant time column from file2 (ถ้ามี _x, _y) ***
# ลบเฉพาะ column จาก file2 ที่ซ้ำ และเปลี่ยนชื่อ _x กลับเป็นชื่อเดิม
if f"{time_col_df1}_x" in merged.columns:
    # มี duplicate time column
    merged = merged.rename(columns={f"{time_col_df1}_x": time_col_df1})
    merged = merged.drop(columns=[f"{time_col_df1}_y"])
    print(f"Renamed {time_col_df1}_x to {time_col_df1} and dropped {time_col_df1}_y")
elif time_col_df1 != time_col_df2 and f"{time_col_df2}" in merged.columns:
    # ชื่อ time column ต่างกัน ลบ column จาก file2
    merged = merged.drop(columns=[time_col_df2])
    print(f"Dropped duplicate time column: {time_col_df2}")

# *** Save merged data ***
output_file = 'merged_data.csv'
merged.to_csv(output_file, index=False)
print(f"\nMerged data saved to: {output_file}")
print(f"Final shape: {merged.shape}")
print(f"\nMissing values per column:")
print(merged.isnull().sum())