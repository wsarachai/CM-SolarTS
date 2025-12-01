import pandas as pd
import numpy as np
from datetime import datetime
import os

def preprocess_solar_data_simple_fill(df):
    """
    Preprocess ข้อมูลโซลาร์เซลล์แบบง่าย - จัดการ Missing Values ด้วย ffill เท่านั้น
    """
    print("เริ่มต้นการจัดการ Missing Values ด้วย ffill...")
    
    # สร้าง copy ของ DataFrame เพื่อป้องกันการเปลี่ยนแปลงข้อมูลต้นฉบับ
    df_processed = df.copy()
    
    # 1. แปลงคอลัมน์ Datetime (แก้ไขปัญหา format)
    print("1. แปลงคอลัมน์ Datetime...")
    
    # ตรวจสอบรูปแบบวันที่ในข้อมูลก่อน
    print(f"ตัวอย่าง Datetime ในข้อมูล: {df_processed['datetime'].iloc[150:160].tolist()}")
    
    # ใช้การแปลงวันที่แบบอัตโนมัติ (ไม่ต้องระบุ format)
    try:
        df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])
        print("✅ แปลง Datetime สำเร็จด้วย pd.to_datetime() แบบอัตโนมัติ")
    except Exception as e:
        print(f"❌ การแปลงแบบอัตโนมัติล้มเหลว: {e}")
        # ลองใช้วิธีอื่น
        try:
            df_processed['datetime'] = pd.to_datetime(df_processed['datetime'], format='%Y-%m-%d %H:%M:%S')
            print("✅ แปลง Datetime สำเร็จด้วย format '%Y-%m-%d %H:%M:%S'")
        except Exception as e2:
            print(f"❌ การแปลงด้วย format ล้มเหลว: {e2}")
            # ใช้วิธีสุดท้าย - แปลงแบบ errors='coerce'
            df_processed['datetime'] = pd.to_datetime(df_processed['datetime'], errors='coerce')
            print("⚠️  แปลง Datetime ด้วย errors='coerce' (ค่าที่แปลงไม่ได้จะเป็น NaT)")
    
    # 2. เรียงลำดับตามเวลาก่อนทำ ffill
    print("2. เรียงลำดับข้อมูลตามเวลา...")
    df_processed = df_processed.sort_values('datetime').reset_index(drop=True)
    
    # 3. การจัดการ Missing Values ด้วย ffill
    print("3. การจัดการ Missing Values ด้วย ffill...")
    
    # กำหนดคอลัมน์ต่างๆ และวิธีการ fill ที่เหมาะสม
    fill_strategies = {
        # อุณหภูมิและ irradiation - ใช้ ffill แบบไม่มี limit
        'temperature_irradiation': {
            'columns': ['ambient_temperature', 'temperature_measurement',
            'total_irradiation', 'utci_mean', 'cc', 'q', 'r', 't', 'fal', 'sp',
            't2m', 'tp', 'wind_speed', 'wind_direction', 'wind_speed10',
            'wind_direction10', 'net_radiation', 'total_downward_radiation',
            'net_heat_flux', 'dewpoint', 'dewpoint2m'],
            'method': 'ffill',
            'limit': None
        },
        # ข้อมูลพลังงาน - ใช้ ffill แบบมี limit เพื่อป้องกันการ fill ที่ยาวเกินไป
        'energy': {
            'columns': ['current_power'],
            'method': 'ffill', 
            'limit': 6  # limit ที่ 6 ชั่วโมง (ครึ่งวัน)
        }
    }
    
    # นับจำนวน missing values ก่อนทำการ fill
    print("\nจำนวน Missing Values ก่อนทำการ fill:")
    for category, strategy in fill_strategies.items():
        for col in strategy['columns']:
            if col in df_processed.columns:
                missing_count = df_processed[col].isnull().sum()
                if missing_count > 0:
                    print(f"  {col}: {missing_count} missing values")
    
    # ทำการ fill ข้อมูลตาม strategy ที่กำหนด
    for category, strategy in fill_strategies.items():
        for col in strategy['columns']:
            if col in df_processed.columns:
                if strategy['method'] == 'ffill':
                    df_processed[col] = df_processed[col].ffill(limit=strategy['limit'])
                    print(f"  ✅ {col}: ffill with limit={strategy['limit']}")
    
    # 4. ตรวจสอบผลลัพธ์หลัง fill
    print("\nจำนวน Missing Values หลังทำการ fill:")
    total_missing_after = 0
    all_columns = []
    for category, strategy in fill_strategies.items():
        all_columns.extend(strategy['columns'])
    
    for col in all_columns:
        if col in df_processed.columns:
            missing_count = df_processed[col].isnull().sum()
            total_missing_after += missing_count
            if missing_count > 0:
                print(f"  ⚠️  {col}: {missing_count} missing values (ยังเหลือ)")
            else:
                print(f"  ✅ {col}: ไม่มี missing values")
    
    print(f"\nรวม missing values ที่เหลือ: {total_missing_after}")
    
    # 5. สำหรับ missing values ที่ยังเหลืออยู่ ให้ใช้วิธีอื่นเสริม
    if total_missing_after > 0:
        print("\n5. จัดการ missing values ที่เหลือด้วยวิธีเสริม...")
        
        # สำหรับข้อมูลตัวเลข: ใช้ค่าเฉลี่ยของข้อมูลที่อยู่รอบๆ
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                # ใช้ interpolation สำหรับข้อมูลตัวเลข
                df_processed[col] = df_processed[col].interpolate(method='linear')
                remaining = df_processed[col].isnull().sum()
                if remaining > 0:
                    # หากยังมี missing อยู่ให้ใช้ค่าเฉลี่ย
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                print(f"  ✅ {col}: interpolate + fillna(mean)")
    
    print("✅ การจัดการ Missing Values เสร็จสิ้น!")
    return df_processed

def add_cyclical_time_features(df):
    """
    เพิ่มฟีเจอร์เวลาแบบ cyclical (sin/cos) สำหรับการทำโมเดล machine learning
    """
    print("\n🔧 เพิ่มฟีเจอร์เวลาแบบ cyclical...")
    
    # สร้างคอลัมน์เวลาจาก Datetime
    df_with_features = df.copy()
    
    # แยกส่วนต่างๆ ของเวลา
    df_with_features['hour'] = df_with_features['datetime'].dt.hour
    df_with_features['day_of_week'] = df_with_features['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_with_features['day_of_month'] = df_with_features['datetime'].dt.day
    df_with_features['month'] = df_with_features['datetime'].dt.month
    
    # สร้าง cyclical features ด้วย sin/cos transformation
    # ชั่วโมง (24-hour cycle)
    df_with_features['hour_sin'] = np.sin(2 * np.pi * df_with_features['hour'] / 24)
    df_with_features['hour_cos'] = np.cos(2 * np.pi * df_with_features['hour'] / 24)
    
    # วันในสัปดาห์ (7-day cycle)
    df_with_features['day_of_week_sin'] = np.sin(2 * np.pi * df_with_features['day_of_week'] / 7)
    df_with_features['day_of_week_cos'] = np.cos(2 * np.pi * df_with_features['day_of_week'] / 7)
    
    # วันในเดือน (ประมาณ 30-day cycle)
    df_with_features['day_of_month_sin'] = np.sin(2 * np.pi * df_with_features['day_of_month'] / 30)
    df_with_features['day_of_month_cos'] = np.cos(2 * np.pi * df_with_features['day_of_month'] / 30)
    
    # เดือน (12-month cycle)
    df_with_features['month_sin'] = np.sin(2 * np.pi * df_with_features['month'] / 12)
    df_with_features['month_cos'] = np.cos(2 * np.pi * df_with_features['month'] / 12)
    
    print("✅ เพิ่ม cyclical features สำเร็จ:")
    print(f"   - hour_sin, hour_cos (24-hour cycle)")
    print(f"   - day_of_week_sin, day_of_week_cos (7-day cycle)")
    print(f"   - day_of_month_sin, day_of_month_cos (30-day cycle)")
    print(f"   - month_sin, month_cos (12-month cycle)")
    
    # ลบคอลัมน์กลางที่ไม่ต้องการใช้
    df_with_features = df_with_features.drop(['hour', 'day_of_week', 'day_of_month', 'month'], axis=1)
    
    return df_with_features

def analyze_fill_results(original_df, filled_df):
    """
    วิเคราะห์ผลลัพธ์การ fill ข้อมูล
    """
    print("\n" + "="*50)
    print("การวิเคราะห์ผลลัพธ์การ Fill ข้อมูล")
    print("="*50)
    
    # เปรียบเทียบ missing values ก่อนและหลัง
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    print("\nการเปรียบเทียบ Missing Values:")
    print("คอลัมน์".ljust(25) + "ก่อน fill".ljust(12) + "หลัง fill".ljust(12) + "ลดลง")
    print("-" * 60)
    
    total_reduction = 0
    for col in numeric_cols:
        if col in original_df.columns and col in filled_df.columns:
            before = original_df[col].isnull().sum()
            after = filled_df[col].isnull().sum()
            reduction = before - after
            total_reduction += reduction
            
            print(f"{col.ljust(25)}{str(before).ljust(12)}{str(after).ljust(12)}{reduction}")
    
    print("-" * 60)
    print(f"รวมลดลง: {total_reduction} missing values")
    
    # แสดงสถิติพื้นฐาน
    print("\nสถิติพื้นฐานหลัง fill:")
    important_cols = ['Current Power', 'Grid Feed In', 'Internal Power Supply', 
                     'Ambient Temperature', 'Module Temperature', 'Total Irradiation']
    
    for col in important_cols:
        if col in filled_df.columns:
            print(f"\n{col}:")
            print(f"  ค่าเฉลี่ย: {filled_df[col].mean():.2f}")
            print(f"  สูงสุด: {filled_df[col].max():.2f}")
            print(f"  ต่ำสุด: {filled_df[col].min():.2f}")
            print(f"  Missing: {filled_df[col].isnull().sum()}")

def save_simple_processed_data(df_processed, filename_prefix='solar_data_simple_fill'):
    """
    บันทึกข้อมูลที่ผ่านการ fill แล้ว
    """
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs('select_sin_cosine/processed_data', exist_ok=True)
    
    # Timestamp สำหรับชื่อไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # บันทึกเป็น CSV
    csv_filename = f'select_sin_cosine/processed_data/{filename_prefix}_{timestamp}.csv'
    df_processed.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ บันทึกไฟล์: {csv_filename}")
    return csv_filename

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # โหลดข้อมูล
    df = pd.read_csv('merge_15min_added.csv')
    
    print("ข้อมูลต้นฉบับ:")
    print(f"รูปแบบ: {df.shape}")
    print(f"จำนวน missing values ทั้งหมด: {df.isnull().sum().sum()}")
    
    # เรียกใช้ฟังก์ชัน fill อย่างง่าย
    df_filled = preprocess_solar_data_simple_fill(df)
    
    # เพิ่ม cyclical time features
    df_with_features = add_cyclical_time_features(df_filled)
    
    # วิเคราะห์ผลลัพธ์
    analyze_fill_results(df, df_filled)
    
    # บันทึกไฟล์
    csv_path = save_simple_processed_data(df_with_features)
    
    print(f"\n🎯 การจัดการ Missing Values และเพิ่มฟีเจอร์เวลาเสร็จสมบูรณ์!")
    print(f"📁 ไฟล์ที่บันทึก: {csv_path}")
    
    # แสดงตัวอย่างข้อมูลหลัง fill
    print(f"\nตัวอย่างข้อมูลหลัง Fill และเพิ่มฟีเจอร์ (5 แถวแรก):")
    print(df_with_features.head().to_string())
    
    # แสดงคอลัมน์ทั้งหมด
    print(f"\nคอลัมน์ทั้งหมดในข้อมูลที่ประมวลผลแล้ว:")
    print(df_with_features.columns.tolist())
    
    # ตรวจสอบ cyclical features
    cyclical_cols = [col for col in df_with_features.columns if 'sin' in col or 'cos' in col]
    print(f"\nCyclical features ที่เพิ่มเข้ามา: {cyclical_cols}")
    
    # แสดงสถิติของ cyclical features
    print(f"\nสถิติของ Cyclical Features:")
    for col in cyclical_cols:
        print(f"{col}: min={df_with_features[col].min():.3f}, max={df_with_features[col].max():.3f}, mean={df_with_features[col].mean():.3f}")