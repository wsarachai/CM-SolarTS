import pandas as pd
import numpy as np
from datetime import datetime
import os

def preprocess_solar_data_simple_fill(df):
    """
    Preprocess ข้อมูลโซลาร์เซลล์แบบง่าย - จัดการ Missing Values ด้วย ffill เท่านั้น
    และเพิ่ม Lag Features
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
    
    # 6. เพิ่ม Lag Features และ Time Features
    print("\n6. เพิ่ม Lag Features และ Time Features...")
    df_processed = add_time_features(df_processed)  # เพิ่ม time features ก่อน
    df_processed = add_lag_features(df_processed)   # แล้วค่อยเพิ่ม lag features
    
    return df_processed

def add_lag_features(df):
    """
    เพิ่ม Lag Features (1 ชั่วโมงและ 24 ชั่วโมง) สำหรับคอลัมน์สำคัญ
    """
    df_lagged = df.copy()

    # กำหนดคอลัมน์ที่จะสร้าง lag features ตามรูปแบบที่ต้องการ
    lag_mapping = {
        'current_power': 'Current_Power',
        'ambient_temperature': 'Ambient_Temp',
        'temperature_measurement': 'Module_Temp', 
        'total_irradiation': 'Total_Irradiation',
        'utci_mean': 'UTCI_Mean',
        'cc': 'Cloud_Cover',
        'q': 'Specific_Humidity',
        'r': 'Relative_Humidity',
        't': 'Air_Temperature',
        'fal': 'Forecast_Accumulated_Liquid',
        'sp': 'Surface_Pressure',
        't2m': '2m_Temperature',
        'tp': 'Total_Precipitation',
        'wind_speed': 'Wind_Speed',
        'wind_direction': 'Wind_Direction',
        'wind_speed10': 'Wind_Speed10',
        'wind_direction10': 'Wind_Direction10',
        'net_radiation': 'Net_Radiation',
        'total_downward_radiation': 'Total_Downward_Radiation',
        'net_heat_flux': 'Net_Heat_Flux',
        'dewpoint': 'Dewpoint',
        'dewpoint2m': 'Dewpoint2m'
    }
    
    # กรองเฉพาะคอลัมน์ที่มีอยู่ใน DataFrame
    available_mapping = {orig: new for orig, new in lag_mapping.items() if orig in df_lagged.columns}
    
    print(f"สร้าง Lag Features สำหรับคอลัมน์: {list(available_mapping.keys())}")
    
    # สร้าง lag 1 ชั่วโมง
    for orig_col, new_prefix in available_mapping.items():
        df_lagged[f'{new_prefix}_Lag1'] = df_lagged[orig_col].shift(1)
        print(f"  ✅ สร้าง {new_prefix}_Lag1 จาก {orig_col}")
    
    # สร้าง lag 24 ชั่วโมง
    for orig_col, new_prefix in available_mapping.items():
        df_lagged[f'{new_prefix}_Lag24'] = df_lagged[orig_col].shift(24)
        print(f"  ✅ สร้าง {new_prefix}_Lag24 จาก {orig_col}")
    
    # นับจำนวนคอลัมน์ที่เพิ่มมา
    original_cols = len([col for col in df.columns if 'Lag' not in col])
    new_cols = len([col for col in df_lagged.columns if 'Lag' not in col])
    print(f"เพิ่ม Lag Features: {len(available_mapping) * 2} คอลัมน์")
    
    return df_lagged

def add_time_features(df):
    """
    เพิ่มฟีเจอร์เวลาสำหรับช่วยในการทำนาย
    """
    df_time = df.copy()
    
    # แยกส่วนต่างๆ ของเวลา
    df_time['hour'] = df_time['datetime'].dt.hour
    df_time['day_of_week'] = df_time['datetime'].dt.dayofweek  # 0=จันทร์, 6=อาทิตย์
    df_time['day_of_month'] = df_time['datetime'].dt.day
    df_time['month'] = df_time['datetime'].dt.month
    df_time['is_weekend'] = (df_time['day_of_week'] >= 5).astype(int)
    
    # เพิ่มฟีเจอร์ตามฤดูกาล (สำหรับโซลาร์เซลล์สำคัญมาก)
    df_time['season'] = df_time['month'] % 12 // 3 + 1
    # 1: ฤดูหนาว (ธ.ค.-ก.พ.), 2: ฤดูร้อน (มี.ค.-พ.ค.), 
    # 3: ฤดูฝน (มิ.ย.-ส.ค.), 4: ฤดูใบไม้ร่วง (ก.ย.-พ.ย.)
    
    # เพิ่มฟีเจอร์เวลาของวัน (กลางคืน/กลางวัน)
    df_time['is_daytime'] = ((df_time['hour'] >= 6) & (df_time['hour'] <= 18)).astype(int)
    
    print("✅ เพิ่มฟีเจอร์เวลาเสร็จสิ้น")
    print(f"  - ชั่วโมง, วันในสัปดาห์, วันในเดือน, เดือน")
    print(f"  - วันหยุด, ฤดูกาล, ช่วงกลางวัน/กลางคืน")
    
    return df_time

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
    important_cols = ['ambient_temperature', 'current_power', 'temperature_measurement',
       'total_irradiation', 'utci_mean', 'cc', 'q', 'r', 't', 'fal', 'sp',
       't2m', 'tp', 'wind_speed', 'wind_direction', 'wind_speed10',
       'wind_direction10', 'net_radiation', 'total_downward_radiation',
       'net_heat_flux', 'dewpoint', 'dewpoint2m']
    
    for col in important_cols:
        if col in filled_df.columns:
            print(f"\n{col}:")
            print(f"  ค่าเฉลี่ย: {filled_df[col].mean():.2f}")
            print(f"  สูงสุด: {filled_df[col].max():.2f}")
            print(f"  ต่ำสุด: {filled_df[col].min():.2f}")
            print(f"  Missing: {filled_df[col].isnull().sum()}")
    
    # แสดงตัวอย่าง lag features
    lag_cols = [col for col in filled_df.columns if 'Lag' in col]
    if lag_cols:
        print(f"\nLag Features ที่สร้าง:")
        for col in lag_cols[:8]:  # แสดงแค่ 8 คอลัมน์แรก
            missing_count = filled_df[col].isnull().sum()
            print(f"  {col}: {missing_count} missing values")

def save_simple_processed_data(df_processed, filename_prefix='solar_data_simple_fill'):
    """
    บันทึกข้อมูลที่ผ่านการ fill แล้ว
    """
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs('src/non-tf/select_lag/processed_data', exist_ok=True)
    
    # Timestamp สำหรับชื่อไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # บันทึกเป็น CSV
    csv_filename = f'src/non-tf/select_lag/processed_data/{filename_prefix}_{timestamp}.csv'
    df_processed.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ บันทึกไฟล์: {csv_filename}")
    
    # บันทึกข้อมูลสถิติเกี่ยวกับคอลัมน์ใหม่
    original_columns = ['ambient_temperature', 'current_power', 'temperature_measurement',
       'total_irradiation', 'utci_mean', 'cc', 'q', 'r', 't', 'fal', 'sp',
       't2m', 'tp', 'wind_speed', 'wind_direction', 'wind_speed10',
       'wind_direction10', 'net_radiation', 'total_downward_radiation',
       'net_heat_flux', 'dewpoint', 'dewpoint2m']
    
    new_columns = [col for col in df_processed.columns if col not in original_columns]
    print(f"📊 คอลัมน์ใหม่ที่เพิ่ม: {len(new_columns)} คอลัมน์")
    
    # แยกแสดง lag features ตามประเภท
    lag1_cols = [col for col in new_columns if 'Lag1' in col]
    lag24_cols = [col for col in new_columns if 'Lag24' in col]
    time_cols = [col for col in new_columns if 'Lag' not in col]
    
    if lag1_cols:
        print(f"   - Lag 1h: {len(lag1_cols)} คอลัมน์")
    if lag24_cols:
        print(f"   - Lag 24h: {len(lag24_cols)} คอลัมน์")
    if time_cols:
        print(f"   - Time Features: {len(time_cols)} คอลัมน์")
    
    return csv_filename

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # โหลดข้อมูล
    df = pd.read_csv('data_15min_clean.csv')
    
    print("ข้อมูลต้นฉบับ:")
    print(f"รูปแบบ: {df.shape}")
    print(f"จำนวน missing values ทั้งหมด: {df.isnull().sum().sum()}")
    
    # เรียกใช้ฟังก์ชัน fill อย่างง่าย
    df_filled = preprocess_solar_data_simple_fill(df)
    
    # วิเคราะห์ผลลัพธ์
    analyze_fill_results(df, df_filled)
    
    # บันทึกไฟล์
    csv_path = save_simple_processed_data(df_filled)
    
    print(f"\n🎯 การจัดการ Missing Values และการสร้าง Features เสร็จสมบูรณ์!")
    print(f"📁 ไฟล์ที่บันทึก: {csv_path}")
    print(f"📈 ขนาดข้อมูลหลังเพิ่ม features: {df_filled.shape}")
    
    # แสดงตัวอย่างข้อมูลหลัง fill - แสดงเฉพาะ field หลักตามที่ต้องการ
    print(f"\nตัวอย่างข้อมูลหลัง Fill (5 แถวแรก) - แสดงเฉพาะ Field หลัก:")
    
    # กำหนด field หลักตามที่คุณต้องการ
    
    main_fields = [
        'datetime', 'ambient_temperature', 'current_power', 'temperature_measurement',
       'total_irradiation', 'utci_mean', 'cc', 'q', 'r', 't', 'fal', 'sp',
       't2m', 'tp', 'wind_speed', 'wind_direction', 'wind_speed10',
       'wind_direction10', 'net_radiation', 'total_downward_radiation',
       'net_heat_flux', 'dewpoint', 'dewpoint2m', 'Day sin', 'Day cos',
       'Year sin', 'Year cos', 'day_sin_39.0d', 'day_cos_39.0d',
       'day_sin_19.5d', 'day_cos_19.5d', 'Year', 'Month', 'Day']
    
    # กรองเฉพาะ field ที่มีอยู่ใน DataFrame
    available_main_fields = [col for col in main_fields if col in df_filled.columns]
    
    # แสดงเฉพาะ field หลัก
    print(df_filled[available_main_fields].head().to_string())
    
    # แสดงตัวอย่าง lag features แยกต่างหาก
    print(f"\nตัวอย่าง Lag Features (5 แถวแรก):")
    lag_fields = [col for col in df_filled.columns if 'Lag' in col]
    if lag_fields:
        # เลือกแสดงบาง lag features ที่สำคัญ
        important_lags = ['Current_Power_Lag1', 'Current_Power_Lag24', 
                         'Ambient_Temp_Lag1', 'Ambient_Temp_Lag24']
        available_lags = [col for col in important_lags if col in df_filled.columns]
        if available_lags:
            display_lag_data = df_filled[['datetime'] + available_lags].head()
            print(display_lag_data.to_string())