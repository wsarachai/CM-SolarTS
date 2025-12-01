import pandas as pd
import numpy as np
from datetime import datetime
import os
import tensorflow as tf

def preprocess_solar_data(df):
    """
    Preprocess ข้อมูลโซลาร์เซลล์แบบละเอียด
    """
    print("เริ่มต้นการ Preprocess ข้อมูล...")
    
    # สร้าง copy ของ DataFrame เพื่อป้องกันการเปลี่ยนแปลงข้อมูลต้นฉบับ
    df_processed = df.copy()
    
    # 1. แปลงคอลัมน์ Datetime
    print("1. แปลงคอลัมน์ Datetime...")
    df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])
    
    # 2. สร้างคอลัมน์เวลาและวันที่
    print("2. สร้างคอลัมน์เวลาและวันที่...")
    df_processed['Year'] = df_processed['datetime'].dt.year
    df_processed['Month'] = df_processed['datetime'].dt.month
    df_processed['Day'] = df_processed['datetime'].dt.day
    df_processed['Hour'] = df_processed['datetime'].dt.hour
    df_processed['DayOfWeek'] = df_processed['datetime'].dt.dayofweek
    df_processed['DayName'] = df_processed['datetime'].dt.day_name()
    
    # 3. สร้างคอลัมน์ไตรมาส
    print("3. สร้างคอลัมน์ไตรมาส...")
    df_processed['Quarter'] = df_processed['datetime'].dt.quarter
    df_processed['Year_Quarter'] = df_processed['Year'].astype(str) + '-Q' + df_processed['Quarter'].astype(str)
    
    #     5. ภาพรวมบนวงกลม
    # text
    #         เดือน 3 (1.00, 0.00)
    #           ↑
    # เดือน 2 ←   → เดือน 4
    # (0.87,0.50)   (0.87,-0.50)
    #           ↓
    #         เดือน 6 (0.00,-1.00)
    # จะเห็นว่า: เดือน 12 (-0.00, 1.00) อยู่ใกล้เดือน 1 (0.50, 0.87) บนวงกลม!

    # 6. ประโยชน์ในการวิจัยโซลาร์เซลล์
    # สำหรับข้อมูลพลังงานแสงอาทิตย์:
    # ฤดูร้อน (มีนาคม-มิถุนายน) → ค่า sine สูง

    # ฤดูหนาว (พฤศจิกายน-กุมภาพันธ์) → ค่า cosine สูง

    #โมเดลเข้าใจ รูปแบบตามฤดูกาลได้ดีขึ้น
    
    # 4. Cyclical Encoding สำหรับเวลา
    print("4. สร้าง Cyclical Encoding...")
    df_processed['Hour_sin'] = np.sin(2 * np.pi * df_processed['Hour'] / 24)
    df_processed['Hour_cos'] = np.cos(2 * np.pi * df_processed['Hour'] / 24)
    df_processed['Month_sin'] = np.sin(2 * np.pi * df_processed['Month'] / 12)
    df_processed['Month_cos'] = np.cos(2 * np.pi * df_processed['Month'] / 12)
    df_processed['DayOfWeek_sin'] = np.sin(2 * np.pi * df_processed['DayOfWeek'] / 7)
    df_processed['DayOfWeek_cos'] = np.cos(2 * np.pi * df_processed['DayOfWeek'] / 7)
    
    # 5. สร้างคอลัมน์ฤดูกาล
    print("5. สร้างคอลัมน์ฤดูกาล...")
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    df_processed['Season'] = df_processed['Month'].apply(get_season)
    
    # 6. One-Hot Encoding สำหรับฤดูกาล
    season_dummies = pd.get_dummies(df_processed['Season'], prefix='Season')
    df_processed = pd.concat([df_processed, season_dummies], axis=1)
    
    # 7. สร้างคอลัมน์ช่วงเวลาของวัน
    print("6. สร้างคอลัมน์ช่วงเวลาของวัน...")
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df_processed['TimeOfDay'] = df_processed['Hour'].apply(get_time_of_day)
    
    # 8. ตั้งค่ากลางคืน (Night Time Settings)
    print("7. ตั้งค่ากลางคืน...")
    night_mask = (df_processed['Hour'] < 6) | (df_processed['Hour'] > 18)
    
    # คอลัมน์พลังงานที่ควรเป็น 0 ตอนกลางคืน
    energy_cols = ['current_power']
    
    for col in energy_cols:
        if col in df_processed.columns:
            df_processed.loc[night_mask, col] = 0
    
    # 9. การจัดการ Missing Values
    print("8. การจัดการ Missing Values...")
    
    # FFill สำหรับอุณหภูมิและ irradiation
    temp_irradiation_cols = [col for col in df_processed.columns if 'current_power' != col]
    for col in temp_irradiation_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].ffill()
    
    # FFill with limit สำหรับพลังงาน
    energy_cols_ffill = ['current_power']
    for col in energy_cols_ffill:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].ffill(limit=2)
    
    # 10. คำนวณคอลัมน์ใหม่
    print("9. คำนวณคอลัมน์ใหม่...")
    
    # Efficiency Ratio : คำนวณความแตกต่างระหว่างอุณหภูมิแผงโซลาร์เซลล์กับอุณหภูมิแวดล้อม
    if 'temperature_measurement' in df_processed.columns and 'ambient_temperature' in df_processed.columns:
        df_processed['Temp_Difference'] = df_processed['temperature_measurement'] - df_processed['ambient_temperature']
    
    # 11. สร้าง Lag Features 
    print("10. สร้าง Lag Features...")
    df_processed = df_processed.sort_values('datetime')
    
    # Lag 1 hour Lag 1 ชั่วโมง: คาดการณ์พลังงานจากชั่วโมงที่แล้ว
    #ambient_temperature,current_power,temperature_measurement,total_irradiation,utci_mean,cc,q,r,t,fal,sp,t2m,tp
    df_processed['Current_Power_Lag1'] = df_processed['current_power'].shift(1)
    df_processed['Ambient_Temp_Lag1'] = df_processed['ambient_temperature'].shift(1)
    
    
    # Lag 24 hours (วันก่อนหน้า)
    df_processed['Current_Power_Lag24'] = df_processed['current_power'].shift(24)
    df_processed['Ambient_Temp_Lag24'] = df_processed['ambient_temperature'].shift(24)
    
    # 12. Rolling Statistics คำนวณค่าเฉลี่ยจาก 3 แถวล่าสุด (รวมแถวปัจจุบัน)
    # Hour  Power  Rolling_Mean_3h
    # 10:00  100    100.0    ← (100)/1
    # 11:00  150    125.0    ← (100+150)/2  
    # 12:00  200    150.0    ← (100+150+200)/3
    # 13:00  180    176.7    ← (150+200+180)/3
    # พารามิเตอร์:

    # window=3: ใช้ 3 ชั่วโมงล่าสุด

    # min_periods=1: คำนวณได้แม้มีข้อมูลแค่ 1 แถว

    # ประโยชน์ในการวิจัย:

    # 3 ชั่วโมง: จับรูปแบบระยะสั้น (Short-term trend) 24 ชั่วโมง: จับรูปแบบรายวัน (Daily pattern) ลด noise ในข้อมูล
    print("11. สร้าง Rolling Statistics...")
    df_processed['Current_Power_Rolling_Mean_3h'] = df_processed['current_power'].rolling(window=3, min_periods=1).mean()
    df_processed['Current_Power_Rolling_Mean_24h'] = df_processed['current_power'].rolling(window=24, min_periods=1).mean()
    
    df_processed['Ambient_Temp_Rolling_Mean_3h'] = df_processed['ambient_temperature'].rolling(window=3, min_periods=1).mean()
    df_processed['Ambient_Temp_Rolling_Mean_24h'] = df_processed['ambient_temperature'].rolling(window=24, min_periods=1).mean()
    
    # 13. คอลัมน์ Daylight
    print("12. สร้างคอลัมน์ Daylight...")
    df_processed['Is_Daylight'] = ((df_processed['Hour'] >= 6) & (df_processed['Hour'] <= 18)).astype(int)
    
    # 14. สร้างคอลัมน์ Weekend
    print("13. สร้างคอลัมน์ Weekend...")
    df_processed['is_weekend'] = (df_processed['DayOfWeek'] >= 5).astype(int)
    
    # 15. เรียงลำดับตามเวลา
    print("14. เรียงลำดับข้อมูล...")
    df_processed = df_processed.sort_values('datetime').reset_index(drop=True)
    
    print("✅ Preprocessing เสร็จสิ้น!")
    return df_processed

def save_processed_data(df_processed, filename_prefix='solar_data_processed'):
    """
    บันทึกข้อมูลที่ผ่านการ preprocessing แล้ว
    """
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs('src/non-tf/select_lag_sin_cosin_feature/processed_data', exist_ok=True)
    
    # Timestamp สำหรับชื่อไฟล์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # บันทึกเป็น CSV
    csv_filename = f'src/non-tf/select_lag_sin_cosin_feature/processed_data/{filename_prefix}_{timestamp}.csv'
    df_processed.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    # บันทึกเป็น Excel (ยืดหยุ่น — ถ้าไม่มี openpyxl จะไม่ทำให้เกิดข้อผิดพลาด)
    excel_filename = f'src/non-tf/select_lag_sin_cosin_feature/processed_data/{filename_prefix}_{timestamp}.xlsx'
    try:
        # openpyxl is the default engine for .xlsx; if it's missing, pandas raises ModuleNotFoundError
        df_processed.to_excel(excel_filename, index=False)
        excel_wrote = True
    except ModuleNotFoundError:
        print("⚠️  openpyxl not installed — skipping Excel export. Install with: pip install openpyxl")
        excel_wrote = False
    except Exception as e:
        # Other errors writing excel should be reported but not crash the script
        print(f"⚠️  Failed to write Excel file: {e}")
        excel_wrote = False
    
    # บันทึกข้อมูลพื้นฐานเป็น JSON
    info_filename = f'src/non-tf/select_lag_sin_cosin_feature/processed_data/processing_info_{timestamp}.json'
    
    processing_info = {
        'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'original_columns': list(df_processed.columns),
        'data_shape': df_processed.shape,
        'date_range': {
            'start': df_processed['datetime'].min().strftime("%Y-%m-%d %H:%M:%S"),
            'end': df_processed['datetime'].max().strftime("%Y-%m-%d %H:%M:%S")
        },
        'file_paths': {
            'csv': csv_filename,
            'excel': excel_filename if excel_wrote else None
        }
    }
    
    import json
    with open(info_filename, 'w', encoding='utf-8') as f:
        json.dump(processing_info, f, indent=2, ensure_ascii=False)
    
    print(f"✅ บันทึกไฟล์ CSV: {csv_filename}")
    if excel_wrote:
        print(f"✅ บันทึกไฟล์ Excel: {excel_filename}")
    else:
        print(f"⚠️  Excel export skipped (openpyxl missing or error). Use 'pip install openpyxl' to enable it.")
    print(f"✅ บันทึกไฟล์ข้อมูล: {info_filename}")
    
    return csv_filename, (excel_filename if excel_wrote else None)

def analyze_processed_data(df_processed):
    """
    วิเคราะห์ข้อมูลหลัง preprocessing
    """
    print("\n" + "="*50)
    print("การวิเคราะห์ข้อมูลหลัง Preprocessing")
    print("="*50)
    
    print(f"รูปแบบข้อมูล: {df_processed.shape}")
    print(f"ช่วงวันที่: {df_processed['datetime'].min()} ถึง {df_processed['datetime'].max()}")
    print(f"จำนวนวัน: {df_processed['datetime'].dt.date.nunique()} วัน")
    
    print("\nคอลัมน์ทั้งหมด:")
    for i, col in enumerate(df_processed.columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\nจำนวนคอลัมน์ทั้งหมด: {len(df_processed.columns)} คอลัมน์")
    
    # สถิติพื้นฐาน
    print("\nสถิติพื้นฐานของคอลัมน์สำคัญ:")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    important_cols = ['current_power', 'Grid Feed In', 'Internal Power Supply', 
                     'Ambient Temperature', 'Module Temperature', 'Total Irradiation']
    
    for col in important_cols:
        if col in df_processed.columns:
            print(f"\n{col}:")
            print(f"  ค่าเฉลี่ย: {df_processed[col].mean():.2f}")
            print(f"  สูงสุด: {df_processed[col].max():.2f}")
            print(f"  ต่ำสุด: {df_processed[col].min():.2f}")
            print(f"  Missing: {df_processed[col].isnull().sum()}")

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # โหลดข้อมูล
    csv_file = tf.keras.utils.get_file(origin='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/data_15min_clean.csv.zip')
    df = pd.read_csv(csv_file)
    
    # เรียกใช้ฟังก์ชัน preprocessing
    df_processed = preprocess_solar_data(df)
    
    # วิเคราะห์ข้อมูล
    analyze_processed_data(df_processed)
    
    # บันทึกไฟล์
    csv_path, excel_path = save_processed_data(df_processed)
    
    print(f"\n🎯 Preprocessing เสร็จสมบูรณ์!")
    print(f"📁 ไฟล์ที่บันทึก:")
    print(f"   - CSV: {csv_path}")
    print(f"   - Excel: {excel_path}")
    
    # แสดงตัวอย่างข้อมูลหลัง preprocessing
    print(f"\nตัวอย่างข้อมูลหลัง Preprocessing (5 แถวแรก):")
    print(df_processed.head().to_string())