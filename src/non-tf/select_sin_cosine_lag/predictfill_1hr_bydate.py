import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

class CSVSolarPredictor:
    def __init__(self, target_column='current_power'):
        self.target_column = target_column
        self.models = {}
        self.predictions = {}
        self.results = {}
        
    def load_and_prepare_data(self, csv_path, test_size=0.2, random_state=42):
        """โหลดข้อมูลจาก CSV และเตรียมสำหรับ training"""
        print(f"📂 กำลังโหลดข้อมูลจาก: {csv_path}")
        
        # โหลด CSV
        self.df = pd.read_csv(csv_path)
        
        # ตรวจสอบคอลัมน์ Datetime
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            # เรียงข้อมูลตามเวลา
            self.df = self.df.sort_values('datetime')
            print(f"📅 ช่วงวันที่ในข้อมูล: {self.df['datetime'].min()} ถึง {self.df['datetime'].max()}")
            
            # สร้างคอลัมน์ปีและเดือน
            self.df['Year'] = self.df['datetime'].dt.year
            self.df['Month'] = self.df['datetime'].dt.month
            self.df['Day'] = self.df['datetime'].dt.day
        
        # ตรวจสอบ target column
        if self.target_column not in self.df.columns:
            available_cols = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64']]
            print(f"❌ ไม่พบคอลัมน์ '{self.target_column}'")
            print(f"✅ คอลัมน์ที่เป็นตัวเลขที่มี: {available_cols}")
            return None
        
        print(f"✅ โหลดข้อมูลสำเร็จ: {self.df.shape}")
        print(f"✅ คอลัมน์ทั้งหมด: {list(self.df.columns)}")
        
        return self.df
    
    def select_features(self, exclude_columns=None):
        """เลือก features ที่จะใช้ในการฝึกโมเดล"""
        if exclude_columns is None:
        
            exclude_columns = [
               'datetime', 'Season' , 
            ]
        
        # คอลัมน์ที่ไม่ใช้เป็น features
        excluded = exclude_columns + [self.target_column]
        
        # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลขและไม่ใช่ target
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in numerical_features if col not in excluded]
        
        # ลบคอลัมน์ที่อาจมีค่า NaN มากเกินไป
        self.feature_columns = [col for col in self.feature_columns 
                              if self.df[col].isnull().sum() / len(self.df) < 0.5]
        
        print(f"🔧 ใช้ features จำนวน: {len(self.feature_columns)}")
        print("📋 Features ที่เลือก:", self.feature_columns)
        
        return self.feature_columns
    
    def split_data(self, split_by='random', test_size=0.2, years=None, custom_ranges=None):
        """แบ่งข้อมูลเป็น train/test set
        split_by: 'random', 'year', หรือ 'custom'
        custom_ranges: สำหรับ split_by='custom' กำหนดช่วงวันที่
        """
        X = self.df[self.feature_columns]
        y = self.df[self.target_column]
        
        if split_by == 'year' and years and 'Year' in self.df.columns:
            # แบ่งตามปี
            train_mask = self.df['Year'].isin(years['train'])
            test_mask = self.df['Year'].isin(years['test'])
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            print(f"📅 แบ่งตามปี: Train={years['train']}, Test={years['test']}")
            
        elif split_by == 'custom' and custom_ranges and 'datetime' in self.df.columns:
            # วิธีที่ 3: แบ่งตามช่วงวันที่ที่กำหนดเอง
            train_start, train_end = custom_ranges['train']
            test_ranges = custom_ranges['test']
            
            # สร้าง mask สำหรับ training data
            train_mask = (self.df['datetime'] >= train_start) & (self.df['datetime'] <= train_end)
            
            # สร้าง mask สำหรับ test data (หลายช่วง)
            test_mask = pd.Series(False, index=self.df.index)
            for test_range in test_ranges:
                test_start, test_end = test_range
                range_mask = (self.df['datetime'] >= test_start) & (self.df['datetime'] <= test_end)
                test_mask = test_mask | range_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            print(f"📅 แบ่งแบบกำหนดเอง:")
            print(f"   Train: {train_start} ถึง {train_end}")
            print(f"   Test: {len(test_ranges)} ช่วงเวลา")
            for i, (start, end) in enumerate(test_ranges, 1):
                print(f"        {i}. {start} ถึง {end}")
            
        else:
            # แบ่งแบบสุ่ม
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
            print("🎲 แบ่งข้อมูลแบบสุ่ม")
        
        print(f"📊 ขนาด Train set: {X_train.shape}")
        print(f"📊 ขนาด Test set: {X_test.shape}")
        
        # ตรวจสอบว่ามีข้อมูลพอไหม
        if X_train.shape[0] == 0:
            print("⚠️  คำเตือน: Train set ว่างเปล่า!")
        if X_test.shape[0] == 0:
            print("⚠️  คำเตือน: Test set ว่างเปล่า!")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """กำหนดโมเดลที่จะใช้"""
        self.models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.1,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.1,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """ฝึกและประเมินโมเดล"""
        print("\n🚀 เริ่มฝึกโมเดล...")
        
        self.predictions = {}
        self.results = {}

        for name, model in self.models.items():
            print(f"📚 กำลังฝึก {name}...")
            
            try:
                # ฝึกโมเดล
                model.fit(X_train, y_train)
                
                # ทำนาย
                y_pred = model.predict(X_test)
                self.predictions[name] = y_pred
                
                # คำนวณเมตริก
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                self.results[name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'model': model
                }
                
                print(f"   ✅ {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
                
            except Exception as e:
                print(f"   ❌ {name} ผิดพลาด: {e}")
        
        return self.results

    def plot_comparison(self, y_test, n_samples=200):
        """พล็อตผลลัพธ์แบบเปรียบเทียบ"""
        if not self.predictions:
            print("❌ ยังไม่มีผลการทำนาย")
            return
        
        # สร้าง figure และ axes
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Actual vs Predicted (Scatter Plot)
        sample_idx = np.random.choice(len(y_test), min(n_samples, len(y_test)), replace=False)
        
        colors = ['red', 'blue', 'green']
        markers = ['o', 's', '^']
        
        for i, (name, y_pred) in enumerate(self.predictions.items()):
            axes[0, 0].scatter(
                y_test.iloc[sample_idx], 
                y_pred[sample_idx], 
                alpha=0.6, 
                label=name,
                color=colors[i],
                marker=markers[i],
                s=30
            )
        
        # เส้น perfect prediction
        min_val = min(y_test.min(), min([y_pred.min() for y_pred in self.predictions.values()]))
        max_val = max(y_test.max(), max([y_pred.max() for y_pred in self.predictions.values()]))
        
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time Series Prediction (Line Plot)
        if n_samples < len(y_test):
            # พล็อต actual แค่ครั้งเดียว
            axes[0, 1].plot(
                y_test.values[:n_samples], 
                label='Actual', 
                alpha=0.8, 
                color='black', 
                linewidth=2
            )
            
            # พล็อต prediction ของแต่ละโมเดล
            for i, (name, y_pred) in enumerate(self.predictions.items()):
                axes[0, 1].plot(
                    y_pred[:n_samples], 
                    label=name, 
                    alpha=0.7, 
                    color=colors[i],
                    linewidth=1.5
                )
            
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Power Value')
            axes[0, 1].set_title('Time Series Prediction (First {} Samples)'.format(n_samples))
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model Comparison (Bar Chart)
        model_names = list(self.results.keys())
        mae_values = [self.results[name]['MAE'] for name in model_names]
        rmse_values = [self.results[name]['RMSE'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x_pos - width/2, mae_values, width, label='MAE', color='skyblue', alpha=0.8)
        bars2 = axes[1, 0].bar(x_pos + width/2, rmse_values, width, label='RMSE', color='lightcoral', alpha=0.8)
        
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Error Values')
        axes[1, 0].set_title('Model Comparison (MAE vs RMSE)')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # เพิ่มค่าบน bar
        for bar, value in zip(bars1, mae_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01, 
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        for bar, value in zip(bars2, rmse_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01, 
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Feature Importance (Horizontal Bar Chart)
        best_model_name = min(self.results.items(), key=lambda x: x[1]['MAE'])[0]
        best_model = self.results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)  # Top 10 features
            
            if len(feature_imp) > 0:
                axes[1, 1].barh(feature_imp['feature'], feature_imp['importance'], color='lightgreen', alpha=0.8)
                axes[1, 1].set_xlabel('Importance Score')
                axes[1, 1].set_title(f'Top 10 Feature Importance - {best_model_name}')
                
                # เพิ่มค่า importance บน bar
                for i, (_, row) in enumerate(feature_imp.iterrows()):
                    axes[1, 1].text(row['importance'] + max(feature_imp['importance'])*0.01, i, 
                                   f'{row["importance"]:.3f}', va='center', fontsize=9)
            else:
                axes[1, 1].text(0.5, 0.5, 'No feature importance data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Feature Importance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        # ปรับ layout
        plt.tight_layout()
        
        # เพิ่ม overall title
        fig.suptitle('Solar Power Prediction Model Comparison', fontsize=16, y=1.02)
        
        plt.show()
        
        # พิมพ์สรุปผล
        print("\n📊 สรุปผลการประเมินโมเดล:")
        print("-" * 50)
        for name, result in self.results.items():
            print(f"🏷️  {name}:")
            print(f"   MAE:  {result['MAE']:.2f}")
            print(f"   RMSE: {result['RMSE']:.2f}")
            print(f"   R²:   {result['R2']:.4f}")
            print()
    
    def save_models(self, folder_path='src/non-tf/select_sin_cosine_lag/saved_models'):
        """บันทึกโมเดลที่ฝึกแล้ว"""
        os.makedirs(folder_path, exist_ok=True)
        
        for name, result in self.results.items():
            filename = os.path.join(folder_path, f'{name}_model.pkl')
            joblib.dump(result['model'], filename)
            print(f"💾 บันทึก {name} ที่: {filename}")
    
    def load_model(self, model_path, model_name):
        """โหลดโมเดลที่บันทึกไว้"""
        model = joblib.load(model_path)
        self.models[model_name] = model
        print(f"📥 โหลดโมเดล {model_name} สำเร็จ")
        return model

# 2. ฟังก์ชันใช้งานแบบง่ายๆ
def run_analysis_from_csv(csv_path, target_column='current_power', split_method='year'):
    """รันการวิเคราะห์ทั้งหมดจากไฟล์ CSV"""
    print("="*60)
    print("🔬 Solar Power Prediction from CSV")
    print("="*60)
    
    # สร้าง predictor
    predictor = CSVSolarPredictor(target_column)
    
    # 1. โหลดข้อมูล
    df = predictor.load_and_prepare_data(csv_path)
    if df is None:
        return
    
    # 2. เลือก features
    feature_columns = predictor.select_features()
    
    # 3. แบ่งข้อมูล (เลือกวิธีใดวิธีหนึ่ง)
    if split_method == 'random':
        # วิธีที่ 1: แบ่งแบบสุ่ม
        X_train, X_test, y_train, y_test = predictor.split_data(split_by='random', test_size=0.2)
        
    elif split_method == 'year':
        # วิธีที่ 2: แบ่งตามปี (ถ้ามีคอลัมน์ Year)
        X_train, X_test, y_train, y_test = predictor.split_data(
            split_by='year', 
            years={'train': [2022, 2023, 2024], 'test': [2021]}
        )
        
    elif split_method == 'custom':
        # วิธีที่ 3: แบ่งตามช่วงวันที่ที่กำหนดเอง
        # Train: 2022-2024 ทุกเดือน, Test: 2021 เดือน 6-12 และ 2025 เดือน 1-4
        custom_ranges = {
            'train': ('2022-01-01', '2024-12-31'),
            'test': [
                ('2021-06-01', '2021-12-31'),  # 2021 เดือน 6-12
                ('2025-01-01', '2025-04-30')   # 2025 เดือน 1-4
            ]
        }
        X_train, X_test, y_train, y_test = predictor.split_data(
            split_by='custom', 
            custom_ranges=custom_ranges
        )
    
    # 4. กำหนดและฝึกโมเดล
    predictor.initialize_models()
    results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 5. แสดงผลลัพธ์
    predictor.plot_comparison(y_test)
    
    # 6. บันทึกโมเดล
    predictor.save_models()
    
    # 7. สรุปผล
    for model_name, metrics in results.items():
        print(f"\n🔍 โมเดล: {model_name}")
        print(f"📊 MAE: {metrics['MAE']:.2f}")
        print(f"📊 RMSE: {metrics['RMSE']:.2f}")
        print(f"📊 R²: {metrics['R2']:.4f}")
    
    best_model = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\n🏆 โมเดลที่ดีที่สุด: {best_model[0]}")
    print(f"📊 MAE: {best_model[1]['MAE']:.2f}")
    print(f"📊 RMSE: {best_model[1]['RMSE']:.2f}")
    print(f"📊 R²: {best_model[1]['R2']:.4f}")
    
    return predictor, results

# 3. ฟังก์ชันสำหรับวิธีที่ 3 โดยเฉพาะ
def run_custom_split_analysis(csv_path, target_column='current_power', train_range=None, test_ranges=None):
    """รันการวิเคราะห์ด้วยการแบ่งข้อมูลแบบกำหนดเอง"""
    print("="*60)
    print("🔬 Solar Power Prediction - Custom Split Analysis")
    print("="*60)
    
    # กำหนดค่าเริ่มต้นถ้าไม่ระบุ
    if train_range is None:
        train_range = ('2022-01-01', '2024-12-31')  # 2022-2024 ทุกเดือน
    
    if test_ranges is None:
        test_ranges = [
            ('2021-11-01', '2021-12-31'),  # 2021 เดือน 6-12
            ('2025-01-01', '2025-04-30')   # 2025 เดือน 1-4
        ]
    
    # สร้าง predictor
    predictor = CSVSolarPredictor(target_column)
    
    # 1. โหลดข้อมูล
    df = predictor.load_and_prepare_data(csv_path)
    if df is None:
        return
    
    # 2. เลือก features
    feature_columns = predictor.select_features()
    
    # 3. แบ่งข้อมูลแบบกำหนดเอง
    custom_ranges = {
        'train': train_range,
        'test': test_ranges
    }
    
    X_train, X_test, y_train, y_test = predictor.split_data(
        split_by='custom', 
        custom_ranges=custom_ranges
    )
    
    # ตรวจสอบว่ามีข้อมูลพอไหม
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("❌ ไม่มีข้อมูลเพียงพอสำหรับการฝึกหรือทดสอบ")
        print("📊 ข้อมูลที่มีอยู่ในช่วงวันที่:")
        print(f"   ข้อมูลทั้งหมด: {df['datetime'].min()} ถึง {df['datetime'].max()}")
        return
    
    # 4. กำหนดและฝึกโมเดล
    predictor.initialize_models()
    results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # 5. แสดงผลลัพธ์
    predictor.plot_comparison(y_test)
    
    # 6. บันทึกโมเดล
    predictor.save_models()
    
    # 7. สรุปผล
    for model_name, metrics in results.items():
        print(f"\n🔍 โมเดล: {model_name}")
        print(f"📊 MAE: {metrics['MAE']:.2f}")
        print(f"📊 RMSE: {metrics['RMSE']:.2f}")
        print(f"📊 R²: {metrics['R2']:.4f}")
    
    best_model = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\n🏆 โมเดลที่ดีที่สุด: {best_model[0]}")
    print(f"📊 MAE: {best_model[1]['MAE']:.2f}")
    print(f"📊 RMSE: {best_model[1]['RMSE']:.2f}")
    print(f"📊 R²: {best_model[1]['R2']:.4f}")
    
    return predictor, results

# 4. ตัวอย่างการใช้งาน
if __name__ == "__main__":
    csv_file_path = "src/non-tf/select_sin_cosine_lag/processed_data/solar_data_simple_fill.csv"  # เปลี่ยนเป็น path ของคุณ
    
    print("เลือกวิธีแบ่งข้อมูล:")
    print("1. แบ่งแบบสุ่ม")
    print("2. แบ่งตามปี")
    print("3. แบ่งตามช่วงวันที่ที่กำหนดเอง (Train 2022-2024, Test 2021เดือน6-12 + 2025เดือน1-4)")

    choice = input("เลือกวิธี (1/2/3): ").strip()

    if choice == '1':
        # วิธีที่ 1: แบ่งแบบสุ่ม
        predictor, results = run_analysis_from_csv(
            csv_path=csv_file_path,
            target_column='current_power',
            split_method='random'
        )
    elif choice == '2':
        # วิธีที่ 2: แบ่งตามปี
        predictor, results = run_analysis_from_csv(
            csv_path=csv_file_path,
            target_column='current_power',
            split_method='year'
        )
    elif choice == '3':
        # วิธีที่ 3: แบ่งตามช่วงวันที่ที่กำหนดเอง
        predictor, results = run_custom_split_analysis(
            csv_path=csv_file_path,
            target_column='current_power'
        )
    else:
        print("❌ ไม่มีตัวเลือกนี้ ใช้วิธีที่ 3 เป็นค่าเริ่มต้น")
        predictor, results = run_custom_split_analysis(
            csv_path=csv_file_path,
            target_column='current_power'
        )