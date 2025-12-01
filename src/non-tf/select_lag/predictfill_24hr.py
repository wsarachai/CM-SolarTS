import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CSVSolarPredictor:
    def __init__(self, target_column='current_power'):
        self.target_column = target_column
        self.df = None
        self.feature_columns = None
        self.models = {}
        self.model_features = {}  # เก็บข้อมูล features ของแต่ละโมเดล
        
    def load_data(self, csv_path):
        """Load data from CSV"""
        self.df = pd.read_csv(csv_path)
        
        # Convert Datetime column to datetime object
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        print(f"✅ Data loaded successfully: {len(self.df)} rows")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        return self.df
    
    def create_basic_features(self, df):
        """Create basic numerical features only (avoid categorical)"""
        df = df.copy()
        
        # Create basic time features (numerical only)
        df['Year'] = df['datetime'].dt.year
        df['Month'] = df['datetime'].dt.month
        df['Day'] = df['datetime'].dt.day
        df['Hour'] = df['datetime'].dt.hour
        df['DayOfWeek'] = df['datetime'].dt.dayofweek
        df['Quarter'] = df['datetime'].dt.quarter
        
        # Create cyclical features (numerical)
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * (df['Month'] - 1) / 12)
        df['Month_cos'] = np.cos(2 * np.pi * (df['Month'] - 1) / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Create day/night feature
        df['Is_Daylight'] = df['Hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)
        
        return df
    
    def prepare_features(self, use_columns=None):
        """Prepare features for model training - numerical only"""
        if self.df is None:
            raise ValueError("❌ Load data before calling this function")
        
        # Create basic features (numerical only)
        self.df = self.create_basic_features(self.df)
        
        # Check target data
        print(f"📊 Target data ('{self.target_column}'):")
        print(f"   Mean: {self.df[self.target_column].mean():.2f}")
        print(f"   Max: {self.df[self.target_column].max():.2f}")
        print(f"   Min: {self.df[self.target_column].min():.2f}")
        print(f"   NaN values: {self.df[self.target_column].isna().sum()}")

        # Select only numerical feature columns
        if use_columns is None:
            # Exclude datetime, target, and categorical columns
            exclude_columns = ['datetime', self.target_column, 'Grid Feed In', 
                             'Internal Power Supply', 'External Energy Supply',
                             'Self Consumption', 'Self_Consumption_%']
            
            # Select only numerical columns
            numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numerical_columns 
                                  if col not in exclude_columns 
                                  and not col.startswith('Unnamed')
                                  and not self.df[col].isna().all()]
        
        print(f"📊 Using {len(self.feature_columns)} numerical features: {self.feature_columns}")
        
        # Check for NaN values in features
        for col in self.feature_columns:
            nan_count = self.df[col].isna().sum()
            if nan_count > 0:
                print(f"⚠️  Column {col} has {nan_count} NaN values")
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        return self.df, self.feature_columns
    
    def load_models_from_folder(self, models_folder):
        """Load all trained models and their expected features"""
        if not os.path.exists(models_folder):
            raise ValueError(f"❌ Models folder '{models_folder}' does not exist")
        
        model_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]
        
        if not model_files:
            raise ValueError(f"❌ No model files found in '{models_folder}'")
        
        print(f"📁 Loading models from '{models_folder}':")
        
        for model_file in model_files:
            model_name = os.path.splitext(model_file)[0]
            model_path = os.path.join(models_folder, model_file)
            
            try:
                model = joblib.load(model_path)
                self.models[model_name] = model
                
                # เก็บข้อมูล features ที่โมเดลคาดหวัง
                if hasattr(model, 'feature_names_in_'):
                    self.model_features[model_name] = list(model.feature_names_in_)
                    print(f"   ✅ {model_name} loaded - expects features: {list(model.feature_names_in_)}")
                else:
                    self.model_features[model_name] = None
                    print(f"   ✅ {model_name} loaded - unknown feature names")
                
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {e}")
        
        print(f"✅ Total models loaded: {len(self.models)}")
        return self.models

def align_features_with_model(X_data, model, model_name, available_features):
    """Align features to match what the model expects"""
    try:
        # ถ้าโมเดลมี feature_names_in_ ใช้ features นั้น
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            print(f"   🔍 {model_name} expects: {expected_features}")
            
            # ตรวจสอบว่า features ที่ต้องการมีอยู่ในข้อมูลหรือไม่
            missing_features = set(expected_features) - set(available_features)
            if missing_features:
                print(f"   ⚠️  Missing features: {missing_features}")
                # สร้าง features ที่ขาดหาย (ใช้ค่าเฉลี่ยหรือศูนย์)
                for feature in missing_features:
                    if feature not in X_data.columns:
                        X_data[feature] = 0  # หรือใช้ค่าที่เหมาะสมอื่นๆ
                        print(f"   🔧 Created missing feature: {feature}")
            
            # เลือกและเรียงลำดับ features ตามที่โมเดลต้องการ
            X_aligned = X_data[expected_features].copy()
            
        else:
            # ถ้าโมเดลไม่มี feature_names_in_ ใช้ features ที่มีอยู่
            print(f"   🔍 {model_name} has no feature names, using available features")
            X_aligned = X_data[available_features].copy()
        
        # ตรวจสอบและแปลง data types
        for col in X_aligned.columns:
            if X_aligned[col].dtype == 'object':
                print(f"   🔧 Converting {col} from {X_aligned[col].dtype} to numeric")
                X_aligned[col] = pd.to_numeric(X_aligned[col], errors='coerce')
                X_aligned[col].fillna(0, inplace=True)
        
        # ตรวจสอบ NaN
        if X_aligned.isnull().any().any():
            print(f"   🔧 Filling NaN values")
            X_aligned = X_aligned.fillna(0)
        
        print(f"   ✅ Final shape for {model_name}: {X_aligned.shape}")
        return X_aligned
        
    except Exception as e:
        print(f"   ❌ Error aligning features for {model_name}: {e}")
        return None

def prepare_forecast_data(predictor, forecast_date='2025-06-05'):
    """Prepare forecast data for specific date"""
    
    forecast_dt = pd.to_datetime(forecast_date)
    
    # Get data for forecast date
    forecast_data = predictor.df[predictor.df['datetime'].dt.date == forecast_dt.date()].copy()
    
    if len(forecast_data) == 0:
        raise ValueError(f"❌ No data found for forecast date {forecast_date}")
    
    print(f"📊 Forecast day data ({forecast_date}): {len(forecast_data)} hours")
    
    # Prepare basic features
    X_forecast = forecast_data[predictor.feature_columns].copy()
    
    # Check for NaN values and fill them
    for col in predictor.feature_columns:
        if X_forecast[col].isna().any():
            print(f"⚠️  Filling NaN values in {col} with mean")
            X_forecast[col].fillna(predictor.df[col].mean(), inplace=True)
    
    return forecast_data, X_forecast

def forecast_with_models(predictor, X_forecast):
    """Forecast using all loaded models with feature alignment"""
    
    predictions = {}
    
    for model_name, model in predictor.models.items():
        print(f"🎯 Predicting with {model_name}...")
        
        try:
            # จัด alignment features ให้ตรงกับโมเดล
            X_aligned = align_features_with_model(
                X_forecast, model, model_name, predictor.feature_columns
            )
            
            if X_aligned is None:
                print(f"   ❌ Feature alignment failed for {model_name}")
                predictions[model_name] = None
                continue
            
            # Make predictions
            y_pred = model.predict(X_aligned)
            
            # Ensure no negative predictions
            y_pred = np.maximum(0, y_pred)
            
            predictions[model_name] = y_pred
            print(f"   ✅ {model_name} prediction successful - {len(y_pred)} predictions")
            
        except Exception as e:
            print(f"   ❌ {model_name} prediction failed: {e}")
            predictions[model_name] = None
    
    return predictions

def evaluate_models(actual_values, predictions, model_names):
    """Evaluate model performance"""
    
    metrics = {}
    
    for model_name in model_names:
        if model_name in predictions and predictions[model_name] is not None:
            y_true = actual_values
            y_pred = predictions[model_name]
            
            # ตรวจสอบว่ามีข้อมูลพอที่จะประเมิน
            if len(y_true) != len(y_pred):
                print(f"   ⚠️  Length mismatch for {model_name}: true={len(y_true)}, pred={len(y_pred)}")
                continue
            
            # Calculate metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            
            metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse, 
                'R2': r2,
                'MAPE': mape,
                'Avg_Prediction': np.mean(y_pred),
                'Max_Prediction': np.max(y_pred)
            }
    
    return metrics

def plot_comparison(forecast_data, predictions, metrics, forecast_date):
    """Plot comparison between actual and predicted values"""
    
    plt.figure(figsize=(15, 10))
    
    # Plot actual values
    hours = range(len(forecast_data))
    actual_power = forecast_data['current_power'].values
    
    plt.plot(hours, actual_power, label='Actual Power', 
             linewidth=3, color='black', marker='o', markersize=6)
    
    # Colors for different models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    # Plot predictions for each model
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        if y_pred is not None and len(y_pred) == len(actual_power):
            color = colors[i % len(colors)]
            model_metrics = metrics.get(model_name, {})
            mae = model_metrics.get('MAE', 0)
            
            plt.plot(hours, y_pred, 
                     label=f'{model_name} (MAE: {mae:.0f}W)', 
                     linewidth=2, color=color, marker='s', markersize=4, 
                     linestyle='--', alpha=0.8)
    
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Power (Watts)', fontsize=14)
    plt.title(f'Solar Power Forecast vs Actual - {forecast_date}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(hours, [f'{h:02d}:00' for h in hours], rotation=45)
    plt.tight_layout()
    plt.show()

def print_model_ranking(metrics):
    """Print model ranking"""
    
    if not metrics:
        print("❌ No metrics available for ranking")
        return None
    
    print("\n🏆 MODEL PERFORMANCE RANKING")
    print("="*80)
    
    # Rank by MAE (Lower is better)
    print("\n📊 Ranking by MAE (Mean Absolute Error) - Lower is better:")
    print("-"*80)
    ranked_mae = sorted(metrics.items(), key=lambda x: x[1]['MAE'])
    for i, (model_name, metric) in enumerate(ranked_mae):
        print(f"{i+1}. {model_name}: {metric['MAE']:.2f} Watts")
    
    # Rank by R² (Higher is better)
    print("\n📊 Ranking by R² (R-squared) - Higher is better:")
    print("-"*80)
    ranked_r2 = sorted(metrics.items(), key=lambda x: x[1]['R2'], reverse=True)
    for i, (model_name, metric) in enumerate(ranked_r2):
        print(f"{i+1}. {model_name}: {metric['R2']:.4f}")
    
    # Best model overall
    if ranked_mae:
        best_model = ranked_mae[0][0]
        best_mae = ranked_mae[0][1]['MAE']
        best_r2 = ranked_r2[0][1]['R2'] if ranked_r2 else 0
        
        print(f"\n🎯 BEST OVERALL MODEL: {best_model}")
        print(f"   MAE: {best_mae:.2f} Watts")
        print(f"   R²: {best_r2:.4f}")
        if 'MAPE' in ranked_mae[0][1]:
            print(f"   MAPE: {ranked_mae[0][1]['MAPE']:.2f}%")
        
        return best_model
    
    return None

def print_model_ranking(metrics):
    """Print detailed model ranking with multiple metrics"""
    
    if not metrics:
        print("❌ No metrics available for ranking")
        return None
    
    print("\n🏆 DETAILED MODEL PERFORMANCE RANKING")
    print("="*100)
    
    # Rank by MAE (Lower is better)
    print("\n🥇 Ranking by MAE (Mean Absolute Error) - Lower is better:")
    print("-"*100)
    ranked_mae = sorted(metrics.items(), key=lambda x: x[1]['MAE'])
    for i, (model_name, metric) in enumerate(ranked_mae):
        print(f"{i+1}. {model_name}:")
        print(f"   MAE: {metric['MAE']:.2f} Watts")
        print(f"   R²: {metric['R2']:.4f}")
        if 'MAPE' in metric:
            print(f"   MAPE: {metric['MAPE']:.2f}%")
        print(f"   Avg Prediction: {metric['Avg_Prediction']:.2f}W")
        print()
    
    # Rank by R² (Higher is better)
    print("\n🥈 Ranking by R² (R-squared) - Higher is better:")
    print("-"*100)
    ranked_r2 = sorted(metrics.items(), key=lambda x: x[1]['R2'], reverse=True)
    for i, (model_name, metric) in enumerate(ranked_r2):
        print(f"{i+1}. {model_name}: R² = {metric['R2']:.4f}, MAE = {metric['MAE']:.2f}W")
    
    # Rank by MAPE (Lower is better)
    print("\n🥉 Ranking by MAPE (Mean Absolute Percentage Error) - Lower is better:")
    print("-"*100)
    ranked_mape = sorted(metrics.items(), key=lambda x: x[1]['MAPE'])
    for i, (model_name, metric) in enumerate(ranked_mape):
        print(f"{i+1}. {model_name}: MAPE = {metric['MAPE']:.2f}%, MAE = {metric['MAE']:.2f}W")
    
    # Overall ranking using composite score
    print("\n🏅 Overall Composite Ranking (Weighted Score):")
    print("-"*100)
    
    # Calculate composite scores (weighted average of normalized metrics)
    composite_scores = {}
    for model_name, metric in metrics.items():
        # Normalize metrics (0-1 scale, where 1 is best)
        # MAE: lower is better -> inverse normalization
        max_mae = max(m['MAE'] for m in metrics.values())
        min_mae = min(m['MAE'] for m in metrics.values())
        norm_mae = 1 - ((metric['MAE'] - min_mae) / (max_mae - min_mae)) if max_mae != min_mae else 1
        
        # R²: higher is better -> direct normalization
        max_r2 = max(m['R2'] for m in metrics.values())
        min_r2 = min(m['R2'] for m in metrics.values())
        norm_r2 = (metric['R2'] - min_r2) / (max_r2 - min_r2) if max_r2 != min_r2 else 1
        
        # MAPE: lower is better -> inverse normalization
        max_mape = max(m['MAPE'] for m in metrics.values())
        min_mape = min(m['MAPE'] for m in metrics.values())
        norm_mape = 1 - ((metric['MAPE'] - min_mape) / (max_mape - min_mape)) if max_mape != min_mape else 1
        
        # Weighted composite score (adjust weights as needed)
        composite_score = (0.4 * norm_mae) + (0.4 * norm_r2) + (0.2 * norm_mape)
        composite_scores[model_name] = composite_score
    
    # Rank by composite score
    ranked_composite = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_name, score) in enumerate(ranked_composite):
        metric = metrics[model_name]
        print(f"{i+1}. {model_name}:")
        print(f"   Composite Score: {score:.3f}")
        print(f"   MAE: {metric['MAE']:.2f} Watts (Weight: 40%)")
        print(f"   R²: {metric['R2']:.4f} (Weight: 40%)")
        print(f"   MAPE: {metric['MAPE']:.2f}% (Weight: 20%)")
        print(f"   Avg Power: {metric['Avg_Prediction']:.0f}W")
        print()
    
    # Best model overall
    if ranked_composite:
        best_model = ranked_composite[0][0]
        best_score = ranked_composite[0][1]
        best_metric = metrics[best_model]
        
        print(f"🎯 BEST OVERALL MODEL: {best_model}")
        print(f"   Composite Score: {best_score:.3f}")
        print(f"   MAE: {best_metric['MAE']:.2f} Watts")
        print(f"   R²: {best_metric['R2']:.4f}")
        print(f"   MAPE: {best_metric['MAPE']:.2f}%")
        print(f"   Average Predicted Power: {best_metric['Avg_Prediction']:.0f}W")
        
        return best_model
    
    return None

def print_daily_summary(forecast_data, predictions, metrics):
    """Print daily power generation summary"""
    print("\n📊 DAILY POWER GENERATION SUMMARY")
    print("="*80)
    
    actual_total = forecast_data['current_power'].sum()
    print(f"🔋 Actual Total Generation: {actual_total:.0f} Watt-hours")
    print(f"📈 Actual Average Power: {forecast_data['current_power'].mean():.0f}W")
    print(f"⏰ Generation Hours: {len(forecast_data)} hours")
    print()
    
    print("📈 PREDICTED DAILY GENERATION:")
    print("-"*80)
    
    summary_data = []
    for model_name, metric in metrics.items():
        if model_name in predictions and predictions[model_name] is not None:
            pred_total = predictions[model_name].sum()
            accuracy = 100 - abs((pred_total - actual_total) / actual_total * 100)
            summary_data.append({
                'Model': model_name,
                'Predicted_Total': pred_total,
                'Accuracy': accuracy,
                'MAE': metric['MAE'],
                'R2': metric['R2']
            })
    
    # Sort by accuracy (highest first)
    summary_data.sort(key=lambda x: x['Accuracy'], reverse=True)
    
    for i, data in enumerate(summary_data):
        print(f"{i+1}. {data['Model']}:")
        print(f"   Predicted Total: {data['Predicted_Total']:.0f} Watt-hours")
        print(f"   Accuracy: {data['Accuracy']:.1f}%")
        print(f"   MAE: {data['MAE']:.2f}W, R²: {data['R2']:.4f}")
        print()

# Main execution
if __name__ == "__main__":
    # Configuration
    CSV_PATH = "src/non-tf/select_lag/processed_data/solar_data_simple_fill.csv"
    MODELS_FOLDER = "src/non-tf/select_lag/saved_models"
    FORECAST_DATE = '2025-06-05'
    
    print("🔮 SOLAR POWER FORECASTING WITH FEATURE ALIGNMENT")
    print("="*60)
    
    try:
        # 1. โหลดข้อมูล
        print("📁 Step 1: Loading data...")
        predictor = CSVSolarPredictor(target_column='current_power')
        predictor.load_data(CSV_PATH)
        
        # 2. เตรียม features (numerical only)
        print("🔧 Step 2: Preparing numerical features...")
        predictor.prepare_features()
        
        # 3. โหลดโมเดล
        print("🤖 Step 3: Loading models...")
        predictor.load_models_from_folder(MODELS_FOLDER)
        
        if not predictor.models:
            print("❌ No models loaded. Cannot proceed.")
            exit()
        
        # 4. ตรวจสอบข้อมูลสำหรับวันที่ต้องการ
        forecast_day_data = predictor.df[predictor.df['datetime'].dt.date == pd.to_datetime(FORECAST_DATE).date()]
        
        if len(forecast_day_data) == 0:
            print(f"❌ No data found for forecast date {FORECAST_DATE}")
            exit()
        
        print(f"✅ Found {len(forecast_day_data)} records for {FORECAST_DATE}")
        
        # 5. เตรียมข้อมูลสำหรับการทำนาย
        print("📊 Step 4: Preparing forecast data...")
        forecast_data, X_forecast = prepare_forecast_data(predictor, FORECAST_DATE)
        
        # 6. ทำนายด้วยโมเดลทั้งหมด (พร้อมจัด alignment features)
        print("🔮 Step 5: Making predictions with feature alignment...")
        predictions = forecast_with_models(predictor, X_forecast)
        
        # 7. ประเมินผลโมเดล
        print("📈 Step 6: Evaluating models...")
        actual_values = forecast_data['current_power'].values
        metrics = evaluate_models(actual_values, predictions, list(predictor.models.keys()))
        
        if not metrics:
            print("❌ No successful predictions to evaluate")
            exit()
        
        # 8. พล็อตกราฟ
        print("📊 Step 7: Plotting results...")
        plot_comparison(forecast_data, predictions, metrics, FORECAST_DATE)
        

        # # 9. แสดงผลการจัดอันดับแบบละเอียด
        # print("🏆 Step 7: Detailed model ranking...")
        # best_model = print_model_ranking(metrics)

        # 10. แสดงสรุปรายวัน
        print_daily_summary(forecast_data, predictions, metrics)

        # # 11. พล็อตกราฟ
        # print("📊 Step 8: Plotting results...")
        # plot_comparison(forecast_data, predictions, metrics, FORECAST_DATE)
        # # 9. แสดงผลการจัดอันดับ
        # best_model = print_model_ranking(metrics)
        
        # # 10. บันทึกผลลัพธ์
        # print("💾 Step 8: Saving results...")
        # results_df = forecast_data[['datetime', 'current_power']].copy()
        # for model_name, y_pred in predictions.items():
        #     if y_pred is not None and len(y_pred) == len(results_df):
        #         results_df[f'Predicted_{model_name}'] = y_pred
        
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # results_file = f"forecast_results_{FORECAST_DATE}_{timestamp}.csv"
        # results_df.to_csv(results_file, index=False)
        # print(f"✅ Results saved to: {results_file}")
        
        # if best_model:
        #     print(f"\n🎉 Recommendation: Use '{best_model}' for future predictions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()