"""
Comprehensive training test for the unified forecasting system.
Tests actual model training, prediction, and evaluation with minimal dataset.
"""

import sys
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def create_mock_data():
    """Create minimal synthetic data for testing."""
    print("Creating mock data for training test...")
    
    # Create 100 samples of synthetic data
    np.random.seed(42)
    n_samples = 100
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    
    # Generate synthetic features with clear patterns
    time_of_day = np.sin(2 * np.pi * np.arange(n_samples) / 96)
    base_power = np.maximum(0, time_of_day + 0.1 * np.random.randn(n_samples))
    
    data = {
        'Grid Feed In': base_power * 0.8 + 0.1 * np.random.randn(n_samples),
        'External Energy Supply': 0.2 * np.random.randn(n_samples),
        'Internal Power Supply': base_power * 0.3 + 0.05 * np.random.randn(n_samples),
        'Current Power': base_power + 0.1 * np.random.randn(n_samples),
        'Self Consumption': base_power * 0.4 + 0.05 * np.random.randn(n_samples),
        'Ambient Temperature': 25 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 96) + 2 * np.random.randn(n_samples),
        'Module Temperature': 30 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / 96) + 3 * np.random.randn(n_samples),
        'Total Irradiation': np.maximum(0, time_of_day * 800 + 50 * np.random.randn(n_samples))
    }
    
    df = pd.DataFrame(data, index=timestamps)
    
    # Add cyclical features
    hour = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Split data
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Simple normalization
    for df_split in [train_df, val_df, test_df]:
        for col in df_split.select_dtypes(include=[np.number]).columns:
            mean_val = train_df[col].mean()
            std_val = train_df[col].std()
            if std_val > 0:
                df_split[col] = (df_split[col] - mean_val) / std_val
    
    print(f"[OK] Created training data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    return train_df, val_df, test_df

def test_svm_training():
    """Test SVM model training, prediction, and evaluation."""
    print("\n" + "="*50)
    print("TESTING SVM TRAINER")
    print("="*50)
    
    try:
        import window_generator
        import model_factory
        
        # Create data and window generator
        train_df, val_df, test_df = create_mock_data()
        wg = window_generator.WindowGenerator(
            input_width=6,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['Current Power']
        )
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Create SVM trainer
        svm_trainer = model_factory.ModelFactory.create_trainer(
            'svm', 
            wg, 
            {
                'dev_mode': True, 
                'checkpoint_path': os.path.join(temp_dir, 'svm_test.pkl'),
                'optimization_method': 'grid',
                'optimization_params': {
                    'C': [0.1, 1.0],
                    'epsilon': [0.1, 0.2]
                },
                'cv_folds': 2
            }
        )
        
        print("[OK] SVM trainer created")
        
        # Test training
        print("Training SVM model...")
        training_results = svm_trainer.fit()
        print(f"[OK] SVM training completed: {training_results}")
        
        # Test prediction
        print("Making predictions...")
        predictions = svm_trainer.predict()
        print(f"[OK] SVM predictions generated: shape {predictions.shape}")
        
        # Test scoring
        print("Evaluating model...")
        scores = svm_trainer.score()
        print(f"[OK] SVM evaluation: {scores}")
        
        # Test model saving
        print("Saving model...")
        save_path = svm_trainer.save_model()
        print(f"[OK] SVM model saved to: {save_path}")
        
        # Test model loading
        print("Loading model...")
        svm_trainer.load_model()
        print("[OK] SVM model loaded successfully")
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] SVM training test failed: {e}")
        return False

def test_tensorflow_training():
    """Test TensorFlow model training, prediction, and evaluation."""
    print("\n" + "="*50)
    print("TESTING TENSORFLOW TRAINER")
    print("="*50)
    
    try:
        import window_generator
        import model_factory
        
        # Create data and window generator
        train_df, val_df, test_df = create_mock_data()
        wg = window_generator.WindowGenerator(
            input_width=6,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['Current Power']
        )
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Create TensorFlow trainer
        tf_trainer = model_factory.ModelFactory.create_trainer(
            'tensorflow', 
            wg, 
            {
                'dev_mode': True, 
                'max_epochs': 3,
                'patience': 2,
                'checkpoint_path': os.path.join(temp_dir, 'tf_test.h5')
            }
        )
        
        print("[OK] TensorFlow trainer created")
        
        # Test training
        print("Training TensorFlow model...")
        training_results = tf_trainer.fit()
        print(f"[OK] TensorFlow training completed: {training_results}")
        
        # Test prediction
        print("Making predictions...")
        predictions = tf_trainer.predict()
        print(f"[OK] TensorFlow predictions generated: shape {predictions.shape}")
        
        # Test scoring
        print("Evaluating model...")
        scores = tf_trainer.score()
        print(f"[OK] TensorFlow evaluation: {scores}")
        
        # Test model saving
        print("Saving model...")
        save_path = tf_trainer.save_model()
        print(f"[OK] TensorFlow model saved to: {save_path}")
        
        # Test model loading
        print("Loading model...")
        tf_trainer.load_model()
        print("[OK] TensorFlow model loaded successfully")
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] TensorFlow training test failed: {e}")
        return False

def test_adaboost_training():
    """Test AdaBoost model training, prediction, and evaluation."""
    print("\n" + "="*50)
    print("TESTING ADABOOST TRAINER")
    print("="*50)
    
    try:
        import window_generator
        import model_factory
        
        # Create data and window generator
        train_df, val_df, test_df = create_mock_data()
        wg = window_generator.WindowGenerator(
            input_width=6,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['Current Power']
        )
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Create AdaBoost trainer
        ada_trainer = model_factory.ModelFactory.create_trainer(
            'adaboost', 
            wg, 
            {
                'dev_mode': True, 
                'n_estimators': 5,
                'learning_rate': 1.0,
                'checkpoint_path': os.path.join(temp_dir, 'ada_test.pkl')
            }
        )
        
        print("[OK] AdaBoost trainer created")
        
        # Test training
        print("Training AdaBoost model...")
        training_results = ada_trainer.fit(n_splits=2)
        print(f"[OK] AdaBoost training completed: {training_results}")
        
        # Test prediction
        print("Making predictions...")
        predictions = ada_trainer.predict()
        print(f"[OK] AdaBoost predictions generated: shape {predictions.shape}")
        
        # Test scoring
        print("Evaluating model...")
        scores = ada_trainer.score()
        print(f"[OK] AdaBoost evaluation: {scores}")
        
        # Test model saving
        print("Saving model...")
        save_path = ada_trainer.save_model()
        print(f"[OK] AdaBoost model saved to: {save_path}")
        
        # Test model loading
        print("Loading model...")
        ada_trainer.load_model()
        print("[OK] AdaBoost model loaded successfully")
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] AdaBoost training test failed: {e}")
        return False

def main():
    """Run comprehensive training tests."""
    print("=" * 60)
    print("UNIFIED FORECASTING SYSTEM - COMPREHENSIVE TRAINING TEST")
    print("=" * 60)
    
    results = []
    
    # Test SVM training
    results.append(("SVM", test_svm_training()))
    
    # Test TensorFlow training
    results.append(("TensorFlow", test_tensorflow_training()))
    
    # Test AdaBoost training
    results.append(("AdaBoost", test_adaboost_training()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for model_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {model_name} training test")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TRAINING TESTS PASSED SUCCESSFULLY!")
        print("The unified forecasting system is working correctly.")
    else:
        print("SOME TRAINING TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)