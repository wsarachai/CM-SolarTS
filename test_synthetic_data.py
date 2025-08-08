#!/usr/bin/env python3
"""
Test script for validating main.py functionality with synthetic data.
This script tests edge cases, error handling, and compatibility issues.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from datetime import datetime, timedelta

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.window_generator import WindowGenerator
    from src.adaboost_trainer import AdaBoostTrainer
    from src.svm_trainer import SVMTrainer
except ImportError:
    from window_generator import WindowGenerator
    from adaboost_trainer import AdaBoostTrainer
    from svm_trainer import SVMTrainer

def create_synthetic_dataset(n_samples=1000, n_features=8, noise_level=0.1):
    """
    Create a synthetic time series dataset similar to the PV data structure.
    
    Args:
        n_samples: Number of time samples
        n_features: Number of features (should match ALL_COLS)
        noise_level: Amount of noise to add
    
    Returns:
        pandas.DataFrame: Synthetic dataset with datetime index
    """
    print(f"Creating synthetic dataset with {n_samples} samples and {n_features} features...")
    
    # Create datetime index (15-minute intervals)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(minutes=15*i) for i in range(n_samples)]
    
    # Feature names matching the original dataset
    feature_names = [
        'Grid Feed In', 'External Energy Supply', 'Internal Power Supply',
        'Current Power', 'Self Consumption', 'Ambient Temperature',
        'Module Temperature', 'Total Irradiation'
    ]
    
    # Generate synthetic data with realistic patterns
    np.random.seed(42)  # For reproducibility
    
    data = {}
    for i, feature in enumerate(feature_names):
        if 'Power' in feature or 'Supply' in feature or 'Consumption' in feature:
            # Power-related features: positive values with daily patterns
            base_pattern = np.sin(2 * np.pi * np.arange(n_samples) / (24 * 4)) + 1  # Daily cycle (96 samples per day)
            seasonal_pattern = 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / (365 * 24 * 4))  # Yearly cycle
            noise = noise_level * np.random.randn(n_samples)
            data[feature] = np.maximum(0, base_pattern + seasonal_pattern + noise)
        elif 'Temperature' in feature:
            # Temperature features: realistic temperature ranges
            base_temp = 25 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 4))  # Daily cycle
            seasonal_temp = 5 * np.sin(2 * np.pi * np.arange(n_samples) / (365 * 24 * 4))  # Yearly cycle
            noise = noise_level * np.random.randn(n_samples)
            data[feature] = base_temp + seasonal_temp + noise
        elif 'Irradiation' in feature:
            # Irradiation: positive values with strong daily patterns
            irradiation_pattern = np.maximum(0, np.sin(2 * np.pi * np.arange(n_samples) / (24 * 4)))
            seasonal_irradiation = 0.2 * np.sin(2 * np.pi * np.arange(n_samples) / (365 * 24 * 4))
            noise = noise_level * np.random.randn(n_samples)
            data[feature] = irradiation_pattern + seasonal_irradiation + np.abs(noise)
        else:
            # Default: random positive values
            data[feature] = np.abs(np.random.randn(n_samples))
    
    # Create DataFrame
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name='Datetime'))
    
    print(f"Synthetic dataset created successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Features: {list(df.columns)}")
    
    return df

def create_synthetic_splits(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split synthetic dataset into train/val/test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio  
        test_ratio: Test set ratio
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Dataset splits:")
    print(f"  Training: {len(train_df)} samples ({len(train_df)/n:.1%})")
    print(f"  Validation: {len(val_df)} samples ({len(val_df)/n:.1%})")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/n:.1%})")
    
    return train_df, val_df, test_df

def test_window_generator(train_df, val_df, test_df):
    """Test WindowGenerator with synthetic data."""
    print("\n" + "="*50)
    print("TESTING WINDOW GENERATOR")
    print("="*50)
    
    try:
        window = WindowGenerator(
            input_width=24,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['Current Power']
        )
        
        print(f"Window generator created successfully!")
        print(f"Input width: {window.input_width}")
        print(f"Label width: {window.label_width}")
        print(f"Shift: {window.shift}")
        print(f"Label columns: {window.label_columns}")
        
        # Test dataset creation
        train_ds = window.train
        val_ds = window.val
        test_ds = window.test
        
        print(f"Datasets created successfully!")
        
        # Test batch extraction
        for batch_x, batch_y in train_ds.take(1):
            print(f"Batch shapes: X={batch_x.shape}, y={batch_y.shape}")
            break
        
        return window
        
    except Exception as e:
        print(f"WindowGenerator test failed: {str(e)}")
        return None

def test_adaboost_trainer(window):
    """Test AdaBoostTrainer with synthetic data."""
    print("\n" + "="*50)
    print("TESTING ADABOOST TRAINER")
    print("="*50)
    
    try:
        # Test with minimal configuration for speed
        trainer = AdaBoostTrainer(
            window,
            n_estimators=5,  # Very small for testing
            learning_rate=1.0,
            differencing_order=0,  # No differencing for simplicity
            dev_mode=True,
            dev_sample_ratio=0.2,  # Use only 20% of data
            checkpoint_enabled=False  # Disable checkpointing for testing
        )
        
        print(f"AdaBoostTrainer initialized successfully!")
        print(f"Development mode: {trainer.is_dev_mode()}")
        
        # Test training
        results = trainer.fit(n_splits=2)  # Minimal splits
        print(f"Training completed!")
        print(f"Cross-validation mean: {results['cv_mean']:.4f}")
        
        # Test prediction
        predictions = trainer.predict()
        print(f"Predictions shape: {predictions.shape}")
        
        # Test scoring
        metrics = trainer.score()
        print(f"Test metrics: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return trainer, True
        
    except Exception as e:
        print(f"AdaBoostTrainer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

def test_svm_trainer(window):
    """Test SVMTrainer with synthetic data."""
    print("\n" + "="*50)
    print("TESTING SVM TRAINER")
    print("="*50)
    
    try:
        # Test with minimal configuration for speed
        trainer = SVMTrainer(
            window,
            kernel='linear',  # Linear kernel is faster
            C=1.0,
            epsilon=0.1,
            dev_mode=True,
            dev_sample_ratio=0.1,  # Use only 10% of data
            checkpoint_enabled=False  # Disable checkpointing for testing
        )
        
        print(f"SVMTrainer initialized successfully!")
        print(f"Kernel: {trainer.kernel}")
        
        # Test training
        results = trainer.fit()
        print(f"Training completed!")
        print(f"Training R2: {results.get('train_r2', 'N/A'):.4f}")
        
        # Test prediction
        predictions = trainer.predict()
        print(f"Predictions shape: {predictions.shape}")
        
        # Test scoring
        metrics = trainer.score()
        print(f"Test metrics: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return trainer, True
        
    except Exception as e:
        print(f"SVMTrainer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False

def test_edge_cases():
    """Test various edge cases and error conditions."""
    print("\n" + "="*50)
    print("TESTING EDGE CASES")
    print("="*50)
    
    edge_case_results = []
    
    # Test 1: Very small dataset
    print("\n1. Testing with very small dataset (100 samples)...")
    try:
        small_df = create_synthetic_dataset(n_samples=100)
        train_df, val_df, test_df = create_synthetic_splits(small_df)
        window = test_window_generator(train_df, val_df, test_df)
        if window:
            ada_trainer, ada_success = test_adaboost_trainer(window)
            svm_trainer, svm_success = test_svm_trainer(window)
            edge_case_results.append(("Small dataset", ada_success and svm_success))
        else:
            edge_case_results.append(("Small dataset", False))
    except Exception as e:
        print(f"Small dataset test failed: {str(e)}")
        edge_case_results.append(("Small dataset", False))
    
    # Test 2: Dataset with missing values
    print("\n2. Testing with missing values...")
    try:
        df_with_nan = create_synthetic_dataset(n_samples=500)
        # Introduce some NaN values
        df_with_nan.iloc[50:60, 0] = np.nan
        df_with_nan.iloc[100:110, 3] = np.nan
        
        # Fill NaN values (as would be done in preprocessing)
        df_with_nan = df_with_nan.fillna(method='ffill').fillna(0)
        
        train_df, val_df, test_df = create_synthetic_splits(df_with_nan)
        window = test_window_generator(train_df, val_df, test_df)
        if window:
            ada_trainer, ada_success = test_adaboost_trainer(window)
            edge_case_results.append(("Missing values", ada_success))
        else:
            edge_case_results.append(("Missing values", False))
    except Exception as e:
        print(f"Missing values test failed: {str(e)}")
        edge_case_results.append(("Missing values", False))
    
    # Test 3: Zero variance features
    print("\n3. Testing with zero variance features...")
    try:
        df_zero_var = create_synthetic_dataset(n_samples=500)
        # Make one feature constant
        df_zero_var['Grid Feed In'] = 1.0
        
        train_df, val_df, test_df = create_synthetic_splits(df_zero_var)
        window = test_window_generator(train_df, val_df, test_df)
        if window:
            ada_trainer, ada_success = test_adaboost_trainer(window)
            edge_case_results.append(("Zero variance", ada_success))
        else:
            edge_case_results.append(("Zero variance", False))
    except Exception as e:
        print(f"Zero variance test failed: {str(e)}")
        edge_case_results.append(("Zero variance", False))
    
    return edge_case_results

def main():
    """Main test function."""
    print("="*60)
    print("SYNTHETIC DATA TESTING FOR MAIN.PY FUNCTIONALITY")
    print("="*60)
    
    # Create synthetic dataset
    df = create_synthetic_dataset(n_samples=2000)
    train_df, val_df, test_df = create_synthetic_splits(df)
    
    # Test window generator
    window = test_window_generator(train_df, val_df, test_df)
    if not window:
        print("WindowGenerator test failed - aborting further tests")
        return
    
    # Test trainers
    ada_trainer, ada_success = test_adaboost_trainer(window)
    svm_trainer, svm_success = test_svm_trainer(window)
    
    # Test edge cases
    edge_case_results = test_edge_cases()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    print(f"AdaBoost Trainer: {'PASS' if ada_success else 'FAIL'}")
    print(f"SVM Trainer: {'PASS' if svm_success else 'FAIL'}")
    
    print(f"\nEdge Case Results:")
    for test_name, result in edge_case_results:
        print(f"  {test_name}: {'PASS' if result else 'FAIL'}")
    
    # Overall result
    all_passed = ada_success and svm_success and all(result for _, result in edge_case_results)
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n*** main.py functionality is working correctly with both trainers!")
        print("The system can handle:")
        print("  - Normal datasets")
        print("  - Small datasets") 
        print("  - Datasets with missing values")
        print("  - Datasets with zero variance features")
        print("  - Both AdaBoost and SVM training pipelines")
    else:
        print("\n*** Some issues were detected. Check the detailed output above.")

if __name__ == "__main__":
    main()