"""
Simple standalone test to verify core functionality without import issues.
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

def test_imports():
    """Test that we can import all modules."""
    print("Testing imports...")
    
    try:
        import base_trainer
        print("[OK] BaseTrainer imported successfully")
    except Exception as e:
        print(f"[FAIL] BaseTrainer import failed: {e}")
        return False
    
    try:
        import svm_trainer
        print("[OK] SVMTrainer imported successfully")
    except Exception as e:
        print(f"[FAIL] SVMTrainer import failed: {e}")
        return False
    
    try:
        import tensorflow_trainer
        print("[OK] TensorFlowTrainer imported successfully")
    except Exception as e:
        print(f"[FAIL] TensorFlowTrainer import failed: {e}")
        return False
    
    try:
        import adaboost_trainer
        print("[OK] AdaBoostTrainer imported successfully")
    except Exception as e:
        print(f"[FAIL] AdaBoostTrainer import failed: {e}")
        return False
    
    try:
        import model_factory
        print("[OK] ModelFactory imported successfully")
    except Exception as e:
        print(f"[FAIL] ModelFactory import failed: {e}")
        return False
    
    try:
        import window_generator
        print("[OK] WindowGenerator imported successfully")
    except Exception as e:
        print(f"[FAIL] WindowGenerator import failed: {e}")
        return False
    
    try:
        import data_loader
        print("[OK] DataLoader imported successfully")
    except Exception as e:
        print(f"[FAIL] DataLoader import failed: {e}")
        return False
    
    return True

def create_mock_data():
    """Create minimal synthetic data for testing."""
    print("Creating mock data...")
    
    # Create 50 samples of synthetic data
    np.random.seed(42)
    n_samples = 50
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    
    # Generate synthetic features
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
    
    print(f"[OK] Created mock data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    return train_df, val_df, test_df

def test_window_generator(train_df, val_df, test_df):
    """Test WindowGenerator functionality."""
    print("Testing WindowGenerator...")
    
    try:
        import window_generator
        
        wg = window_generator.WindowGenerator(
            input_width=4,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['Current Power']
        )
        
        # Test that we can get datasets
        train_ds = wg.train
        val_ds = wg.val
        test_ds = wg.test
        
        print("[OK] WindowGenerator created successfully")
        print(f"[OK] Train dataset: {train_ds}")
        print(f"[OK] Val dataset: {val_ds}")
        print(f"[OK] Test dataset: {test_ds}")
        
        return wg
        
    except Exception as e:
        print(f"[FAIL] WindowGenerator test failed: {e}")
        return None

def test_model_factory(window_generator):
    """Test ModelFactory functionality."""
    print("Testing ModelFactory...")
    
    try:
        import model_factory
        
        # Test getting available models
        models = model_factory.ModelFactory.get_available_models()
        print(f"[OK] Available models: {models}")
        
        # Create temp directory for checkpoints
        temp_dir = tempfile.mkdtemp()
        
        # Test creating SVM trainer
        svm_trainer = model_factory.ModelFactory.create_trainer(
            'svm',
            window_generator,
            {'dev_mode': True, 'checkpoint_path': os.path.join(temp_dir, 'temp_svm.pkl')}
        )
        print("[OK] SVM trainer created successfully")
        
        # Test creating TensorFlow trainer
        tf_trainer = model_factory.ModelFactory.create_trainer(
            'tensorflow',
            window_generator,
            {'dev_mode': True, 'max_epochs': 1, 'checkpoint_path': os.path.join(temp_dir, 'temp_tf.h5')}
        )
        print("[OK] TensorFlow trainer created successfully")
        
        # Test creating AdaBoost trainer
        ada_trainer = model_factory.ModelFactory.create_trainer(
            'adaboost',
            window_generator,
            {'dev_mode': True, 'n_estimators': 2, 'checkpoint_path': os.path.join(temp_dir, 'temp_ada.pkl')}
        )
        print("[OK] AdaBoost trainer created successfully")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return svm_trainer, tf_trainer, ada_trainer
        
    except Exception as e:
        print(f"[FAIL] ModelFactory test failed: {e}")
        return None, None, None

def test_trainer_interfaces(trainers):
    """Test that all trainers implement the BaseTrainer interface."""
    print("Testing trainer interfaces...")
    
    svm_trainer, tf_trainer, ada_trainer = trainers
    
    for name, trainer in [('SVM', svm_trainer), ('TensorFlow', tf_trainer), ('AdaBoost', ada_trainer)]:
        if trainer is None:
            continue
            
        try:
            # Test interface methods exist
            assert hasattr(trainer, 'fit'), f"{name} missing fit method"
            assert hasattr(trainer, 'predict'), f"{name} missing predict method"
            assert hasattr(trainer, 'score'), f"{name} missing score method"
            assert hasattr(trainer, 'save_model'), f"{name} missing save_model method"
            assert hasattr(trainer, 'load_model'), f"{name} missing load_model method"
            assert hasattr(trainer, 'get_model_info'), f"{name} missing get_model_info method"
            
            # Test get_model_info
            info = trainer.get_model_info()
            assert isinstance(info, dict), f"{name} get_model_info should return dict"
            assert 'model_type' in info, f"{name} missing model_type in info"
            assert 'trained' in info, f"{name} missing trained in info"
            
            print(f"[OK] {name} trainer interface test passed")
            
        except Exception as e:
            print(f"[FAIL] {name} trainer interface test failed: {e}")

def main():
    """Run all tests."""
    print("=" * 50)
    print("UNIFIED FORECASTING SYSTEM - SIMPLE TEST")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("Import tests failed. Stopping.")
        return False
    
    print("\n" + "-" * 30)
    
    # Test 2: Mock data creation
    train_df, val_df, test_df = create_mock_data()
    
    print("\n" + "-" * 30)
    
    # Test 3: WindowGenerator
    wg = test_window_generator(train_df, val_df, test_df)
    if wg is None:
        print("WindowGenerator test failed. Stopping.")
        return False
    
    print("\n" + "-" * 30)
    
    # Test 4: ModelFactory
    trainers = test_model_factory(wg)
    if all(t is None for t in trainers):
        print("ModelFactory test failed. Stopping.")
        return False
    
    print("\n" + "-" * 30)
    
    # Test 5: Trainer interfaces
    test_trainer_interfaces(trainers)
    
    print("\n" + "=" * 50)
    print("SIMPLE TEST COMPLETED")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)