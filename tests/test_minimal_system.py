"""
Minimal test suite for the unified forecasting system with reduced dataset.

This module contains streamlined tests using a minimal dataset subset
for rapid debugging and validation logic verification.
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Disable TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import modules with fallback handling
try:
    import base_trainer
    import svm_trainer
    import tensorflow_trainer
    import adaboost_trainer
    import model_factory
    import forecasting_system
    import window_generator
    import data_loader
    
    BaseTrainer = base_trainer.BaseTrainer
    SVMTrainer = svm_trainer.SVMTrainer
    TensorFlowTrainer = tensorflow_trainer.TensorFlowTrainer
    AdaBoostTrainer = adaboost_trainer.AdaBoostTrainer
    ModelFactory = model_factory.ModelFactory
    ForecastingSystem = forecasting_system.ForecastingSystem
    WindowGenerator = window_generator.WindowGenerator
    DataLoader = data_loader.DataLoader
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class MockDataLoader:
    """Mock data loader that creates minimal synthetic data for testing."""
    
    def __init__(self):
        self.data = None
        
    def create_minimal_data(self):
        """Create minimal synthetic PV data for testing."""
        # Create 100 samples of synthetic data (much smaller than real dataset)
        np.random.seed(42)  # For reproducible tests
        
        # Generate synthetic time series data
        n_samples = 100
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
        
        # Create synthetic features that mimic real PV data patterns
        time_of_day = np.sin(2 * np.pi * np.arange(n_samples) / 96)  # Daily cycle
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
        
        self.data = pd.DataFrame(data, index=timestamps)
        return self.data
    
    def download_and_load(self):
        """Mock download - just create synthetic data."""
        self.create_minimal_data()
        
    def preprocess(self):
        """Mock preprocessing - add minimal cyclical features."""
        if self.data is None:
            self.create_minimal_data()
            
        # Add simple cyclical features
        hour = self.data.index.hour
        self.data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
    def split_and_normalize(self):
        """Split into train/val/test with minimal sizes."""
        if self.data is None:
            self.preprocess()
            
        # Very small splits for fast testing
        n = len(self.data)
        train_end = int(0.7 * n)  # 70 samples
        val_end = int(0.85 * n)   # 15 samples
        
        train_df = self.data.iloc[:train_end].copy()
        val_df = self.data.iloc[train_end:val_end].copy()
        test_df = self.data.iloc[val_end:].copy()
        
        # Simple normalization
        for df in [train_df, val_df, test_df]:
            for col in df.select_dtypes(include=[np.number]).columns:
                mean_val = train_df[col].mean()
                std_val = train_df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
                    
        return train_df, val_df, test_df


class TestMinimalSystem(unittest.TestCase):
    """Test the unified forecasting system with minimal data."""
    
    def setUp(self):
        """Set up test fixtures with minimal data."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal window generator for testing
        mock_loader = MockDataLoader()
        mock_loader.download_and_load()
        mock_loader.preprocess()
        train_df, val_df, test_df = mock_loader.split_and_normalize()
        
        self.window_generator = WindowGenerator(
            input_width=4,  # Much smaller window
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['Current Power']
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_base_trainer_interface(self):
        """Test that BaseTrainer is properly defined."""
        # Test that BaseTrainer exists and is abstract
        self.assertTrue(hasattr(BaseTrainer, 'fit'))
        self.assertTrue(hasattr(BaseTrainer, 'predict'))
        self.assertTrue(hasattr(BaseTrainer, 'score'))
        
        # Test that we can't instantiate BaseTrainer directly
        with self.assertRaises(TypeError):
            BaseTrainer(self.window_generator)
    
    def test_svm_trainer_minimal(self):
        """Test SVMTrainer with minimal configuration."""
        trainer = SVMTrainer(
            self.window_generator,
            dev_mode=True,
            optimization_method='grid',
            cv_folds=2,  # Minimal CV folds
            checkpoint_path=os.path.join(self.temp_dir, 'svm_test.pkl')
        )
        
        # Test model info before training
        info = trainer.get_model_info()
        self.assertEqual(info['model_type'], 'svm')
        self.assertFalse(info['trained'])
        
        # Test training with minimal parameters
        results = trainer.fit(cv_folds=2)
        self.assertIsInstance(results, dict)
        self.assertIn('best_score', results)
        
        # Test prediction
        predictions = trainer.predict()
        self.assertIsInstance(predictions, np.ndarray)
        self.assertGreater(len(predictions), 0)
        
        # Test scoring
        scores = trainer.score()
        self.assertIsInstance(scores, dict)
        self.assertIn('rmse', scores)
        self.assertIn('mae', scores)
        
        # Test model saving
        save_path = trainer.save_model()
        self.assertTrue(os.path.exists(save_path))
    
    def test_tensorflow_trainer_minimal(self):
        """Test TensorFlowTrainer with minimal configuration."""
        trainer = TensorFlowTrainer(
            self.window_generator,
            dev_mode=True,
            max_epochs=2,  # Very few epochs for fast testing
            patience=1,
            checkpoint_path=os.path.join(self.temp_dir, 'tf_test.h5')
        )
        
        # Test model info before training
        info = trainer.get_model_info()
        self.assertEqual(info['model_type'], 'tensorflow')
        self.assertFalse(info['trained'])
        
        # Test training
        results = trainer.fit()
        self.assertIsInstance(results, dict)
        self.assertIn('best_epoch', results)
        
        # Test prediction
        predictions = trainer.predict()
        self.assertIsInstance(predictions, np.ndarray)
        
        # Test scoring
        scores = trainer.score()
        self.assertIsInstance(scores, dict)
        self.assertIn('rmse', scores)
        
        # Test model saving
        save_path = trainer.save_model()
        self.assertTrue(os.path.exists(save_path))
    
    def test_adaboost_trainer_minimal(self):
        """Test AdaBoostTrainer with minimal configuration."""
        trainer = AdaBoostTrainer(
            self.window_generator,
            dev_mode=True,
            n_estimators=3,  # Very few estimators for fast testing
            checkpoint_path=os.path.join(self.temp_dir, 'ada_test.pkl')
        )
        
        # Test model info before training
        info = trainer.get_model_info()
        self.assertEqual(info['model_type'], 'adaboost')
        self.assertFalse(info['trained'])
        
        # Test training
        results = trainer.fit(n_splits=2)
        self.assertIsInstance(results, dict)
        self.assertIn('cv_mean', results)
        
        # Test prediction
        predictions = trainer.predict()
        self.assertIsInstance(predictions, np.ndarray)
        
        # Test scoring
        scores = trainer.score()
        self.assertIsInstance(scores, dict)
        self.assertIn('rmse', scores)
        
        # Test model saving
        save_path = trainer.save_model()
        self.assertTrue(os.path.exists(save_path))
    
    def test_model_factory_minimal(self):
        """Test ModelFactory with minimal configuration."""
        # Test getting available models
        models = ModelFactory.get_available_models()
        self.assertIn('svm', models)
        self.assertIn('tensorflow', models)
        self.assertIn('adaboost', models)
        
        # Test creating trainers
        svm_trainer = ModelFactory.create_trainer('svm', self.window_generator, {'dev_mode': True})
        self.assertIsInstance(svm_trainer, SVMTrainer)
        
        tf_trainer = ModelFactory.create_trainer('tensorflow', self.window_generator, {'dev_mode': True})
        self.assertIsInstance(tf_trainer, TensorFlowTrainer)
        
        ada_trainer = ModelFactory.create_trainer('adaboost', self.window_generator, {'dev_mode': True})
        self.assertIsInstance(ada_trainer, AdaBoostTrainer)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            ModelFactory.create_trainer('invalid_model', self.window_generator)


if __name__ == '__main__':
    # Set up minimal logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests with minimal verbosity
    unittest.main(verbosity=2)