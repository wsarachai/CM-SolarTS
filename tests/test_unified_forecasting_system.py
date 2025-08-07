"""
Test suite for the unified forecasting system.

This module contains comprehensive tests for the unified forecasting system,
including tests for BaseTrainer implementations, ModelFactory, and ForecastingSystem.
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
import numpy as np
import logging

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from base_trainer import BaseTrainer
    from svm_trainer import SVMTrainer
    from tensorflow_trainer import TensorFlowTrainer
    from adaboost_trainer import AdaBoostTrainer
    from model_factory import ModelFactory
    from forecasting_system import ForecastingSystem
    from window_generator import WindowGenerator
    from data_loader import DataLoader
except ImportError:
    # Fallback to src module imports
    from src.base_trainer import BaseTrainer
    from src.svm_trainer import SVMTrainer
    from src.tensorflow_trainer import TensorFlowTrainer
    from src.adaboost_trainer import AdaBoostTrainer
    from src.model_factory import ModelFactory
    from src.forecasting_system import ForecastingSystem
    from src.window_generator import WindowGenerator
    from src.data_loader import DataLoader

# Disable TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class TestBaseTrainerInterface(unittest.TestCase):
    """Test the BaseTrainer interface compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple mock window generator for testing
        data_loader = DataLoader()
        data_loader.download_and_load()
        data_loader.preprocess()
        train_df, val_df, test_df = data_loader.split_and_normalize()
        
        self.window_generator = WindowGenerator(
            input_width=24,
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
    
    def test_svm_trainer_interface(self):
        """Test SVMTrainer implements BaseTrainer interface correctly."""
        trainer = SVMTrainer(
            self.window_generator,
            dev_mode=True,
            checkpoint_path=os.path.join(self.temp_dir, 'svm_test.pkl')
        )
        
        # Test interface methods exist
        self.assertTrue(hasattr(trainer, 'fit'))
        self.assertTrue(hasattr(trainer, 'predict'))
        self.assertTrue(hasattr(trainer, 'score'))
        self.assertTrue(hasattr(trainer, 'save_model'))
        self.assertTrue(hasattr(trainer, 'load_model'))
        self.assertTrue(hasattr(trainer, 'get_model_info'))
        
        # Test model info before training
        info = trainer.get_model_info()
        self.assertEqual(info['model_type'], 'svm')
        self.assertFalse(info['trained'])
        
        # Test training
        results = trainer.fit(cv_folds=2)
        self.assertIsInstance(results, dict)
        self.assertIn('best_score', results)
        
        # Test prediction
        predictions = trainer.predict()
        self.assertIsInstance(predictions, np.ndarray)
        
        # Test scoring
        scores = trainer.score()
        self.assertIsInstance(scores, dict)
        self.assertIn('rmse', scores)
        self.assertIn('mae', scores)
        self.assertIn('r2', scores)
        
        # Test model saving
        save_path = trainer.save_model()
        self.assertTrue(os.path.exists(save_path))
    
    def test_tensorflow_trainer_interface(self):
        """Test TensorFlowTrainer implements BaseTrainer interface correctly."""
        trainer = TensorFlowTrainer(
            self.window_generator,
            dev_mode=True,
            max_epochs=2,
            checkpoint_path=os.path.join(self.temp_dir, 'tf_test.h5')
        )
        
        # Test interface methods exist
        self.assertTrue(hasattr(trainer, 'fit'))
        self.assertTrue(hasattr(trainer, 'predict'))
        self.assertTrue(hasattr(trainer, 'score'))
        self.assertTrue(hasattr(trainer, 'save_model'))
        self.assertTrue(hasattr(trainer, 'load_model'))
        self.assertTrue(hasattr(trainer, 'get_model_info'))
        
        # Test model info before training
        info = trainer.get_model_info()
        self.assertEqual(info['model_type'], 'tensorflow')
        self.assertFalse(info['trained'])
        
        # Test training
        results = trainer.fit()
        self.assertIsInstance(results, dict)
        self.assertIn('best_epoch', results)
        self.assertIn('best_val_loss', results)
        
        # Test prediction
        predictions = trainer.predict()
        self.assertIsInstance(predictions, np.ndarray)
        
        # Test scoring
        scores = trainer.score()
        self.assertIsInstance(scores, dict)
        self.assertIn('rmse', scores)
        self.assertIn('mae', scores)
        
        # Test model saving
        save_path = trainer.save_model()
        self.assertTrue(os.path.exists(save_path))
    
    def test_adaboost_trainer_interface(self):
        """Test AdaBoostTrainer implements BaseTrainer interface correctly."""
        trainer = AdaBoostTrainer(
            self.window_generator,
            dev_mode=True,
            n_estimators=5,
            checkpoint_path=os.path.join(self.temp_dir, 'ada_test.pkl')
        )
        
        # Test interface methods exist
        self.assertTrue(hasattr(trainer, 'fit'))
        self.assertTrue(hasattr(trainer, 'predict'))
        self.assertTrue(hasattr(trainer, 'score'))
        self.assertTrue(hasattr(trainer, 'save_model'))
        self.assertTrue(hasattr(trainer, 'load_model'))
        self.assertTrue(hasattr(trainer, 'get_model_info'))
        
        # Test model info before training
        info = trainer.get_model_info()
        self.assertEqual(info['model_type'], 'adaboost')
        self.assertFalse(info['trained'])
        
        # Test training
        results = trainer.fit(n_splits=2)
        self.assertIsInstance(results, dict)
        self.assertIn('cv_mean', results)
        self.assertIn('cv_std', results)
        
        # Test prediction
        predictions = trainer.predict()
        self.assertIsInstance(predictions, np.ndarray)
        
        # Test scoring
        scores = trainer.score()
        self.assertIsInstance(scores, dict)
        self.assertIn('rmse', scores)
        self.assertIn('mae', scores)
        self.assertIn('r2', scores)
        
        # Test model saving
        save_path = trainer.save_model()
        self.assertTrue(os.path.exists(save_path))


class TestModelFactory(unittest.TestCase):
    """Test the ModelFactory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple mock window generator for testing
        data_loader = DataLoader()
        data_loader.download_and_load()
        data_loader.preprocess()
        train_df, val_df, test_df = data_loader.split_and_normalize()
        
        self.window_generator = WindowGenerator(
            input_width=24,
            label_width=1,
            shift=1,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=['Current Power']
        )
    
    def test_get_available_models(self):
        """Test getting available model types."""
        models = ModelFactory.get_available_models()
        self.assertIn('svm', models)
        self.assertIn('tensorflow', models)
        self.assertIn('adaboost', models)
    
    def test_create_svm_trainer(self):
        """Test creating SVM trainer through factory."""
        trainer = ModelFactory.create_trainer('svm', self.window_generator)
        self.assertIsInstance(trainer, SVMTrainer)
        self.assertIsInstance(trainer, BaseTrainer)
    
    def test_create_tensorflow_trainer(self):
        """Test creating TensorFlow trainer through factory."""
        trainer = ModelFactory.create_trainer('tensorflow', self.window_generator)
        self.assertIsInstance(trainer, TensorFlowTrainer)
        self.assertIsInstance(trainer, BaseTrainer)
    
    def test_create_adaboost_trainer(self):
        """Test creating AdaBoost trainer through factory."""
        trainer = ModelFactory.create_trainer('adaboost', self.window_generator)
        self.assertIsInstance(trainer, AdaBoostTrainer)
        self.assertIsInstance(trainer, BaseTrainer)
    
    def test_create_trainer_with_config(self):
        """Test creating trainer with custom configuration."""
        config = {'dev_mode': True, 'n_estimators': 10}
        trainer = ModelFactory.create_trainer('adaboost', self.window_generator, config)
        self.assertTrue(trainer.dev_mode)
        self.assertEqual(trainer.n_estimators, 10)
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with self.assertRaises(ValueError):
            ModelFactory.create_trainer('invalid_model', self.window_generator)
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = ModelFactory.get_default_config('svm')
        self.assertIsInstance(config, dict)
        self.assertIn('kernel', config)
        self.assertIn('C', config)


class TestForecastingSystem(unittest.TestCase):
    """Test the ForecastingSystem orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test configuration
        self.test_config = {
            'system': {
                'checkpoint_dir': self.temp_dir
            },
            'models': {
                'svm': {
                    'enabled': True,
                    'config': {'dev_mode': True}
                },
                'tensorflow': {
                    'enabled': True,
                    'config': {'dev_mode': True, 'max_epochs': 2}
                },
                'adaboost': {
                    'enabled': True,
                    'config': {'dev_mode': True, 'n_estimators': 5}
                }
            }
        }
        
        self.forecasting_system = ForecastingSystem()
        self.forecasting_system.config = self.test_config
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialize_data(self):
        """Test data initialization."""
        self.forecasting_system.initialize_data()
        self.assertIsNotNone(self.forecasting_system.data_loader)
        self.assertIsNotNone(self.forecasting_system.window_generator)
    
    def test_train_model(self):
        """Test model training through system."""
        self.forecasting_system.initialize_data()
        
        # Test training SVM
        results = self.forecasting_system.train_model('svm')
        self.assertIsInstance(results, dict)
        self.assertIn('svm', self.forecasting_system.trainers)
        self.assertTrue(self.forecasting_system.models_trained['svm'])
    
    def test_evaluate_model(self):
        """Test model evaluation through system."""
        self.forecasting_system.initialize_data()
        self.forecasting_system.train_model('svm')
        
        evaluation = self.forecasting_system.evaluate_model('svm')
        self.assertIsInstance(evaluation, dict)
        self.assertIn('rmse', evaluation)
    
    def test_compare_models(self):
        """Test model comparison."""
        self.forecasting_system.initialize_data()
        
        # Train multiple models
        self.forecasting_system.train_model('svm')
        self.forecasting_system.train_model('adaboost')
        
        comparison = self.forecasting_system.compare_models(['svm', 'adaboost'])
        self.assertIsInstance(comparison, dict)
        self.assertIn('svm', comparison)
        self.assertIn('adaboost', comparison)
    
    def test_get_status(self):
        """Test system status reporting."""
        status = self.forecasting_system.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn('data_initialized', status)
        self.assertIn('available_models', status)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBaseTrainerInterface))
    test_suite.addTest(unittest.makeSuite(TestModelFactory))
    test_suite.addTest(unittest.makeSuite(TestForecastingSystem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)