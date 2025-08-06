import unittest
import numpy as np
import pandas as pd
import warnings

from typing import List

# Import AdaBoostTrainer and WindowGenerator
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from adaboost_trainer import AdaBoostTrainer

try:
    from window_generator import WindowGenerator
    from data_loader import DataLoader
except ImportError:
    from src.window_generator import WindowGenerator
    from src.data_loader import DataLoader

class TestAdaBoostTrainer(unittest.TestCase):
    """Comprehensive test suite for AdaBoostTrainer."""

    @classmethod
    def setUpClass(cls):
        # Load real data using the same pipeline as main.py
        DATASET_HOST = 'https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/'
        DATASET_FILE = 'export_device_1_basic_aggregated_15minutes.csv.gz'
        ALL_COLS = [
            'Grid Feed In', 'External Energy Supply', 'Internal Power Supply',
            'Current Power', 'Self Consumption', 'Ambient Temperature',
            'Module Temperature', 'Total Irradiation'
        ]
        try:
            cls.data_loader = DataLoader(DATASET_HOST, DATASET_FILE, ALL_COLS)
            cls.df = cls.data_loader.download_and_load()
            cls.df = cls.data_loader.preprocess()
            cls.train_df, cls.val_df, cls.test_df = cls.data_loader.split_and_normalize()
            cls.window = WindowGenerator(
                input_width=24, label_width=1, shift=1,
                train_df=cls.train_df, val_df=cls.val_df, test_df=cls.test_df,
                label_columns=['Current Power']
            )
            cls.real_data_available = True
        except Exception as e:
            warnings.warn(f"Could not load real data: {e}")
            cls.real_data_available = False

    def test_training_and_convergence(self):
        """Test AdaBoostTrainer can fit and converge on real data."""
        if not self.real_data_available:
            self.skipTest("Real data not available")
        trainer = AdaBoostTrainer(self.window, n_estimators=10, learning_rate=0.5)
        trainer.fit()
        score = trainer.score()
        self.assertTrue(score > 0.0, f"Model should achieve positive R^2, got {score}")

    def test_prediction_accuracy(self):
        """Test prediction accuracy and generalization on test set."""
        if not self.real_data_available:
            self.skipTest("Real data not available")
        trainer = AdaBoostTrainer(self.window, n_estimators=20, learning_rate=1.0)
        trainer.fit()
        y_pred = trainer.predict()
        # Get true labels
        _, y_true, _ = trainer._extract_xy(self.window.test)
        # Check shape
        self.assertEqual(y_pred.shape, y_true.shape)
        # Check reasonable error
        mae = np.mean(np.abs(y_pred - y_true))
        self.assertLess(mae, 1.0, f"Mean absolute error should be < 1.0, got {mae}")

    def test_empty_dataset(self):
        """Test behavior with empty datasets."""
        empty_df = pd.DataFrame(columns=self.train_df.columns)
        window = WindowGenerator(
            input_width=24, label_width=1, shift=1,
            train_df=empty_df, val_df=empty_df, test_df=empty_df,
            label_columns=['Current Power']
        )
        trainer = AdaBoostTrainer(window)
        with self.assertRaises(ValueError):
            trainer.fit()

    def test_single_class(self):
        """Test behavior with single-class (constant label) data."""
        df = self.train_df.copy()
        df['Current Power'] = 42.0
        window = WindowGenerator(
            input_width=24, label_width=1, shift=1,
            train_df=df, val_df=df, test_df=df,
            label_columns=['Current Power']
        )
        trainer = AdaBoostTrainer(window)
        trainer.fit()
        y_pred = trainer.predict()
        # All predictions should be close to the constant value
        self.assertTrue(np.allclose(y_pred, 42.0, atol=1e-2))

    def test_highly_imbalanced(self):
        """Test behavior with highly imbalanced data."""
        from sklearn.ensemble import AdaBoostClassifier
        df = self.train_df.copy()
        # Create binary labels: 1 if 'Current Power' > 50, 0 otherwise
        df['Current Power'] = (df['Current Power'] > 50).astype(int)
        window = WindowGenerator(
            input_width=24, label_width=1, shift=1,
            train_df=df, val_df=df, test_df=df,
            label_columns=['Current Power']
        )
        trainer = AdaBoostTrainer(window)
        trainer.model = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
        trainer.fit()
        y_pred = trainer.predict()
        # Check that predictions are not all zero (model learns minority)
        self.assertTrue(np.any(y_pred > 0.5))

    def test_invalid_input(self):
        """Test error handling for malformed input."""
        # Pass None as window generator
        with self.assertRaises(AttributeError):
            AdaBoostTrainer(None).fit()
        # Pass invalid label index
        with self.assertRaises(IndexError):
            AdaBoostTrainer(self.window, label_index=100).fit()

    def test_feature_importances(self):
        """Test feature importances output shape and type."""
        if not self.real_data_available:
            self.skipTest("Real data not available")
        trainer = AdaBoostTrainer(self.window)
        trainer.fit()
        importances = trainer.get_feature_importances()
        self.assertIsInstance(importances, np.ndarray)
        self.assertEqual(importances.ndim, 1)
        # Should match input feature size
        n_features = self.window.input_width * len(self.window.train_df.columns)
        self.assertEqual(importances.shape[0], n_features)

if __name__ == "__main__":
    unittest.main()