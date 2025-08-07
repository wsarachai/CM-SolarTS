from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import logging

class BaseTrainer(ABC):
    """
    Abstract base class for all forecasting model trainers.

    This class defines the standard interface that all model trainers must implement,
    ensuring consistency across different forecasting algorithms (TensorFlow, AdaBoost, SVM, etc.).
    """

    def __init__(self, window_generator, **kwargs):
        self.window_generator = window_generator
        self.model = None
        self.is_trained = False
        self.hyperparameters = {}
        self.training_history = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fit(self, **kwargs) -> Dict[str, Any]:
        """Train the model and return training history/metrics."""
        pass

    @abstractmethod
    def predict(self, dataset: Optional[Any] = None) -> np.ndarray:
        """Make predictions on the given dataset."""
        pass

    @abstractmethod
    def score(self, dataset: Optional[Any] = None) -> Dict[str, float]:
        """Calculate evaluation metrics and return as dictionary."""
        pass

    @abstractmethod
    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model to disk and return the path."""
        pass

    @abstractmethod
    def load_model(self, path: Optional[str] = None) -> None:
        """Load a trained model from disk."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        pass

    # Common utility methods
    def calculate_rmse(self, dataset: Optional[Any] = None) -> float:
        """Calculate Root Mean Squared Error."""
        predictions = self.predict(dataset)
        actual = self._extract_actual_values(dataset)
        return np.sqrt(np.mean((actual - predictions) ** 2))

    def calculate_mae(self, dataset: Optional[Any] = None) -> float:
        """Calculate Mean Absolute Error."""
        predictions = self.predict(dataset)
        actual = self._extract_actual_values(dataset)
        return np.mean(np.abs(actual - predictions))

    def calculate_mape(self, dataset: Optional[Any] = None) -> float:
        """Calculate Mean Absolute Percentage Error."""
        predictions = self.predict(dataset)
        actual = self._extract_actual_values(dataset)
        # Avoid division by zero
        mask = actual != 0
        if np.any(mask):
            return np.mean(np.abs((actual[mask] - predictions[mask]) / actual[mask])) * 100
        else:
            return float('inf')

    def calculate_metrics(self, dataset: Optional[Any] = None) -> Dict[str, float]:
        """Calculate common evaluation metrics."""
        predictions = self.predict(dataset)
        actual = self._extract_actual_values(dataset)
        
        # Ensure predictions and actual have the same shape
        if predictions.shape != actual.shape:
            predictions = predictions.flatten()
            actual = actual.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((actual - predictions) ** 2))
        mae = np.mean(np.abs(actual - predictions))
        
        # Calculate MAPE (avoiding division by zero)
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual[non_zero_mask] - predictions[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # Calculate R²
        ss_res = np.sum((actual - predictions) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        }

    def _extract_actual_values(self, dataset: Optional[Any] = None) -> np.ndarray:
        """Extract actual values from dataset for metric calculations."""
        if dataset is None:
            dataset = self.window_generator.test
        
        # Handle different dataset types
        if hasattr(dataset, '__iter__') and not isinstance(dataset, np.ndarray):
            # Handle TensorFlow dataset
            labels_list = []
            for batch in dataset:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    labels = batch[1]
                    if hasattr(labels, 'numpy'):
                        labels = labels.numpy()
                    labels_list.append(labels)
            
            if labels_list:
                return np.concatenate(labels_list)
            else:
                raise ValueError("Could not extract labels from dataset")
        else:
            # Handle numpy arrays
            return dataset