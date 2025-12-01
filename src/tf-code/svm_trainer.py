import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Any, Dict, Optional, Tuple, Union
try:
    from .base_trainer import BaseTrainer
except ImportError:
    from base_trainer import BaseTrainer

class SVMTrainer(BaseTrainer):
    """
    Support Vector Machine (SVM) trainer for time series forecasting.

    This class implements the BaseTrainer interface for SVM-based forecasting models,
    providing hyperparameter optimization, cross-validation, and evaluation capabilities.
    """

    def __init__(self, window_generator, **kwargs):
        """
        Initialize the SVM trainer.

        Args:
            window_generator: WindowGenerator instance for data handling
            **kwargs: Additional configuration parameters
        """
        super().__init__(window_generator, **kwargs)
        
        # SVM-specific parameters
        self.kernel = kwargs.get('kernel', 'rbf')
        self.C = kwargs.get('C', 1.0)
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.gamma = kwargs.get('gamma', 'scale')
        self.scaler = StandardScaler()
        
        # Hyperparameter optimization settings
        self.optimization_method = kwargs.get('optimization_method', 'grid')  # 'grid' or 'random'
        self.optimization_params = kwargs.get('optimization_params', {})
        self.cv_folds = kwargs.get('cv_folds', 5)
        
        # Development mode settings
        self.dev_mode = kwargs.get('dev_mode', False)
        self.dev_sample_ratio = kwargs.get('dev_sample_ratio', 0.2)
        
        # Model persistence
        self.checkpoint_enabled = kwargs.get('checkpoint_enabled', True)
        self.checkpoint_path = kwargs.get('checkpoint_path', 'svm_model.pkl')
        self.load_existing_model = kwargs.get('load_existing_model', False)
        
        # Initialize hyperparameters
        self.hyperparameters = {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma
        }
        
        self.logger.info(f"Initialized SVMTrainer with kernel={self.kernel}, C={self.C}, epsilon={self.epsilon}")

        if self.load_existing_model:
            self._load_model()

    def fit(self, **kwargs) -> Dict[str, Any]:
        """
        Train the SVM model with optional hyperparameter optimization.

        Args:
            **kwargs: Additional training parameters

        Returns:
            Dict containing training history and metrics
        """
        try:
            self.logger.info("Starting SVM model training")

            if self.load_existing_model and self.is_trained:
                self.logger.info("Using loaded model - skipping training")
                return {
                    'model_loaded': True,
                    **self.training_history
                }
            
            # Extract training data
            X_train, y_train = self._extract_features_labels(self.window_generator.train)
            
            # Apply development mode if enabled
            if self.dev_mode:
                # DEV MODE: Reduce training set size for faster iteration
                X_train = X_train[:500]
                y_train = y_train[:500]
                self.logger.info(f"Development mode: using {len(X_train)} samples")

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Hyperparameter optimization if enabled
            if self.optimization_params:
                self.logger.info(f"Performing {self.optimization_method} search for hyperparameters")
                self.model = self._optimize_hyperparameters(X_train_scaled, y_train)
                self.hyperparameters.update(self.model.get_params())
            else:
                # Train with default parameters
                self.model = SVR(
                    kernel=self.kernel,
                    C=self.C,
                    epsilon=self.epsilon,
                    gamma=self.gamma
                )
                self.model.fit(X_train_scaled, y_train)
            
            self.is_trained = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train_scaled)
            training_metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_predictions)),
                'train_mae': mean_absolute_error(y_train, train_predictions),
                'train_r2': r2_score(y_train, train_predictions)
            }
            
            # Save model if checkpointing is enabled
            if self.checkpoint_enabled:
                self.save_model(self.checkpoint_path)
            
            self.training_history.update(training_metrics)
            self.logger.info("SVM model training completed successfully")
            
            return training_metrics
            
        except Exception as e:
            self.logger.error(f"Error during SVM training: {str(e)}")
            raise

    def predict(self, dataset: Optional[Any] = None) -> np.ndarray:
        """
        Make predictions using the trained SVM model.

        Args:
            dataset: Dataset to predict on (uses window_generator.test if None)

        Returns:
            NumPy array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            if dataset is None:
                dataset = self.window_generator.test
            
            X, _ = self._extract_features_labels(dataset)
            X_scaled = self.scaler.transform(X)
            
            predictions = self.model.predict(X_scaled)
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def score(self, dataset: Optional[Any] = None) -> Dict[str, float]:
        """
        Calculate evaluation metrics for the given dataset.

        Args:
            dataset: Dataset to score (uses window_generator.test if None)

        Returns:
            Dictionary of evaluation metrics
        """
        return self.calculate_metrics(dataset)

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained SVM model to disk.

        Args:
            path: File path to save the model (uses checkpoint_path if None)

        Returns:
            Path where the model was saved
        """
        if path is None:
            path = self.checkpoint_path
            
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'hyperparameters': self.hyperparameters,
                'training_history': self.training_history,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, path)
            self.logger.info(f"Model saved to {path}")
            return path
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def _load_model(self) -> bool:
        """
        Load the best model from disk.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise.
        """
        if not self.checkpoint_enabled:
            return False
            
        try:
            if not os.path.exists(self.checkpoint_path):
                self.logger.info(f"No existing model found at {self.checkpoint_path}")
                return False
            
            self.load_model(self.checkpoint_path)
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load model checkpoint: {e}")
            return False

    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a trained SVM model from disk.

        Args:
            path: File path to load the model from (uses checkpoint_path if None)
        """
        if path is None:
            path = self.checkpoint_path
            
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.hyperparameters = model_data['hyperparameters']
            self.training_history = model_data['training_history']
            self.is_trained = model_data['is_trained']
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'model_type': 'svm',
            'trained': self.is_trained,
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'optimization_method': self.optimization_method,
            'cv_folds': self.cv_folds,
            'dev_mode': self.dev_mode,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_path': self.checkpoint_path
        }
        
        if self.is_trained:
            info.update({
                'hyperparameters': self.hyperparameters,
                'training_history': self.training_history
            })
        
        return info

    def _extract_actual_values(self, dataset: Optional[Any] = None) -> np.ndarray:
        """Extract actual values from dataset for metric calculations."""
        if dataset is None:
            dataset = self.window_generator.test
        
        _, y = self._extract_features_labels(dataset)
        return y

    def _extract_features_labels(self, dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from the dataset.

        Args:
            dataset: Dataset to extract from

        Returns:
            Tuple of (features, labels)
        """
        # Convert dataset to numpy arrays if needed
        if hasattr(dataset, '__iter__') and not isinstance(dataset, (np.ndarray, pd.DataFrame)):
            # Handle TensorFlow dataset
            features_list = []
            labels_list = []
            for batch in dataset:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_features = batch[0].numpy()
                    batch_labels = batch[1].numpy()
                    
                    # Flatten features for SVM (from [batch, time, features] to [batch, time*features])
                    if batch_features.ndim == 3:
                        batch_features = batch_features.reshape(batch_features.shape[0], -1)
                    
                    # Flatten labels if needed (from [batch, time, 1] to [batch])
                    if batch_labels.ndim > 1:
                        batch_labels = batch_labels.flatten()
                    
                    features_list.append(batch_features)
                    labels_list.append(batch_labels)
            
            if features_list:
                features = np.vstack(features_list)
                labels = np.concatenate(labels_list)
            else:
                raise ValueError("Could not extract features and labels from dataset")
        else:
            # Handle numpy arrays or pandas DataFrames
            if isinstance(dataset, pd.DataFrame):
                # Assume last column is the label
                features = dataset.iloc[:, :-1].values
                labels = dataset.iloc[:, -1].values
            else:
                # Assume numpy array with last column as label
                features = dataset[:, :-1]
                labels = dataset[:, -1]
        
        return features, labels

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> SVR:
        """
        Perform hyperparameter optimization using GridSearchCV or RandomizedSearchCV.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Best SVR model
        """
        param_grid = self.optimization_params
        
        if self.optimization_method == 'grid':
            search = GridSearchCV(
                SVR(),
                param_grid,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                SVR(),
                param_grid,
                n_iter=min(20, len(param_grid) * 5),
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        search.fit(X, y)
        self.logger.info(f"Best parameters: {search.best_params_}")
        self.logger.info(f"Best score: {-search.best_score_}")
        
        return search.best_estimator_