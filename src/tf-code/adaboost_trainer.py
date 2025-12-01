"""
AdaBoostTrainer: A robust AdaBoost regression wrapper for use with WindowGenerator.

This class integrates with the WindowGenerator to extract windowed time series data,
flattens it for use with scikit-learn's AdaBoostRegressor, and provides flexible
parameterization and efficient training/prediction methods.

Author: Watcharin Sarachai
"""

from typing import Optional, Any, Tuple, Union, Dict
import numpy as np
import os
import pickle
import joblib
import logging
from datetime import datetime
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit

try:
    from .base_trainer import BaseTrainer
except ImportError:
    from base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class AdaBoostTrainer(BaseTrainer):
    """
    AdaBoostTrainer encapsulates an AdaBoost regression model for use with time series data.

    This class provides methods for training, prediction, and evaluation using time series
    cross-validation. It includes a development mode for faster training during testing,
    and model checkpointing functionality to save and restore the best performing model.

    Attributes:
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Learning rate shrinks the contribution of each regressor.
        base_estimator (Any): The base estimator from which the boosted ensemble is built.
        model (AdaBoostRegressor): The underlying AdaBoost regressor.
        dev_mode (bool): Whether development mode is enabled for faster training.
        dev_sample_ratio (float): Ratio of data to use in development mode.
        cv_scores (list): Cross-validation scores from training.
        cv_mean (float): Mean cross-validation score.
        cv_std (float): Standard deviation of cross-validation scores.
        checkpoint_enabled (bool): Whether model checkpointing is enabled.
        checkpoint_path (str): Path to save/load model checkpoints.
        best_model (AdaBoostRegressor): Best performing model checkpoint.
        best_score (float): Best validation score achieved.
        load_existing_model (bool): Whether to load existing model instead of training.
    """

    def __init__(
        self,
        window_generator,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        label_index: int = 0,
        differencing_order: int = 0,
        dev_mode: bool = False,
        dev_sample_ratio: float = 0.1,
        checkpoint_enabled: bool = True,
        checkpoint_path: str = "checkpoints/adaboost_model.pkl",
        load_existing_model: bool = False,
        **kwargs
    ):
        """
        Initialize the AdaBoostTrainer.

        Args:
            window_generator (WindowGenerator): The window generator instance.
            n_estimators (int): Number of boosting stages.
            learning_rate (float): Learning rate.
            label_index (int): Index of the label to use if multiple labels are present.
            differencing_order (int): Order of differencing to apply to target variable.
            dev_mode (bool): Enable development mode for faster training during testing.
            dev_sample_ratio (float): Ratio of data to use in development mode (0.0-1.0).
            checkpoint_enabled (bool): Enable model checkpointing to save best performing model.
            checkpoint_path (str): Path to save/load model checkpoints.
            load_existing_model (bool): Load existing model instead of training new one.
            **kwargs: Additional configuration parameters
        """
        super().__init__(window_generator, **kwargs)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.label_index = label_index
        self.differencing_order = differencing_order
        self.dev_mode = dev_mode
        self.dev_sample_ratio = dev_sample_ratio
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_path = checkpoint_path
        self.load_existing_model = load_existing_model
        
        # Initialize checkpointing attributes
        self.best_model = None
        self.best_score = -np.inf
        
        # Adjust parameters for development mode
        if self.dev_mode:
            # Reduce number of estimators for faster training
            dev_n_estimators = max(5, min(10, self.n_estimators // 5))
            print(f"[DEV MODE] Reducing n_estimators from {self.n_estimators} to {dev_n_estimators}")
            print(f"[DEV MODE] Using {self.dev_sample_ratio:.1%} of training data")
        else:
            dev_n_estimators = self.n_estimators
            
        self.model = AdaBoostRegressor(
            n_estimators=dev_n_estimators,
            learning_rate=self.learning_rate,
            random_state=42
        )
        
        # Initialize cross-validation attributes
        self.cv_scores = []
        self.cv_mean = 0.0
        self.cv_std = 0.0
        
        # Try to load existing model if requested
        if self.load_existing_model and self.checkpoint_enabled:
            self._load_model()

    def _extract_xy(
        self,
        dataset: Any,
        apply_dev_sampling: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts and flattens (X, y) from a tf.data.Dataset produced by WindowGenerator and applies differencing.

        Args:
            dataset (tf.data.Dataset): Dataset yielding (inputs, labels).
            apply_dev_sampling (bool): Whether to apply development mode sampling.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Flattened X and y arrays.
        """
        X_list, y_list = [], []
        for batch_x, batch_y in dataset:
            # batch_x: [batch, input_width, features]
            # batch_y: [batch, label_width, label_features]
            # For AdaBoost, flatten time and features: [batch, input_width * features]
            # Handle both numpy arrays and tensorflow tensors
            if hasattr(batch_x, 'numpy'):
                batch_x_np = batch_x.numpy()
            else:
                batch_x_np = batch_x
                
            if hasattr(batch_y, 'numpy'):
                batch_y_np = batch_y.numpy()
            else:
                batch_y_np = batch_y
            batch_x_flat = batch_x_np.reshape(batch_x_np.shape[0], -1)
            # For regression, use the first label time step and selected label index
            if batch_y_np.ndim == 3:
                batch_y_flat = batch_y_np[:, 0, self.label_index]
            else:
                batch_y_flat = batch_y_np[:, self.label_index]
            X_list.append(batch_x_flat)
            y_list.append(batch_y_flat)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # Apply development mode sampling if enabled
        if apply_dev_sampling and self.dev_mode and self.dev_sample_ratio < 1.0:
            n_samples = len(X)
            n_dev_samples = int(n_samples * self.dev_sample_ratio)
            # Use random sampling to maintain data distribution
            np.random.seed(42)  # For reproducible results
            indices = np.random.choice(n_samples, n_dev_samples, replace=False)
            indices = np.sort(indices)  # Maintain temporal order for time series
            X = X[indices]
            y = y[indices]
            print(f"[DEV MODE] Sampled {n_dev_samples:,} from {n_samples:,} samples ({self.dev_sample_ratio:.1%})")
        
        # Apply differencing to y
        if self.differencing_order > 0:
            y = np.diff(y, n=self.differencing_order)
        return X, y
    def fit(self, **kwargs) -> Dict[str, Any]:
        """
        Fit the AdaBoost regressor using time series cross-validation.

        The data is split into training and validation sets using TimeSeriesSplit, and the
        model is trained on each training set. The final model is trained on the full dataset.
        If checkpointing is enabled, the best performing model is saved and can be restored.

        Args:
            **kwargs: Additional training parameters including:
                - n_splits (int): Number of splits for time series cross-validation (default: 5)
                - dev_mode (bool): Override development mode setting
                
        Returns:
            Dict[str, Any]: Training results including cross-validation scores and metrics
        """
        logger.info("Starting AdaBoost model training")
        
        # Override parameters with kwargs if provided
        n_splits = kwargs.get('n_splits', 5)
        dev_mode = kwargs.get('dev_mode', self.dev_mode)
        # If model was loaded, skip training
        if self.load_existing_model and self.best_model is not None:
            logger.info("Using loaded model - skipping training")
            return {
                'cv_scores': self.cv_scores,
                'cv_mean': self.cv_mean,
                'cv_std': self.cv_std,
                'best_score': self.best_score,
                'model_loaded': True
            }
        
        train_ds = self.window_generator.train
        X, y = self._extract_xy(train_ds, apply_dev_sampling=True)
        # Adjust X to match the size of y after differencing
        if self.differencing_order > 0:
            X = X[self.differencing_order:]
        
        if dev_mode:
            logger.info(f"Development mode: Training with {len(X):,} samples and {self.model.n_estimators} estimators")
        
        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        logger.info(f"Performing {n_splits}-fold time series cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model on this fold
            fold_model = AdaBoostRegressor(
                n_estimators=self.model.n_estimators,
                learning_rate=self.learning_rate,
                random_state=42
            )
            fold_model.fit(X_train, y_train)
            
            # Calculate validation score
            val_score = fold_model.score(X_val, y_val)
            cv_scores.append(val_score)
            
            logger.info(f"  Fold {fold + 1}: R2 score = {val_score:.4f}")
            
            # Check if this is the best model so far and save if checkpointing enabled
            if self.checkpoint_enabled and val_score > self.best_score:
                self.best_score = val_score
                self.best_model = fold_model
                logger.info(f"  [CHECKPOINT] New best model saved with R2 score = {val_score:.4f}")
        
        # Store cross-validation results
        self.cv_scores = cv_scores
        self.cv_mean = np.mean(cv_scores)
        self.cv_std = np.std(cv_scores)
        
        logger.info(f"Cross-validation results: Mean R2 = {self.cv_mean:.4f} (+/-{self.cv_std:.4f})")
        
        # Train final model on full dataset
        logger.info("Training final model on full dataset...")
        self.model.fit(X, y)
        
        # Save the final model if checkpointing is enabled
        if self.checkpoint_enabled:
            self._save_model()
        
        # Return training results
        return {
            'cv_scores': cv_scores,
            'cv_mean': float(self.cv_mean),
            'cv_std': float(self.cv_std),
            'best_score': float(self.best_score),
            'n_splits': n_splits,
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.learning_rate,
            'dev_mode': dev_mode,
            'samples_used': len(X)
        }

    def predict(
        self, 
        dataset: Optional[Any] = None
    ) -> np.ndarray:
        """
        Predict using the AdaBoost regressor.

        Args:
            dataset (tf.data.Dataset, optional): Dataset to predict on. If None, uses test set.

        Returns:
            np.ndarray: Predicted values.
        """
        if dataset is None:
            dataset = self.window_generator.test
        X, y = self._extract_xy(dataset)
        if self.differencing_order > 0:
            X = X[self.differencing_order:]
        return self.model.predict(X)

    def score(
        self,
        dataset: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for the model.

        Args:
            dataset (tf.data.Dataset, optional): Dataset to score on. If None, uses test set.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        logger.info("Evaluating AdaBoost model")
        
        if dataset is None:
            dataset = self.window_generator.test
        X, y = self._extract_xy(dataset)
        if self.differencing_order > 0:
            X = X[self.differencing_order:]
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        r2_score = self.model.score(X, y)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mae = np.mean(np.abs(y - y_pred))
        
        # Calculate MAPE (avoiding division by zero)
        non_zero_mask = y != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        results = {
            'r2': float(r2_score),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }
        
        logger.info(f"Evaluation results: {results}")
        return results

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model. If None, uses checkpoint_path.
            
        Returns:
            Path where the model was saved
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = path or self.checkpoint_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model with metadata
        checkpoint_data = {
            'model': self.best_model if self.best_model is not None else self.model,
            'best_score': self.best_score,
            'cv_scores': self.cv_scores,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'label_index': self.label_index,
            'differencing_order': self.differencing_order,
            'dev_mode': self.dev_mode,
            'dev_sample_ratio': self.dev_sample_ratio,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'adaboost'
        }
        
        joblib.dump(checkpoint_data, save_path)
        logger.info(f"Model saved to {save_path}")
        return save_path
    
    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from. If None, uses checkpoint_path.
        """
        load_path = path or self.checkpoint_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load checkpoint data
        checkpoint_data = joblib.load(load_path)
        
        # Restore model and attributes
        self.model = checkpoint_data['model']
        self.best_model = checkpoint_data['model']
        self.best_score = checkpoint_data['best_score']
        self.cv_scores = checkpoint_data.get('cv_scores', [])
        self.cv_mean = checkpoint_data.get('cv_mean', 0.0)
        self.cv_std = checkpoint_data.get('cv_std', 0.0)
        
        # Restore configuration if available
        self.n_estimators = checkpoint_data.get('n_estimators', self.n_estimators)
        self.learning_rate = checkpoint_data.get('learning_rate', self.learning_rate)
        self.label_index = checkpoint_data.get('label_index', self.label_index)
        self.differencing_order = checkpoint_data.get('differencing_order', self.differencing_order)
        
        logger.info(f"Model loaded from {load_path}")
        logger.info(f"Best R2 score: {self.best_score:.4f}")
        if self.cv_scores:
            logger.info(f"Cross-validation: Mean R2 = {self.cv_mean:.4f} (+/-{self.cv_std:.4f})")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': 'adaboost',
            'trained': self.model is not None,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'label_index': self.label_index,
            'differencing_order': self.differencing_order,
            'dev_mode': self.dev_mode,
            'dev_sample_ratio': self.dev_sample_ratio,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_path': self.checkpoint_path
        }
        
        if self.model is not None:
            info.update({
                'best_score': float(self.best_score),
                'cv_mean': float(self.cv_mean),
                'cv_std': float(self.cv_std),
                'cv_scores': self.cv_scores,
                'feature_importances_available': hasattr(self.model, 'feature_importances_')
            })
        
        return info

    def calculate_rmse(self, dataset: Optional[Any] = None) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE) of the model.

        Args:
            dataset (tf.data.Dataset, optional): Dataset to calculate RMSE on. If None, uses test set.

        Returns:
            float: RMSE.
        """
        if dataset is None:
            dataset = self.window_generator.test
        X, y = self._extract_xy(dataset)
        if self.differencing_order > 0:
            X = X[self.differencing_order:]
        y_pred = self.model.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def calculate_mae(self, dataset: Optional[Any] = None) -> float:
        """
        Calculate the Mean Absolute Error (MAE) of the model.

        Args:
            dataset (tf.data.Dataset, optional): Dataset to calculate MAE on. If None, uses test set.

        Returns:
            float: MAE.
        """
        if dataset is None:
            dataset = self.window_generator.test
        X, y = self._extract_xy(dataset)
        if self.differencing_order > 0:
            X = X[self.differencing_order:]
        y_pred = self.model.predict(X)
        return np.mean(np.abs(y - y_pred))

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the AdaBoost regressor.

        Returns:
            np.ndarray: Feature importances.
        """
        return self.model.feature_importances_
    
    def set_dev_mode(self, enabled: bool, sample_ratio: float = 0.1) -> None:
        """
        Toggle development mode for faster training during testing.
        
        Args:
            enabled (bool): Whether to enable development mode.
            sample_ratio (float): Ratio of data to use in development mode (0.0-1.0).
        """
        self.dev_mode = enabled
        self.dev_sample_ratio = sample_ratio
        
        # Recreate model with adjusted parameters
        if self.dev_mode:
            dev_n_estimators = max(5, min(10, self.n_estimators // 5))
            print(f"[DEV MODE] Enabled - reducing n_estimators from {self.n_estimators} to {dev_n_estimators}")
            print(f"[DEV MODE] Using {self.dev_sample_ratio:.1%} of training data")
        else:
            dev_n_estimators = self.n_estimators
            print(f"[PRODUCTION MODE] Enabled - using full {self.n_estimators} estimators")
            
        self.model = AdaBoostRegressor(
            n_estimators=dev_n_estimators,
            learning_rate=self.learning_rate,
            random_state=42
        )
        
        # Reset cross-validation attributes when changing mode
        self.cv_scores = []
        self.cv_mean = 0.0
        self.cv_std = 0.0
    
    def is_dev_mode(self) -> bool:
        """
        Check if development mode is currently enabled.
        
        Returns:
            bool: True if development mode is enabled, False otherwise.
        """
        return self.dev_mode
    
    def get_cv_results(self) -> dict:
        """
        Get cross-validation results from the last training run.
        
        Returns:
            dict: Dictionary containing cross-validation results with keys:
                - 'scores': List of individual fold scores
                - 'mean': Mean cross-validation score
                - 'std': Standard deviation of cross-validation scores
        """
        return {
            'scores': self.cv_scores,
            'mean': self.cv_mean,
            'std': self.cv_std
        }
    
    def _save_model(self) -> None:
        """
        Save the best model to disk using joblib for efficient serialization.
        """
        if not self.checkpoint_enabled:
            return
            
        try:
            # Save the best model if available, otherwise save the current model
            model_to_save = self.best_model if self.best_model is not None else self.model
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            
            # Save model with metadata
            checkpoint_data = {
                'model': model_to_save,
                'best_score': self.best_score,
                'cv_scores': self.cv_scores,
                'cv_mean': self.cv_mean,
                'cv_std': self.cv_std,
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'dev_mode': self.dev_mode,
                'dev_sample_ratio': self.dev_sample_ratio
            }
            
            joblib.dump(checkpoint_data, self.checkpoint_path)
            print(f"[CHECKPOINT] Model saved to {self.checkpoint_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to save model checkpoint: {e}")
    
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
                print(f"[INFO] No existing model found at {self.checkpoint_path}")
                return False
            
            # Load checkpoint data
            checkpoint_data = joblib.load(self.checkpoint_path)
            
            # Restore model and attributes
            self.best_model = checkpoint_data['model']
            self.best_score = checkpoint_data['best_score']
            self.cv_scores = checkpoint_data.get('cv_scores', [])
            self.cv_mean = checkpoint_data.get('cv_mean', 0.0)
            self.cv_std = checkpoint_data.get('cv_std', 0.0)
            
            # Use the loaded model as the current model
            self.model = self.best_model
            
            print(f"[CHECKPOINT] Model loaded from {self.checkpoint_path}")
            print(f"[CHECKPOINT] Best R2 score: {self.best_score:.4f}")
            if self.cv_scores:
                print(f"[CHECKPOINT] Cross-validation: Mean R2 = {self.cv_mean:.4f} (+/-{self.cv_std:.4f})")
            
            return True
            
        except Exception as e:
            print(f"[WARNING] Failed to load model checkpoint: {e}")
            return False
    
    def restore_best_model(self) -> bool:
        """
        Restore the best model from checkpoint and set it as the current model.
        
        Returns:
            bool: True if model was restored successfully, False otherwise.
        """
        if self.best_model is not None:
            self.model = self.best_model
            print(f"[RESTORE] Best model restored with R2 score = {self.best_score:.4f}")
            return True
        return False
    
    def get_best_score(self) -> float:
        """
        Get the best validation score achieved during training.
        
        Returns:
            float: Best validation score, or -inf if no training has occurred.
        """
        return self.best_score
    
    def delete_checkpoint(self) -> bool:
        """
        Delete the saved model checkpoint file.
        
        Returns:
            bool: True if checkpoint was deleted successfully, False otherwise.
        """
        if not self.checkpoint_enabled:
            return False
            
        try:
            if os.path.exists(self.checkpoint_path):
                os.remove(self.checkpoint_path)
                print(f"[CHECKPOINT] Checkpoint file deleted: {self.checkpoint_path}")
                return True
            else:
                print(f"[INFO] No checkpoint file found at {self.checkpoint_path}")
                return False
        except Exception as e:
            print(f"[WARNING] Failed to delete checkpoint: {e}")
            return False