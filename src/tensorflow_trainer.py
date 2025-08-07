"""
TensorFlow trainer implementation for the unified forecasting system.
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

try:
    from .base_trainer import BaseTrainer
except ImportError:
    from base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class TensorFlowTrainer(BaseTrainer):
    """
    TensorFlow-based forecasting trainer implementing the BaseTrainer interface.
    """
    
    def __init__(self, window_generator, **kwargs):
        """
        Initialize the TensorFlow trainer.
        
        Args:
            window_generator: WindowGenerator instance for data preparation
            **kwargs: Additional configuration parameters
        """
        super().__init__(window_generator, **kwargs)
        
        # Model parameters
        self.max_epochs = kwargs.get('max_epochs', 20)
        self.patience = kwargs.get('patience', 2)
        self.dev_mode = kwargs.get('dev_mode', False)
        self.checkpoint_enabled = kwargs.get('checkpoint_enabled', True)
        self.checkpoint_path = kwargs.get('checkpoint_path', 'checkpoints/tensorflow_model.h5')
        
        # Model and training state
        self.model = None
        self.history = None
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory if needed
        if self.checkpoint_enabled:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
    
    def _create_model(self, input_shape: Tuple[int, ...], num_features: int) -> tf.keras.Model:
        """
        Create a TensorFlow model for time series forecasting.
        
        Args:
            input_shape: Shape of input data
            num_features: Number of input features
            
        Returns:
            Compiled TensorFlow model
        """
        # Simple LSTM model architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        
        return model
    
    def _create_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Create callbacks for model training.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            mode='min',
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint callback
        if self.checkpoint_enabled:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                save_weights_only=False
            )
            callbacks.append(checkpoint)
        
        return callbacks
    
    def fit(self, **kwargs) -> Dict[str, Any]:
        """
        Train the TensorFlow model.
        
        Args:
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        logger.info("Starting TensorFlow model training")
        
        # Override parameters with kwargs if provided
        max_epochs = kwargs.get('max_epochs', self.max_epochs)
        patience = kwargs.get('patience', self.patience)
        dev_mode = kwargs.get('dev_mode', self.dev_mode)
        
        # Get training and validation data
        train_ds = self.window_generator.train
        val_ds = self.window_generator.val
        
        # Determine input shape and number of features
        for example_inputs, _ in train_ds.take(1):
            input_shape = example_inputs.shape[1:]
            num_features = example_inputs.shape[-1]
            break
        
        # Create model
        self.model = self._create_model(input_shape, num_features)
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Adjust epochs for development mode
        if dev_mode:
            max_epochs = min(max_epochs, 5)
            logger.info(f"Development mode enabled, reducing epochs to {max_epochs}")
        
        # Train model
        self.history = self.model.fit(
            train_ds,
            epochs=max_epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        # Find best epoch based on validation loss
        val_loss = self.history.history['val_loss']
        self.best_epoch = np.argmin(val_loss)
        self.best_val_loss = val_loss[self.best_epoch]
        
        logger.info(f"Training completed. Best epoch: {self.best_epoch}, Best val loss: {self.best_val_loss:.4f}")
        
        # Return training results
        return {
            'history': self.history.history,
            'best_epoch': int(self.best_epoch),
            'best_val_loss': float(self.best_val_loss),
            'epochs_trained': len(self.history.history['loss']),
            'model_summary': self._get_model_summary()
        }
    
    def predict(self, dataset: Optional[Any] = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            dataset: Dataset to predict on. If None, uses test set.
            
        Returns:
            Numpy array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        logger.info("Making predictions with TensorFlow model")
        
        # Use test set if no dataset provided
        if dataset is None:
            dataset = self.window_generator.test
        
        # Make predictions
        predictions = self.model.predict(dataset)
        
        return predictions
    
    def score(self, dataset: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.
        
        Args:
            dataset: Dataset to evaluate on. If None, uses test set.
            
        Returns:
            Dictionary of evaluation metrics
        """
        return self.calculate_metrics(dataset)
    
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
        
        # Save model
        self.model.save(save_path)
        
        # Save additional metadata
        metadata = {
            'model_type': 'tensorflow',
            'best_epoch': int(self.best_epoch),
            'best_val_loss': float(self.best_val_loss),
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = save_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
        
        # Load model
        self.model = tf.keras.models.load_model(load_path)
        
        # Load metadata if available
        metadata_path = load_path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.best_epoch = metadata.get('best_epoch', 0)
                self.best_val_loss = metadata.get('best_val_loss', float('inf'))
                self.max_epochs = metadata.get('max_epochs', self.max_epochs)
                self.patience = metadata.get('patience', self.patience)
        
        logger.info(f"Model loaded from {load_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': 'tensorflow',
            'trained': self.model is not None,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'dev_mode': self.dev_mode,
            'checkpoint_enabled': self.checkpoint_enabled,
            'checkpoint_path': self.checkpoint_path
        }
        
        if self.model is not None:
            info.update({
                'best_epoch': int(self.best_epoch),
                'best_val_loss': float(self.best_val_loss),
                'model_summary': self._get_model_summary()
            })
        
        return info
    
    def _get_model_summary(self) -> str:
        """
        Get a string representation of the model summary.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not created yet"
        
        # Capture model summary output
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()
        stream.close()
        
        return summary_str