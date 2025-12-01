import logging
from typing import Dict, Any, Type, Optional

try:
    from .base_trainer import BaseTrainer
    from .svm_trainer import SVMTrainer
    from .tensorflow_trainer import TensorFlowTrainer
    from .adaboost_trainer import AdaBoostTrainer
except ImportError:
    from base_trainer import BaseTrainer
    from svm_trainer import SVMTrainer
    from tensorflow_trainer import TensorFlowTrainer
    from adaboost_trainer import AdaBoostTrainer

class ModelFactory:
    """
    Factory class for creating forecasting model trainers.

    This class implements the factory pattern to create instances of different
    forecasting model trainers based on configuration parameters.
    """

    # Registry of available model trainers
    _model_registry: Dict[str, Type[BaseTrainer]] = {
        'svm': SVMTrainer,
        'tensorflow': TensorFlowTrainer,
        'adaboost': AdaBoostTrainer,
    }

    # Default configurations for each model type
    _default_configs: Dict[str, Dict[str, Any]] = {
        'svm': {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale',
            'optimization_method': 'grid',
            'cv_folds': 5,
            'dev_mode': False,
            'checkpoint_enabled': True,
            'checkpoint_path': 'checkpoints/svm_model.pkl'
        },
        'tensorflow': {
            'max_epochs': 20,
            'patience': 2,
            'dev_mode': False,
            'checkpoint_enabled': True,
            'checkpoint_path': 'checkpoints/tensorflow_model.h5'
        },
        'adaboost': {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'label_index': 0,
            'differencing_order': 0,
            'dev_mode': False,
            'dev_sample_ratio': 0.1,
            'checkpoint_enabled': True,
            'checkpoint_path': 'checkpoints/adaboost_model.pkl',
            'load_existing_model': False
        }
    }

    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseTrainer], default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new model type with the factory.

        Args:
            name: Name identifier for the model type
            model_class: Class implementing BaseTrainer interface
            default_config: Default configuration for the model type
        """
        cls._model_registry[name] = model_class
        if default_config:
            cls._default_configs[name] = default_config
        logging.getLogger(__name__).info(f"Registered model type: {name}")

    @classmethod
    def create_trainer(cls, model_type: str, window_generator, config: Optional[Dict[str, Any]] = None) -> BaseTrainer:
        """
        Create a model trainer instance based on the specified type.

        Args:
            model_type: Type of model to create ('svm', 'tensorflow', 'adaboost')
            window_generator: WindowGenerator instance for data handling
            config: Configuration parameters (uses defaults if None)

        Returns:
            Instance of the requested model trainer

        Raises:
            ValueError: If model_type is not registered
        """
        logger = logging.getLogger(__name__)
        
        if model_type not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
        
        # Get the model class
        model_class = cls._model_registry[model_type]
        
        # Merge default config with provided config
        final_config = cls._default_configs.get(model_type, {}).copy()
        if config:
            final_config.update(config)
        
        logger.info(f"Creating {model_type} trainer with config: {final_config}")
        
        # Create and return the trainer instance
        try:
            return model_class(window_generator, **final_config)
        except Exception as e:
            logger.error(f"Error creating {model_type} trainer: {str(e)}")
            raise

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model types.

        Returns:
            List of registered model type names
        """
        return list(cls._model_registry.keys())

    @classmethod
    def get_default_config(cls, model_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a model type.

        Args:
            model_type: Type of model

        Returns:
            Default configuration dictionary

        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._default_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return cls._default_configs[model_type].copy()

    @classmethod
    def update_default_config(cls, model_type: str, config: Dict[str, Any]) -> None:
        """
        Update the default configuration for a model type.

        Args:
            model_type: Type of model
            config: New configuration parameters to merge with defaults

        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._default_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        cls._default_configs[model_type].update(config)
        logging.getLogger(__name__).info(f"Updated default config for {model_type}")
