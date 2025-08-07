import logging
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

from .data_loader import DataLoader
from .window_generator import WindowGenerator
from .model_factory import ModelFactory
from .base_trainer import BaseTrainer

class ForecastingSystem:
    """
    Main orchestrator for the unified forecasting system.

    This class manages the entire forecasting pipeline including data loading,
    preprocessing, model training, evaluation, and prediction.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the forecasting system.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize data components
        self.data_loader = None
        self.window_generator = None
        
        # Model management
        self.trainers: Dict[str, BaseTrainer] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        
        # System state
        self.is_initialized = False
        self.data_loaded = False
        self.models_trained = {}
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("ForecastingSystem initialized")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        default_config = {
            'data': {
                'dataset_host': 'https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/',
                'dataset_file': 'export_device_1_basic_aggregated_15minutes.csv.gz',
                'all_cols': [
                    'Grid Feed In', 'External Energy Supply', 'Internal Power Supply',
                    'Current Power', 'Self Consumption', 'Ambient Temperature',
                    'Module Temperature', 'Total Irradiation'
                ],
                'train_val_test_split': [0.7, 0.2, 0.1]
            },
            'window': {
                'input_width': 24,
                'label_width': 1,
                'shift': 1,
                'label_columns': ['Current Power']
            },
            'models': {
                'svm': {
                    'enabled': True,
                    'config': {
                        'kernel': 'rbf',
                        'C': 1.0,
                        'epsilon': 0.1,
                        'optimization_method': 'grid',
                        'optimization_params': {
                            'C': [0.1, 1.0, 10.0],
                            'epsilon': [0.01, 0.1, 0.2],
                            'gamma': ['scale', 'auto']
                        }
                    }
                },
                'tensorflow': {
                    'enabled': True,
                    'config': {
                        'max_epochs': 20,
                        'patience': 2
                    }
                },
                'adaboost': {
                    'enabled': True,
                    'config': {
                        'n_estimators': 50,
                        'learning_rate': 1.0
                    }
                }
            },
            'system': {
                'log_level': 'INFO',
                'results_dir': 'results',
                'checkpoint_dir': 'checkpoints'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with defaults
                def merge_dicts(default, user):
                    result = default.copy()
                    for key, value in user.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = merge_dicts(result[key], value)
                        else:
                            result[key] = value
                    return result
                
                default_config = merge_dicts(default_config, user_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading configuration: {str(e)}")
                self.logger.info("Using default configuration")
        
        return default_config

    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config['system'].get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def initialize_data_pipeline(self) -> None:
        """
        Initialize the data pipeline including data loading and window generation.
        """
        try:
            self.logger.info("Initializing data pipeline")
            
            # Initialize data loader
            data_config = self.config['data']
            self.data_loader = DataLoader(
                dataset_host=data_config['dataset_host'],
                dataset_file=data_config['dataset_file'],
                all_cols=data_config['all_cols']
            )
            
            # Load and preprocess data
            df = self.data_loader.download_and_load()
            df = self.data_loader.preprocess()
            train_df, val_df, test_df = self.data_loader.split_and_normalize()
            
            # Initialize window generator
            window_config = self.config['window']
            self.window_generator = WindowGenerator(
                input_width=window_config['input_width'],
                label_width=window_config['label_width'],
                shift=window_config['shift'],
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                label_columns=window_config['label_columns']
            )
            
            self.data_loaded = True
            self.logger.info("Data pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing data pipeline: {str(e)}")
            raise

    def initialize_models(self) -> None:
        """
        Initialize all enabled model trainers.
        """
        try:
            self.logger.info("Initializing model trainers")
            
            if not self.data_loaded:
                self.initialize_data_pipeline()
            
            # Create directories if they don't exist
            results_dir = self.config['system']['results_dir']
            checkpoint_dir = self.config['system']['checkpoint_dir']
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Initialize enabled models
            models_config = self.config['models']
            for model_type, model_info in models_config.items():
                if model_info.get('enabled', False):
                    model_config = model_info.get('config', {})
                    
                    # Set checkpoint path
                    checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_model.pkl")
                    model_config['checkpoint_path'] = checkpoint_path
                    
                    # Create trainer
                    trainer = ModelFactory.create_trainer(
                        model_type=model_type,
                        window_generator=self.window_generator,
                        config=model_config
                    )
                    
                    self.trainers[model_type] = trainer
                    self.model_configs[model_type] = model_config
                    
                    self.logger.info(f"Initialized {model_type} trainer")
            
            self.is_initialized = True
            self.logger.info("Model trainers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def train_models(self, model_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train specified models or all enabled models.

        Args:
            model_types: List of model types to train (trains all if None)

        Returns:
            Dictionary of training results for each model
        """
        try:
            if not self.is_initialized:
                self.initialize_models()
            
            results = {}
            
            # Determine which models to train
            if model_types is None:
                models_to_train = list(self.trainers.keys())
            else:
                models_to_train = [m for m in model_types if m in self.trainers]
            
            self.logger.info(f"Training models: {models_to_train}")
            
            # Train each model
            for model_type in models_to_train:
                trainer = self.trainers[model_type]
                
                try:
                    self.logger.info(f"Training {model_type} model")
                    training_result = trainer.fit()
                    
                    # Evaluate on validation set
                    val_score = trainer.score(self.window_generator.val)
                    training_result['val_score'] = val_score
                    
                    results[model_type] = training_result
                    self.models_trained[model_type] = True
                    
                    self.logger.info(f"{model_type} training completed. Validation score: {val_score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type}: {str(e)}")
                    results[model_type] = {'error': str(e)}
            
            # Save results
            self._save_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in train_models: {str(e)}")
            raise

    def evaluate_models(self, model_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate specified models or all trained models.

        Args:
            model_types: List of model types to evaluate (evaluates all if None)

        Returns:
            Dictionary of evaluation metrics for each model
        """
        try:
            if not self.is_initialized:
                self.initialize_models()
            
            results = {}
            
            # Determine which models to evaluate
            if model_types is None:
                models_to_evaluate = [m for m, trained in self.models_trained.items() if trained]
            else:
                models_to_evaluate = [m for m in model_types if m in self.models_trained and self.models_trained[m]]
            
            self.logger.info(f"Evaluating models: {models_to_evaluate}")
            
            # Evaluate each model
            for model_type in models_to_evaluate:
                trainer = self.trainers[model_type]
                
                try:
                    self.logger.info(f"Evaluating {model_type} model")
                    
                    # Calculate metrics
                    test_score = trainer.score(self.window_generator.test)
                    rmse = trainer.calculate_rmse(self.window_generator.test)
                    mae = trainer.calculate_mae(self.window_generator.test)
                    mape = trainer.calculate_mape(self.window_generator.test)
                    
                    results[model_type] = {
                        'test_score': test_score,
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape
                    }
                    
                    self.logger.info(f"{model_type} evaluation completed. Test score: {test_score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {model_type}: {str(e)}")
                    results[model_type] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in evaluate_models: {str(e)}")
            raise

    def compare_models(self, model_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare performance of specified models or all trained models.

        Args:
            model_types: List of model types to compare (compares all if None)

        Returns:
            DataFrame with comparison metrics
        """
        try:
            evaluation_results = self.evaluate_models(model_types)
            
            # Prepare comparison data
            comparison_data = []
            for model_type, metrics in evaluation_results.items():
                if 'error' not in metrics:
                    row = {
                        'Model': model_type,
                        'R² Score': metrics['test_score'],
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'MAPE': metrics['mape']
                    }
                    comparison_data.append(row)
            
            # Create DataFrame and sort by R² score
            comparison_df = pd.DataFrame(comparison_data)
            if not comparison_df.empty:
                comparison_df = comparison_df.sort_values('R² Score', ascending=False)
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error in compare_models: {str(e)}")
            raise

    def predict(self, model_type: str, steps: int = 1) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_type: Type of model to use for prediction
            steps: Number of steps to predict ahead

        Returns:
            Array of predictions
        """
        try:
            if not self.models_trained.get(model_type, False):
                raise ValueError(f"Model {model_type} is not trained")
            
            trainer = self.trainers[model_type]
            
            # For now, just predict on test set
            # This could be extended for multi-step prediction
            predictions = trainer.predict(self.window_generator.test)
            
            # Return the specified number of steps
            return predictions[:steps] if steps > 0 else predictions
            
        except Exception as e:
            self.logger.error(f"Error in predict: {str(e)}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the forecasting system.

        Returns:
            Dictionary with system status information
        """
        status = {
            'initialized': self.is_initialized,
            'data_loaded': self.data_loaded,
            'available_models': list(self.trainers.keys()),
            'trained_models': [m for m, trained in self.models_trained.items() if trained],
            'config': self.config
        }
        
        # Add model-specific information
        model_info = {}
        for model_type, trainer in self.trainers.items():
            model_info[model_type] = trainer.get_model_info()
        
        status['model_info'] = model_info
        
        return status

    def _save_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save results to file.

        Args:
            results: Results dictionary to save
        """
        try:
            results_dir = self.config['system']['results_dir']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_path = os.path.join(results_dir, f"training_results_{timestamp}.json")
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_numpy(results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")