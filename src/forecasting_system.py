import logging
import json
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from .data_loader import DataLoader
    from .frequency_analyzer import FrequencyAnalyzer
    from .window_generator import WindowGenerator
    from .model_trainer import ModelTrainer
    from .adaboost_trainer import AdaBoostTrainer
    from .svm_trainer import SVMTrainer
    from .model_factory import ModelFactory
    from .base_trainer import BaseTrainer
except ImportError:
    # Fallback for running as a script directly (not as a module)
    from data_loader import DataLoader
    from frequency_analyzer import FrequencyAnalyzer
    from window_generator import WindowGenerator
    from model_trainer import ModelTrainer
    from adaboost_trainer import AdaBoostTrainer
    from svm_trainer import SVMTrainer
    from model_factory import ModelFactory
    from base_trainer import BaseTrainer

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
        self.freq_analyzer = None
        
        # Model management
        self.trainers: Dict[str, BaseTrainer] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        
        # System state
        self.is_initialized = False
        self.data_loaded = False
        self.models_trained = {}
        
        # Data analysis results
        self.frequency_analysis_results = None
        self.dataset_info = None
        
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
        # Determine the base directory of the project
        # This assumes forecasting_system.py is in the 'src' directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(base_dir, 'config.json')

        try:
            with open(default_config_path, 'r') as f:
                default_config = json.load(f)
            self.logger.info(f"Loaded default configuration from {default_config_path}")
        except Exception as e:
            self.logger.error(f"FATAL: Could not load default configuration file: {str(e)}")
            raise  # Re-raise the exception as this is a critical failure

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
            
            # Store dataset information
            self.dataset_info = {
                'total_rows': df.shape[0],
                'columns': list(df.columns),
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                },
                'splits': {
                    'train': len(train_df),
                    'val': len(val_df),
                    'test': len(test_df)
                }
            }
            
            # Display dataset information
            print(f"Total rows: {df.shape[0]}")
            if pd.api.types.is_datetime64_any_dtype(df.index):
                print(df.index.year.unique())
                print(df.index.year.value_counts())
            else:
                print("Warning: DataFrame index is not a DatetimeIndex.")
            
            # Perform frequency analysis if enabled
            freq_config = self.config.get('frequency_analysis', {})
            if freq_config.get('enabled', True):
                self._perform_frequency_analysis(df, freq_config)
            
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

    def _perform_frequency_analysis(self, df: pd.DataFrame, freq_config: Dict[str, Any]) -> None:
        """
        Perform frequency analysis on the specified target column.
        
        Args:
            df: DataFrame to analyze
            freq_config: Frequency analysis configuration
        """
        try:
            target_column = freq_config.get('target_column', 'Current Power')
            sample_period = freq_config.get('sample_period', 900)  # 15 minutes in seconds
            
            if target_column not in df.columns:
                self.logger.warning(f"Target column '{target_column}' not found in dataframe.")
                return
            
            # Handle missing values for FFT
            current_power = df[target_column].fillna(0).to_numpy()
            
            self.freq_analyzer = FrequencyAnalyzer(current_power, sample_period)
            peak_frequencies, peak_periods = self.freq_analyzer.analyze()
            
            # Store results
            self.frequency_analysis_results = {
                'peak_frequencies': peak_frequencies,
                'peak_periods': peak_periods,
                'target_column': target_column,
                'sample_period': sample_period
            }
            
            # Display results
            print("Top 5 frequency peaks:")
            for i, (freq, period) in enumerate(zip(peak_frequencies[:5], peak_periods[:5])):
                print(f"Peak {i+1}: Frequency = {freq:.4f} cycles/year")
                days = period * 365.25
                if days < 1:
                    print(f"    Period ≈ {days*24:.2f} hours")
                elif days < 30:
                    print(f"    Period ~ {days:.2f} days")
                else:
                    print(f"    Period ~ {period*12:.2f} months")
            
            self.logger.info("Frequency analysis completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in frequency analysis: {str(e)}")

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
                    
                    # Create trainer using factory or direct instantiation
                    try:
                        trainer = ModelFactory.create_trainer(
                            model_type=model_type,
                            window_generator=self.window_generator,
                            config=model_config
                        )
                    except (ImportError, AttributeError):
                        # Fallback to direct instantiation if ModelFactory is not available
                        trainer = self._create_trainer_direct(model_type, model_config)
                    
                    self.trainers[model_type] = trainer
                    self.model_configs[model_type] = model_config
                    
                    self.logger.info(f"Initialized {model_type} trainer")
            
            self.is_initialized = True
            self.logger.info("Model trainers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def _create_trainer_direct(self, model_type: str, config: Dict[str, Any]) -> BaseTrainer:
        """
        Create trainer directly without ModelFactory (fallback method).
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            
        Returns:
            Initialized trainer instance
        """
        if model_type == 'adaboost':
            return AdaBoostTrainer(self.window_generator, **config)
        elif model_type == 'svm':
            return SVMTrainer(self.window_generator, **config)
        elif model_type == 'tensorflow':
            return ModelTrainer(max_epochs=config.get('max_epochs', 20))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

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
            
            print("\n" + "="*60)
            print("STARTING MODEL TRAINING AND EVALUATION")
            print("="*60)
            
            # Train each model
            for i, model_type in enumerate(models_to_train, 1):
                trainer = self.trainers[model_type]
                
                print(f"\n{i}. {model_type.upper()} TRAINING")
                print("-" * 30)
                
                try:
                    self.logger.info(f"Training {model_type} model")
                    
                    # Display configuration
                    if hasattr(trainer, 'kernel'):
                        print(f"SVM Configuration: kernel={trainer.kernel}, C={trainer.C}, epsilon={trainer.epsilon}")
                    elif hasattr(trainer, 'is_dev_mode'):
                        print(f"Training mode: {'Development' if trainer.is_dev_mode() else 'Production'}")
                    
                    # Train the model
                    if model_type == 'adaboost':
                        training_result = trainer.fit(n_splits=3)  # Reduced splits for faster testing
                    else:
                        training_result = trainer.fit()
                    
                    print(f"Training completed successfully!")
                    
                    # Display training results
                    if 'cv_mean' in training_result:
                        print(f"Cross-validation results: Mean R2 = {training_result['cv_mean']:.4f} (+/-{training_result['cv_std']:.4f})")
                    elif 'train_r2' in training_result:
                        print(f"Training metrics: R2 = {training_result.get('train_r2', 'N/A'):.4f}")
                    
                    # Make predictions and calculate metrics
                    predictions = trainer.predict()
                    metrics = trainer.score()
                    
                    print(f"{model_type.capitalize()} Test Metrics:")
                    print(f"  R2 Score: {metrics['r2']:.4f}")
                    print(f"  RMSE: {metrics['rmse']:.4f}")
                    print(f"  MAE: {metrics['mae']:.4f}")
                    print(f"  MAPE: {metrics['mape']:.2f}%")
                    
                    # Store comprehensive results
                    training_result.update({
                        'predictions': predictions,
                        'test_metrics': metrics,
                        'model_type': model_type
                    })
                    
                    results[model_type] = training_result
                    self.models_trained[model_type] = True
                    
                    self.logger.info(f"{model_type} training completed. Test R2: {metrics['r2']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type}: {str(e)}")
                    print(f"{model_type.capitalize()} training failed: {str(e)}")
                    results[model_type] = {'error': str(e)}
            
            # Model comparison
            self._display_model_comparison(results)
            
            # Visualization
            if self.config['system'].get('enable_visualization', True):
                self._create_visualization(results)
            
            # Save results
            self._save_results(results)
            
            print("\n" + "="*60)
            print("MODEL TRAINING AND EVALUATION COMPLETED")
            print("="*60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in train_models: {str(e)}")
            raise

    def _display_model_comparison(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Display model comparison results.
        
        Args:
            results: Training results for all models
        """
        print("\n3. MODEL COMPARISON")
        print("-" * 30)
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v and 'test_metrics' in v}
        
        if len(successful_results) >= 2:
            print("Performance Comparison:")
            print(f"{'Metric':<10} ", end="")
            for model_type in successful_results.keys():
                print(f"{model_type.capitalize():<12} ", end="")
            print()
            print("-" * (10 + 12 * len(successful_results)))
            
            metrics_to_compare = ['r2', 'rmse', 'mae', 'mape']
            for metric in metrics_to_compare:
                print(f"{metric.upper():<10} ", end="")
                for model_type in successful_results.keys():
                    value = successful_results[model_type]['test_metrics'][metric]
                    if metric == 'mape':
                        print(f"{value:<12.2f} ", end="")
                    else:
                        print(f"{value:<12.4f} ", end="")
                print()
            
            # Determine best model
            best_model = max(successful_results.keys(), 
                           key=lambda x: successful_results[x]['test_metrics']['r2'])
            print(f"\nBest performing model: {best_model.capitalize()}")
        elif len(successful_results) == 1:
            model_type = list(successful_results.keys())[0]
            print(f"Only {model_type.capitalize()} completed successfully")
        else:
            print("No models completed successfully")

    def _create_visualization(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create visualization of model predictions.
        
        Args:
            results: Training results for all models
        """
        print("\n4. VISUALIZATION")
        print("-" * 30)
        
        try:
            import matplotlib.pyplot as plt
            
            # Filter successful results with predictions
            successful_results = {k: v for k, v in results.items() 
                                if 'error' not in v and 'predictions' in v}
            
            if successful_results:
                # Get actual values for comparison
                test_ds = self.window_generator.test
                
                # Try to get actual values from any available trainer
                y_test = None
                for model_type, result in successful_results.items():
                    trainer = self.trainers[model_type]
                    try:
                        if hasattr(trainer, '_extract_xy'):
                            X_test, y_test = trainer._extract_xy(test_ds)
                            if hasattr(trainer, 'differencing_order') and trainer.differencing_order > 0:
                                y_test = y_test[trainer.differencing_order:]
                        elif hasattr(trainer, '_extract_actual_values'):
                            y_test = trainer._extract_actual_values(test_ds)
                        break
                    except:
                        continue
                
                if y_test is not None:
                    # Plot comparison
                    plt.figure(figsize=(15, 8))
                    
                    # Plot actual values
                    plot_range = min(100, len(y_test))
                    plt.plot(y_test[-plot_range:], label='Actual', linewidth=2, alpha=0.8)
                    
                    # Plot predictions for each model
                    colors = ['red', 'blue', 'green', 'orange']
                    for i, (model_type, result) in enumerate(successful_results.items()):
                        predictions = result['predictions']
                        r2_score = result['test_metrics']['r2']
                        plt.plot(predictions[-plot_range:], 
                                label=f'{model_type.capitalize()} (R2={r2_score:.3f})', 
                                linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])
                    
                    plt.xlabel('Time Steps')
                    plt.ylabel('Current Power')
                    plt.title(f'Model Predictions Comparison (Last {plot_range} Values)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"Visualization completed for {len(successful_results)} model(s)")
                else:
                    print("Could not extract actual values for visualization")
            else:
                print("No models available for visualization")
                
        except Exception as e:
            print(f"Visualization failed: {str(e)}")
            self.logger.error(f"Visualization error: {str(e)}")

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
                        'R2 Score': metrics['test_score'],
                        'RMSE': metrics['rmse'],
                        'MAE': metrics['mae'],
                        'MAPE': metrics['mape']
                    }
                    comparison_data.append(row)
            
            # Create DataFrame and sort by R² score
            comparison_df = pd.DataFrame(comparison_data)
            if not comparison_df.empty:
                comparison_df = comparison_df.sort_values('R2 Score', ascending=False)
            
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
            'config': self.config,
            'dataset_info': self.dataset_info,
            'frequency_analysis': self.frequency_analysis_results
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

    def run_full_pipeline(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline from data loading to model evaluation.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Dictionary containing all results
        """
        try:
            # Initialize system if config provided
            if config_path:
                self.config = self._load_config(config_path)
            
            # Run complete pipeline
            self.initialize_data_pipeline()
            self.initialize_models()
            training_results = self.train_models()
            
            # Return comprehensive results
            return {
                'system_status': self.get_system_status(),
                'training_results': training_results,
                'dataset_info': self.dataset_info,
                'frequency_analysis': self.frequency_analysis_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in full pipeline: {str(e)}")
            raise