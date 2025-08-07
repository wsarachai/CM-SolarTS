# Unified Forecasting System Design

## Overview

This document outlines the design for a comprehensive forecasting system that integrates Support Vector Machine (SVM) models within the existing model_trainer.py framework, providing a unified interface for multiple forecasting algorithms.

## Architecture Components

### 1. BaseTrainer (Abstract Interface)

**File**: `src/base_trainer.py`

```python
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
    def score(self, dataset: Optional[Any] = None) -> float:
        """Calculate the primary evaluation metric (R² score)."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Save the trained model to disk."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load a trained model from disk."""
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        pass

    @abstractmethod
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set hyperparameters."""
        pass

    # Common methods implemented in base class
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
        return np.mean(np.abs((actual - predictions) / actual)) * 100

    @abstractmethod
    def _extract_actual_values(self, dataset: Optional[Any] = None) -> np.ndarray:
        """Extract actual values from dataset for metric calculations."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'hyperparameters': self.get_hyperparameters(),
            'training_history': self.training_history
        }
```

### 2. SVMTrainer Implementation

**File**: `src/svm_trainer.py`

[Previous SVMTrainer implementation - truncated for brevity]

### 3. ModelFactory

**File**: `src/model_factory.py`

[Previous ModelFactory implementation - truncated for brevity]

### 4. ForecastingSystem (Main Orchestrator)

**File**: `src/forecasting_system.py`

```python
import logging
import json
import os
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

try:
    from .model_factory import ModelFactory
    from .base_trainer import BaseTrainer
    from .data_loader import DataLoader
    from .window_generator import WindowGenerator
except ImportError:
    from model_factory import ModelFactory
    from base_trainer import BaseTrainer
    from data_loader import DataLoader
    from window_generator import WindowGenerator

class ForecastingSystem:
    """
    Main orchestrator for the unified forecasting system.

    This class provides a high-level interface for training and using
    different forecasting models with consistent data preprocessing
    and evaluation.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        log_level: str = 'INFO'
    ):
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.config = {}
        self.data_loader = None
        self.window_generator = None
        self.trainers = {}
        self.evaluation_results = {}

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def get_model_info(self, trainer_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if trainer_name not in self.trainers:
            raise ValueError(f"Trainer {trainer_name} not found")

        trainer = self.trainers[trainer_name]
        return trainer.get_model_info()

    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return ModelFactory.get_available_models()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'data_pipeline_ready': self.window_generator is not None,
            'available_models': self.get_available_models(),
            'registered_trainers': list(self.trainers.keys()),
            'trained_models': [name for name, trainer in self.trainers.items() if trainer.is_trained],
            'evaluation_results': self.evaluation_results
        }
```

### 5. Command-Line Interface

**File**: `src/forecast_cli.py`

```python
import argparse
import json
import sys
import os
from typing import Dict, Any

try:
    from .forecasting_system import ForecastingSystem
except ImportError:
    from forecasting_system import ForecastingSystem

class ForecastCLI:
    """
    Command-line interface for the unified forecasting system.

    Provides easy access to training, prediction, and evaluation
    functionality through command-line arguments.
    """

    def __init__(self):
        self.system = None
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description='Unified Forecasting System CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Train an SVM model
  python forecast_cli.py train --model svm --name my_svm --config config.json

  # Train multiple models
  python forecast_cli.py train --model svm,adaboost --config config.json

  # Evaluate a trained model
  python forecast_cli.py evaluate --name my_svm --dataset test

  # Compare multiple models
  python forecast_cli.py compare --names my_svm,my_adaboost

  # Make predictions
  python forecast_cli.py predict --name my_svm --output predictions.csv
            """
        )

        # Global arguments
        parser.add_argument('--config', type=str, default='config.json',
                          help='Configuration file path')
        parser.add_argument('--log-level', type=str, default='INFO',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Logging level')

        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Train command
        train_parser = subparsers.add_parser('train', help='Train models')
        train_parser.add_argument('--model', type=str, required=True,
                                help='Model type(s) to train (comma-separated)')
        train_parser.add_argument('--name', type=str,
                                help='Custom trainer name (for single model)')
        train_parser.add_argument('--dev-mode', action='store_true',
                                help='Enable development mode for faster training')

        # Evaluate command
        eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
        eval_parser.add_argument('--name', type=str, required=True,
                               help='Trainer name to evaluate')
        eval_parser.add_argument('--dataset', type=str, default='test',
                               choices=['train', 'val', 'test'],
                               help='Dataset to evaluate on')

        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare models')
        compare_parser.add_argument('--names', type=str,
                                  help='Trainer names to compare (comma-separated)')
        compare_parser.add_argument('--dataset', type=str, default='test',
                                  choices=['train', 'val', 'test'],
                                  help='Dataset to compare on')
        compare_parser.add_argument('--output', type=str,
                                  help='Output file for comparison results')

        # Predict command
        predict_parser = subparsers.add_parser('predict', help='Make predictions')
        predict_parser.add_argument('--name', type=str, required=True,
                                  help='Trainer name to use for prediction')
        predict_parser.add_argument('--dataset', type=str, default='test',
                                  choices=['train', 'val', 'test'],
                                  help='Dataset to predict on')
        predict_parser.add_argument('--output', type=str,
                                  help='Output file for predictions')

        # Status command
        subparsers.add_parser('status', help='Show system status')

        # List command
        subparsers.add_parser('list', help='List available models')

        return parser

    def _initialize_system(self, config_path: str, log_level: str) -> None:
        """Initialize the forecasting system."""
        self.system = ForecastingSystem(config_path, log_level)

        # Setup data pipeline if configuration exists
        if self.system.config:
            data_config = self.system.config.get('data', {})
            if data_config:
                self.system.setup_data_pipeline(**data_config)

    def _parse_model_list(self, model_str: str) -> list:
        """Parse comma-separated model list."""
        return [model.strip() for model in model_str.split(',')]

    def _parse_name_list(self, name_str: str) -> list:
        """Parse comma-separated name list."""
        return [name.strip() for name in name_str.split(',')]

    def train_command(self, args) -> None:
        """Handle train command."""
        models = self._parse_model_list(args.model)

        for i, model_type in enumerate(models):
            # Generate trainer name
            if args.name and len(models) == 1:
                trainer_name = args.name
            else:
                trainer_name = f"{model_type}_trainer_{i+1}"

            # Get model-specific configuration
            model_config = self.system.config.get('models', {}).get(model_type, {})

            # Add development mode if specified
            if args.dev_mode:
                model_config['dev_mode'] = True

            try:
                # Create and train model
                print(f"Creating {model_type} trainer: {trainer_name}")
                trainer = self.system.create_trainer(model_type, trainer_name, **model_config)

                print(f"Training {trainer_name}...")
                history = self.system.train_model(trainer_name)

                print(f"Training completed for {trainer_name}")
                print(f"Training history: {history}")

            except Exception as e:
                print(f"Error training {trainer_name}: {e}")
                sys.exit(1)

    def evaluate_command(self, args) -> None:
        """Handle evaluate command."""
        try:
            metrics = self.system.evaluate_model(args.name, args.dataset)

            print(f"Evaluation results for {args.name} on {args.dataset} set:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        except Exception as e:
            print(f"Error evaluating {args.name}: {e}")
            sys.exit(1)

    def compare_command(self, args) -> None:
        """Handle compare command."""
        try:
            names = self._parse_name_list(args.names) if args.names else None
            comparison_df = self.system.compare_models(names, args.dataset)

            print(f"Model comparison on {args.dataset} set:")
            print(comparison_df.to_string())

            if args.output:
                comparison_df.to_csv(args.output)
                print(f"Results saved to {args.output}")

        except Exception as e:
            print(f"Error comparing models: {e}")
            sys.exit(1)

    def predict_command(self, args) -> None:
        """Handle predict command."""
        try:
            predictions = self.system.predict(args.name, args.dataset)

            print(f"Generated {len(predictions)} predictions using {args.name}")
            print(f"Prediction statistics:")
            print(f"  Mean: {np.mean(predictions):.4f}")
            print(f"  Std: {np.std(predictions):.4f}")
            print(f"  Min: {np.min(predictions):.4f}")
            print(f"  Max: {np.max(predictions):.4f}")

            if args.output:
                import pandas as pd
                pred_df = pd.DataFrame({'predictions': predictions})
                pred_df.to_csv(args.output, index=False)
                print(f"Predictions saved to {args.output}")

        except Exception as e:
            print(f"Error making predictions: {e}")
            sys.exit(1)

    def status_command(self, args) -> None:
        """Handle status command."""
        status = self.system.get_system_status()

        print("System Status:")
        print(f"  Data pipeline ready: {status['data_pipeline_ready']}")
        print(f"  Available models: {', '.join(status['available_models'])}")
        print(f"  Registered trainers: {', '.join(status['registered_trainers'])}")
        print(f"  Trained models: {', '.join(status['trained_models'])}")

        if status['evaluation_results']:
            print("  Evaluation results available for:")
            for name in status['evaluation_results']:
                print(f"    - {name}")

    def list_command(self, args) -> None:
        """Handle list command."""
        available_models = self.system.get_available_models()

        print("Available model types:")
        for model in available_models:
            print(f"  - {model}")

    def run(self, args=None) -> None:
        """Run the CLI with given arguments."""
        parsed_args = self.parser.parse_args(args)

        if not parsed_args.command:
            self.parser.print_help()
            return

        # Initialize system
        self._initialize_system(parsed_args.config, parsed_args.log_level)

        # Execute command
        if parsed_args.command == 'train':
            self.train_command(parsed_args)
        elif parsed_args.command == 'evaluate':
            self.evaluate_command(parsed_args)
        elif parsed_args.command == 'compare':
            self.compare_command(parsed_args)
        elif parsed_args.command == 'predict':
            self.predict_command(parsed_args)
        elif parsed_args.command == 'status':
            self.status_command(parsed_args)
        elif parsed_args.command == 'list':
            self.list_command(parsed_args)

def main():
    """Main entry point for CLI."""
    cli = ForecastCLI()
    cli.run()

if __name__ == '__main__':
    main()
```

### 6. Configuration System

**File**: `config.json` (Example Configuration)

```json
{
  "data": {
    "dataset_host": "https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/",
    "dataset_file": "export_device_1_basic_aggregated_15minutes.csv.gz",
    "all_cols": [
      "Grid Feed In",
      "External Energy Supply",
      "Internal Power Supply",
      "Current Power",
      "Self Consumption",
      "Ambient Temperature",
      "Module Temperature",
      "Total Irradiation"
    ],
    "input_width": 24,
    "label_width": 1,
    "shift": 1,
    "label_columns": ["Current Power"]
  },
  "models": {
    "svm": {
      "kernel": "rbf",
      "C": 1.0,
      "epsilon": 0.1,
      "gamma": "scale",
      "hyperparameter_optimization": true,
      "optimization_method": "grid",
      "cv_folds": 5,
      "dev_mode": false,
      "checkpoint_enabled": true,
      "checkpoint_path": "models/svm_model.pkl"
    },
    "adaboost": {
      "n_estimators": 50,
      "learning_rate": 1.0,
      "dev_mode": false,
      "checkpoint_enabled": true,
      "checkpoint_path": "models/adaboost_model.pkl"
    },
    "tensorflow": {
      "max_epochs": 20,
      "patience": 2,
      "learning_rate": 0.001,
      "batch_size": 32,
      "checkpoint_enabled": true,
      "checkpoint_path": "models/tensorflow_model"
    }
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

### 7. Enhanced Main Script

**File**: `src/unified_main.py`

```python
"""
Unified main script demonstrating the forecasting system capabilities.

This script shows how to use the unified forecasting system to train
and evaluate multiple models with consistent preprocessing and evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from .forecasting_system import ForecastingSystem
except ImportError:
    from forecasting_system import ForecastingSystem

def main():
    """Main demonstration of the unified forecasting system."""

    # Initialize the forecasting system
    print("Initializing Unified Forecasting System...")
    system = ForecastingSystem(log_level='INFO')

    # Setup data pipeline
    print("Setting up data pipeline...")
    system.setup_data_pipeline(
        dataset_host='https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/',
        dataset_file='export_device_1_basic_aggregated_15minutes.csv.gz',
        all_cols=[
            'Grid Feed In', 'External Energy Supply', 'Internal Power Supply',
            'Current Power', 'Self Consumption', 'Ambient Temperature',
            'Module Temperature', 'Total Irradiation'
        ],
        input_width=24,
        label_width=1,
        shift=1,
        label_columns=['Current Power']
    )

    # Create and train multiple models
    models_to_train = [
        ('svm', 'svm_model', {
            'hyperparameter_optimization': True,
            'dev_mode': False,  # Set to True for faster development
            'checkpoint_path': 'models/svm_model.pkl'
        }),
        ('adaboost', 'adaboost_model', {
            'n_estimators': 50,
            'dev_mode': False,  # Set to True for faster development
            'checkpoint_path': 'models/adaboost_model.pkl'
        })
    ]

    trained_models = []

    for model_type, model_name, model_config in models_to_train:
        try:
            print(f"\n{'='*50}")
            print(f"Training {model_type.upper()} Model: {model_name}")
            print(f"{'='*50}")

            # Create trainer
            trainer = system.create_trainer(model_type, model_name, **model_config)

            # Train model
            training_history = system.train_model(model_name)
            print(f"Training completed for {model_name}")

            trained_models.append(model_name)

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    if not trained_models:
        print("No models were successfully trained!")
        return

    # Evaluate all trained models
    print(f"\n{'='*50}")
    print("Model Evaluation")
    print(f"{'='*50}")

    for model_name in trained_models:
        try:
            metrics = system.evaluate_model(model_name, 'test')
            print(f"\n{model_name} Test Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    # Compare models
    if len(trained_models) > 1:
        print(f"\n{'='*50}")
        print("Model Comparison")
        print(f"{'='*50}")

        try:
            comparison_df = system.compare_models(trained_models, 'test')
            print("\nModel Performance Comparison (Test Set):")
            print(comparison_df.to_string())

            # Save comparison results
            os.makedirs('results', exist_ok=True)
            comparison_df.to_csv('results/model_comparison.csv')
            print("\nComparison results saved to results/model_comparison.csv")

        except Exception as e:
            print(f"Error comparing models: {e}")

    # Generate predictions and visualizations
    print(f"\n{'='*50}")
    print("Generating Predictions and Visualizations")
    print(f"{'='*50}")

    # Create visualization for the best performing model
    best_model = trained_models[0]  # Assuming first model or implement best selection

    try:
        # Generate predictions
        predictions = system.predict(best_model, 'test')

        # Get actual values for comparison
        trainer = system.trainers[best_model]
        actual_values = trainer._extract_actual_values(system.window_generator.test)

        # Create visualization
        plt.figure(figsize=(15, 10))

        # Plot 1: Full comparison
        plt.subplot(2, 1, 1)
        plt.plot(actual_values, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title(f'{best_model} - Full Test Set Comparison')
        plt.xlabel('Time Steps')
        plt.ylabel('Current Power (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Last 100 values for detail
        plt.subplot(2, 1, 2)
        n_detail = min(100, len(actual_values))
        plt.plot(actual_values[-n_detail:], label='Actual', marker='o', markersize=3)
        plt.plot(predictions[-n_detail:], label='Predicted', marker='s', markersize=3)
        plt.title(f'{best_model} - Last {n_detail} Values (Detailed View)')
        plt.xlabel('Time Steps')
        plt.ylabel('Current Power (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{best_model}_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to results/{best_model}_predictions.png")

        # Show plot
        plt.show()

    except Exception as e:
        print(f"Error generating visualizations: {e}")

    # System status
    print(f"\n{'='*50}")
    print("Final System Status")
    print(f"{'='*50}")

    status = system.get_system_status()
    print(f"Data pipeline ready: {status['data_pipeline_ready']}")
    print(f"Available models: {', '.join(status['available_models'])}")
    print(f"Trained models: {', '.join(status['trained_models'])}")

    print("\nUnified Forecasting System demonstration completed!")

if __name__ == "__main__":
    main()
```

## Implementation Plan

### Phase 1: Core Infrastructure

1. Implement `BaseTrainer` abstract class
2. Create `ModelFactory` for model instantiation
3. Implement `SVMTrainer` with hyperparameter optimization
4. Create enhanced wrappers for existing trainers

### Phase 2: System Integration

1. Implement `ForecastingSystem` orchestrator
2. Add configuration management
3. Implement comprehensive error handling and logging
4. Create model persistence system

### Phase 3: User Interface

1. Implement command-line interface
2. Add configuration file support
3. Create example scripts and documentation
4. Add visualization capabilities

### Phase 4: Testing and Validation

1. Create comprehensive test suite
2. Validate model performance consistency
3. Test error handling and edge cases
4. Performance benchmarking

## Usage Examples

### Basic Usage

```python
from forecasting_system import ForecastingSystem

# Initialize system
system = ForecastingSystem('config.json')

# Setup data pipeline
system.setup_data_pipeline(...)

# Create and train SVM model
svm_trainer = system.create_trainer('svm', 'my_svm',
                                   hyperparameter_optimization=True)
system.train_model('my_svm')

# Evaluate model
metrics = system.evaluate_model('my_svm', 'test')
print(f"R² Score: {metrics['r2_score']:.4f}")
```

### Command Line Usage

```bash
# Train multiple models
python forecast_cli.py train --model svm,adaboost --config config.json

# Compare models
python forecast_cli.py compare --names svm_trainer_1,adaboost_trainer_1

# Make predictions
python forecast_cli.py predict --name svm_trainer_1 --output predictions.csv
```

## Benefits

1. **Unified Interface**: Consistent API across all model types
2. **Easy Extension**: Simple to add new forecasting algorithms
3. **Robust Evaluation**: Standardized metrics and cross-validation
4. **Configuration-Driven**: Easy to modify parameters without code changes
5. **Production Ready**: Comprehensive error handling and logging
6. **CLI Support**: Easy integration into automated workflows
7. **Model Persistence**: Save and load trained models reliably
8. **Development Mode**: Fast iteration during development and testing

This design provides a comprehensive, extensible, and user-friendly forecasting system that integrates SVM models while maintaining compatibility with existing TensorFlow and AdaBoost implementations.
