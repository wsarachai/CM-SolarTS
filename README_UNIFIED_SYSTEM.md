# Unified Forecasting System

A comprehensive, modular forecasting system for photovoltaic (PV) power output prediction that consolidates multiple machine learning models into a single, cohesive framework.

## Overview

The Unified Forecasting System provides a standardized interface for training, evaluating, and comparing different forecasting models including Support Vector Machines (SVM), TensorFlow-based neural networks, and AdaBoost ensemble methods. The system is designed with modularity, extensibility, and ease of use in mind.

## Architecture

### Core Components

1. **BaseTrainer**: Abstract base class defining the common interface for all forecasting models
2. **Model Implementations**: Concrete implementations for SVM, TensorFlow, and AdaBoost
3. **ModelFactory**: Factory pattern implementation for creating model instances
4. **ForecastingSystem**: Main orchestrator that manages the entire forecasting pipeline
5. **CLI Interface**: Command-line interface for system interaction

### Key Features

- **Unified Interface**: All models implement the same BaseTrainer interface
- **Standardized Data Pipeline**: Consistent data handling across all models
- **Model Persistence**: Save/load functionality for all model types
- **Performance Metrics**: Comprehensive evaluation metrics (RMSE, MAE, MAPE, R²)
- **Development Mode**: Fast training mode for testing and development
- **Cross-Validation**: Time series cross-validation for robust model evaluation
- **Hyperparameter Optimization**: Grid and random search for SVM models
- **Configuration Management**: JSON-based configuration system
- **Comprehensive Logging**: Detailed logging throughout the system

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd python-prj
```

2. Install dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib joblib
```

3. Create necessary directories:

```bash
mkdir -p checkpoints results logs
```

## Usage

### Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
# Train a single model
python -m src.cli train --model svm --dev-mode

# Train multiple models
python -m src.cli train --model svm tensorflow adaboost

# Evaluate a model
python -m src.cli evaluate --model svm

# Compare multiple models
python -m src.cli compare --models svm tensorflow adaboost

# Make predictions
python -m src.cli predict --model svm --output predictions.csv

# Get system status
python -m src.cli status
```

### Python API

```python
from src.forecasting_system import ForecastingSystem

# Initialize the system
forecasting_system = ForecastingSystem('config.json')

# Initialize data
forecasting_system.initialize_data()

# Train models
svm_results = forecasting_system.train_model('svm', {'dev_mode': True})
tf_results = forecasting_system.train_model('tensorflow', {'max_epochs': 10})

# Evaluate models
svm_eval = forecasting_system.evaluate_model('svm')
tf_eval = forecasting_system.evaluate_model('tensorflow')

# Compare models
comparison = forecasting_system.compare_models(['svm', 'tensorflow'])

# Make predictions
predictions = forecasting_system.predict('svm')
```

### Configuration

The system uses JSON configuration files. Copy `config_template.json` to `config.json` and customize:

```json
{
  "data": {
    "dataset_host": "https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/",
    "dataset_file": "export_device_1_basic_aggregated_15minutes.csv.gz"
  },
  "window": {
    "input_width": 24,
    "label_width": 1,
    "shift": 1
  },
  "models": {
    "svm": {
      "enabled": true,
      "config": {
        "kernel": "rbf",
        "C": 1.0,
        "dev_mode": false
      }
    }
  }
}
```

## Model Types

### Support Vector Machine (SVM)

- **Implementation**: `SVMTrainer`
- **Features**: Hyperparameter optimization, feature scaling, cross-validation
- **Configuration**: Kernel type, C parameter, epsilon, gamma
- **Optimization**: Grid search and random search support

### TensorFlow Neural Networks

- **Implementation**: `TensorFlowTrainer`
- **Architecture**: LSTM-based time series model
- **Features**: Early stopping, model checkpointing, development mode
- **Configuration**: Max epochs, patience, learning rate

### AdaBoost Ensemble

- **Implementation**: `AdaBoostTrainer`
- **Features**: Time series cross-validation, development mode, checkpointing
- **Configuration**: Number of estimators, learning rate, differencing order
- **Special Features**: Handles time series differencing, feature importance analysis

## Development Mode

All models support development mode for faster training during testing:

```python
# Enable development mode
config = {'dev_mode': True}
results = forecasting_system.train_model('svm', config)
```

Development mode features:

- Reduced training data (10% sample for AdaBoost)
- Fewer estimators/epochs
- Faster hyperparameter search
- Maintained model quality for testing

## Model Persistence

All models support save/load functionality:

```python
# Save model
path = forecasting_system.save_model('svm')

# Load model
forecasting_system.load_model('svm', path)
```

Models are saved with metadata including:

- Training configuration
- Performance metrics
- Cross-validation results
- Timestamp information

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_unified_forecasting_system.py -v
```

Or run individual test components:

```bash
# Test base trainer interface
python -m unittest tests.test_unified_forecasting_system.TestBaseTrainerInterface

# Test model factory
python -m unittest tests.test_unified_forecasting_system.TestModelFactory

# Test forecasting system
python -m unittest tests.test_unified_forecasting_system.TestForecastingSystem
```

## Examples

### Basic Usage Example

```python
# See example_usage.py for a complete example
python example_usage.py
```

### Custom Model Registration

```python
from src.model_factory import ModelFactory
from src.base_trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    # Implement required methods
    pass

# Register custom model
ModelFactory.register_model('custom', CustomTrainer, default_config)

# Use custom model
trainer = ModelFactory.create_trainer('custom', window_generator)
```

## Extending the System

### Adding New Models

1. Create a new trainer class inheriting from `BaseTrainer`
2. Implement required methods: `fit()`, `predict()`, `score()`, `save_model()`, `load_model()`, `get_model_info()`
3. Register the model with `ModelFactory`
4. Add default configuration
5. Update tests and documentation

### Custom Metrics

```python
class CustomTrainer(BaseTrainer):
    def score(self, dataset=None):
        # Calculate standard metrics
        base_metrics = super().calculate_metrics(dataset)

        # Add custom metrics
        base_metrics['custom_metric'] = self.calculate_custom_metric(dataset)

        return base_metrics
```

## Logging

The system provides comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('forecasting.log')
    ]
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and the `src` directory is in your Python path
2. **Memory Issues**: Use development mode for large datasets during testing
3. **TensorFlow Warnings**: Set `TF_CPP_MIN_LOG_LEVEL=2` environment variable
4. **Model Loading Errors**: Check file paths and ensure models were saved correctly

### Performance Optimization

- Use development mode during testing
- Enable model checkpointing to avoid retraining
- Use appropriate batch sizes for TensorFlow models
- Consider feature scaling for SVM models

## Contributing

1. Follow the existing code structure and patterns
2. Implement the `BaseTrainer` interface for new models
3. Add comprehensive tests for new functionality
4. Update documentation and examples
5. Use consistent logging and error handling

## License

This project is part of the Python PV Forecasting Research Project.

## Contact

For questions and support, please refer to the project documentation or create an issue in the repository.
