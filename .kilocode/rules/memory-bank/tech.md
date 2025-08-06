# Technology Stack

## Core Technologies

### Programming Language

- **Python**: Primary development language for data science and machine learning

### Machine Learning & Data Science

- **TensorFlow**: Deep learning framework for time series modeling
  - Used for well-established time series architectures: LSTM, GRU, Transformer-based models, and CNN
  - Provides `tf.keras.utils.timeseries_dataset_from_array` for time series data preparation
  - Includes signal processing utilities (`tf.signal.rfft` for FFT analysis)
  - Supports scalable and reproducible framework development
- **Scikit-learn**: Machine learning library for traditional algorithms
  - `AdaBoostRegressor` for ensemble learning
  - `TimeSeriesSplit` for time series cross-validation
  - Model persistence with `joblib`
- **NumPy**: Numerical computing for array operations and mathematical functions
- **Pandas**: Data manipulation and analysis
  - DataFrame operations for time series data
  - CSV reading with compression support
  - DateTime handling and indexing

### Visualization & Analysis

- **Matplotlib**: Plotting and visualization
  - Time series plotting
  - Frequency domain visualization
  - Model prediction visualization
- **Seaborn**: Statistical data visualization (referenced in notebooks)

### Development Environment

- **Jupyter Notebooks**: Research and experimentation
  - Located in `src/jupyter_nb/`
  - Used for LSTM model prototyping and data analysis
- **VS Code**: Primary development environment
- **Git**: Version control

## Data Sources & Formats

- **Remote Dataset**: MJU (Maejo University) PV data
  - URL: `https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/`
  - Format: Compressed CSV (`.csv.gz`)
  - Resolution: 15-minute intervals
- **Local Caching**: TensorFlow's `get_file` utility for dataset caching

## Project Structure Technologies

### Documentation

- **LaTeX**: Academic paper documentation
  - Document class: `svjour3.cls` (Springer journal format)
  - Bibliography: BibTeX with custom style files
  - Multi-chapter structure for research paper

### Configuration Management

- **Kilo Code**: Custom rules and memory bank system
- **Git Ignore**: Standard Python project exclusions

### Project Organization

- **Pathlib**: Platform-independent path handling for project reorganization
- **Automated Structure**: Scripts for maintaining proper `src/` and `tests/` directory organization
- **Dry-run Capability**: Preview changes before executing reorganization

## Key Libraries and Dependencies

### Data Processing

```python
import pandas as pd           # Data manipulation
import numpy as np           # Numerical operations
import tensorflow as tf      # Machine learning framework
from sklearn.ensemble import AdaBoostRegressor  # Ensemble learning
from sklearn.model_selection import TimeSeriesSplit  # Time series CV
import joblib                 # Model persistence
```

### Visualization

```python
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns           # Statistical visualization
```

### System Integration

```python
import os                    # File system operations
import datetime             # Date/time handling
```

## Development Patterns

### Import Strategy

The project uses flexible imports to support both module and script execution:

```python
try:
    from .data_loader import DataLoader  # Module import
except ImportError:
    from data_loader import DataLoader   # Script import
```

### Data Pipeline Architecture

- **Modular Design**: Separate classes for each major functionality
- **Property-Based Access**: Lazy loading of datasets through Python properties
- **Configuration-Driven**: Constants defined at module level

### Model Training Infrastructure

- **Early Stopping**: Prevents overfitting with configurable patience
- **Standardized Metrics**: MSE loss with MAE metrics
- **Callback System**: TensorFlow callbacks for training control
- **Model Checkpointing**: Automatic saving/loading of best performing models
- **Cross-Validation**: Time series cross-validation with TimeSeriesSplit
- **Development Mode**: Fast training mode with reduced data and estimators

## File Format Support

- **CSV**: Plain text comma-separated values
- **CSV.gz**: Gzip-compressed CSV files
- **Automatic Detection**: Built-in gzip file detection
- **Remote Downloads**: HTTP/HTTPS dataset retrieval

## Time Series Specific Features

- **Windowing**: Custom `WindowGenerator` class for time series preparation
- **Cyclical Features**: Sine/cosine encoding for daily and yearly cycles
- **FFT Analysis**: Fast Fourier Transform for frequency domain analysis
- **Peak Detection**: Custom algorithm for identifying dominant frequencies

## Research Integration

- **Jupyter Integration**: Seamless notebook-to-production pipeline
- **LaTeX Documentation**: Academic paper generation alongside code
- **Reproducible Research**: Consistent data processing and model training

## Performance Considerations

- **Batch Processing**: TensorFlow dataset batching (batch_size=32)
- **Memory Efficiency**: Lazy dataset loading and property-based access
- **Caching**: Local dataset caching to avoid repeated downloads
- **Vectorized Operations**: NumPy and TensorFlow for efficient computation

## Development Setup Requirements

- Python 3.x environment
- TensorFlow installation
- Standard data science stack (pandas, numpy, matplotlib)
- Jupyter for notebook development
- LaTeX distribution for documentation compilation
- Git for version control

## Deployment Considerations

- **Modular Architecture**: Easy to deploy individual components
- **Configuration Management**: Environment-specific settings
- **Data Pipeline**: Automated data download and preprocessing
- **Model Persistence**: TensorFlow model saving/loading capabilities
- **Checkpointing System**: Robust model state management with joblib serialization
- **Development/Production Modes**: Configurable training modes for different environments
