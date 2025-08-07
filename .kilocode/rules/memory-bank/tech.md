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

### Climate Data Processing

- **netCDF4**: NetCDF file format handling
  - Reading and writing NetCDF files
  - Metadata extraction and variable analysis
  - Integration with NumPy arrays
- **xarray**: Multi-dimensional labeled data arrays
  - GRIB file processing with cfgrib engine
  - NetCDF data manipulation and analysis
  - Coordinate-based data selection
- **pygrib**: GRIB file format support
  - Alternative GRIB file reading library
  - Message-based GRIB data access
  - Metadata and key extraction
- **cdsapi**: Copernicus Climate Data Store API
  - Automated climate data download
  - CAMS dataset access
  - Authentication and request management

### Visualization & Analysis

- **Matplotlib**: Plotting and visualization
  - Time series plotting
  - Frequency domain visualization
  - Model prediction visualization
  - Global climate data mapping
  - Contour plots and colormaps
- **Seaborn**: Statistical data visualization (referenced in notebooks)
- **mpldatacursor**: Interactive data cursor for matplotlib plots
  - Used in CAMS data analysis for interactive exploration

### Vector Database & Semantic Search

- **Qdrant**: Vector database for semantic search
  - Docker-based deployment
  - HTTP and gRPC API support
  - Cosine similarity distance metric
  - Collection management for code indexing
- **Docker**: Containerization platform
  - Qdrant service deployment
  - Isolated development environment
  - Service orchestration with docker-compose
- **requests**: HTTP client library
  - Qdrant API integration
  - REST API communication
  - JSON data handling

### Development Environment

- **Jupyter Notebooks**: Research and experimentation
  - Located in `src/jupyter_nb/`
  - Used for LSTM model prototyping and data analysis
- **VS Code**: Primary development environment
- **Git**: Version control

## Data Sources & Formats

### Original PV Data

- **Remote Dataset**: MJU (Maejo University) PV data
  - URL: `https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/`
  - Format: Compressed CSV (`.csv.gz`)
  - Resolution: 15-minute intervals
- **Local Caching**: TensorFlow's `get_file` utility for dataset caching

### Climate Data Sources

- **CAMS Gridded Solar Radiation**: Global horizontal, direct, and diffuse irradiation
- **CAMS Global Radiative Forcings**: CO2 instantaneous longwave radiative forcing
- **CAMS MSG+HIMAWARI**: High-resolution satellite-based irradiation data
- **NetCDF Format**: Network Common Data Form for scientific data
- **GRIB Format**: Gridded Binary format for meteorological data

## Project Structure Technologies

### Documentation

- **LaTeX**: Academic paper documentation
  - Document class: `svjour3.cls` (Springer journal format)
  - Bibliography: BibTeX with custom style files
  - Multi-chapter structure for research paper
- **Markdown**: Documentation and analysis summaries
  - Data analysis summaries
  - Setup documentation
  - Memory bank files

### Configuration Management

- **Kilo Code**: Custom rules and memory bank system
- **Git Ignore**: Standard Python project exclusions
- **Docker Compose**: Service orchestration
  - YAML configuration for Qdrant service
  - Volume management for persistent storage

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

### Climate Data Processing

```python
import netCDF4 as nc         # NetCDF file handling
import xarray as xr          # Multi-dimensional data arrays
import pygrib                # GRIB file processing (alternative)
import cdsapi                # Climate Data Store API
```

### Visualization

```python
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns           # Statistical visualization
from mpldatacursor import datacursor  # Interactive data cursor
```

### Vector Database Integration

```python
import requests              # HTTP client for Qdrant API
import json                  # JSON data handling
from qdrant_client import QdrantClient  # Qdrant Python client
```

### System Integration

```python
import os                    # File system operations
import datetime             # Date/time handling
import zipfile              # ZIP archive handling
import binascii             # Binary data examination
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

### Climate Data Processing Patterns

- **Multi-Library Support**: Fallback mechanisms for GRIB processing (pygrib → cfgrib)
- **Format Detection**: Binary file header analysis for format identification
- **Error Handling**: Comprehensive exception handling for data format issues
- **Metadata Extraction**: Systematic extraction of file attributes and variable information

## File Format Support

### Traditional Formats

- **CSV**: Plain text comma-separated values
- **CSV.gz**: Gzip-compressed CSV files
- **Automatic Detection**: Built-in gzip file detection
- **Remote Downloads**: HTTP/HTTPS dataset retrieval

### Scientific Data Formats

- **NetCDF**: Network Common Data Form
  - Hierarchical data structure
  - Self-describing metadata
  - Multi-dimensional arrays
- **GRIB**: Gridded Binary format
  - Meteorological data standard
  - Compressed binary format
  - Message-based structure
- **ZIP**: Archive format support
  - Automatic extraction
  - Content analysis
  - Nested file handling

## Time Series Specific Features

- **Windowing**: Custom `WindowGenerator` class for time series preparation
- **Cyclical Features**: Sine/cosine encoding for daily and yearly cycles
- **FFT Analysis**: Fast Fourier Transform for frequency domain analysis
- **Peak Detection**: Custom algorithm for identifying dominant frequencies

## Research Integration

- **Jupyter Integration**: Seamless notebook-to-production pipeline
- **LaTeX Documentation**: Academic paper generation alongside code
- **Reproducible Research**: Consistent data processing and model training
- **Interactive Analysis**: Data cursor and visualization tools for exploration

## Performance Considerations

- **Batch Processing**: TensorFlow dataset batching (batch_size=32)
- **Memory Efficiency**: Lazy dataset loading and property-based access
- **Caching**: Local dataset caching to avoid repeated downloads
- **Vectorized Operations**: NumPy and TensorFlow for efficient computation
- **Compressed Storage**: Gzip compression for data files
- **Docker Optimization**: Containerized services for consistent performance

## Development Setup Requirements

### Core Environment

- Python 3.x environment
- TensorFlow installation
- Standard data science stack (pandas, numpy, matplotlib)
- Jupyter for notebook development
- LaTeX distribution for documentation compilation
- Git for version control

### Climate Data Processing

- netCDF4 library for NetCDF file handling
- xarray and cfgrib for GRIB file processing
- pygrib (optional, alternative GRIB library)
- cdsapi for climate data download
- CDS API credentials and configuration

### Vector Database

- Docker and Docker Compose
- Qdrant container image
- qdrant-client Python library
- requests library for HTTP API access

## Deployment Considerations

- **Modular Architecture**: Easy to deploy individual components
- **Configuration Management**: Environment-specific settings
- **Data Pipeline**: Automated data download and preprocessing
- **Model Persistence**: TensorFlow model saving/loading capabilities
- **Checkpointing System**: Robust model state management with joblib serialization
- **Development/Production Modes**: Configurable training modes for different environments
- **Containerized Services**: Docker-based deployment for vector database
- **API Integration**: RESTful API access for external data sources
- **Multi-Format Support**: Flexible data format handling for various input types

## Integration Architecture

### Data Flow Integration

- **PV Data Pipeline**: MJU dataset → preprocessing → feature engineering
- **Climate Data Pipeline**: CDS API → download → format detection → analysis
- **Vector Database Pipeline**: Code indexing → semantic search → context retrieval

### Service Integration

- **Qdrant Service**: Docker container with HTTP/gRPC APIs
- **CDS API Service**: External climate data service integration
- **Local Processing**: File-based analysis and visualization tools

### Development Integration

- **Notebook-to-Production**: Jupyter experiments → production code migration
- **Memory Bank System**: Documentation and context preservation
- **Testing Framework**: Comprehensive test suite for all components
