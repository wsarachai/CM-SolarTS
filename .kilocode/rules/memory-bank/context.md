# Current Context

## Project Status

The project is in **active development** phase with core infrastructure components implemented and research documentation in progress.

## Current Work Focus

- **Data Pipeline**: Core data loading, preprocessing, and windowing functionality is implemented
- **Frequency Analysis**: FFT-based frequency analysis for identifying periodic patterns in PV data
- **Model Framework**: Basic TensorFlow model training infrastructure is in place
- **Research Documentation**: LaTeX-based academic paper is being written alongside development

## Recent Changes

- Modular architecture implemented with separate classes for data loading, frequency analysis, window generation, and model training
- Custom dataset from MJU (Maejo University) integrated with 15-minute resolution PV data
- Cyclical time features (daily and yearly cycles) added to preprocessing pipeline
- Frequency analysis capability added to identify dominant patterns in power generation data

## Current Implementation State

### ✅ Completed Components

- [`DataLoader`](src/data_loader.py:6): Handles dataset download, preprocessing, and train/val/test splitting
- [`FrequencyAnalyzer`](src/frequency_analyzer.py:4): FFT-based analysis for identifying periodic patterns
- [`WindowGenerator`](src/window_generator.py:5): Time series windowing for supervised learning
- [`ModelTrainer`](src/model_trainer.py:3): Basic training infrastructure with early stopping
- Data preprocessing with cyclical time features and normalization

### 🚧 In Progress

- Model implementation: LSTM and CNN architectures are referenced but not fully implemented
- Integration between components: Main pipeline exists but model training is commented out
- Research validation: Academic paper structure exists but methodology details are incomplete

### 📋 Next Steps

- Implement specific time series models (LSTM, CNN-based architectures)
- Complete model training pipeline integration
- Add evaluation metrics and model comparison framework
- Implement fault detection and optimization strategies
- Complete research methodology documentation

## Key Insights Discovered

- Dataset contains 8 operational and environmental features with 15-minute resolution
- Frequency analysis reveals daily and seasonal patterns in PV power generation
- Zero-value detection implemented for data quality assessment
- Custom dataset from MJU provides real-world PV system operational data

## Technical Debt & Issues

- Model training pipeline is not fully integrated (commented out in main.py)
- Missing comprehensive evaluation framework
- No fault detection implementation yet
- Limited model architectures implemented
- Research paper methodology section needs completion

## Data Characteristics

- **Source**: MJU (Maejo University) PV installation
- **Resolution**: 15-minute intervals
- **Features**: Grid Feed In, External Energy Supply, Internal Power Supply, Current Power, Self Consumption, Ambient Temperature, Module Temperature, Total Irradiation
- **Preprocessing**: Cyclical time features, normalization, train/val/test split (70/20/10)
