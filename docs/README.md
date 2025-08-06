# Time Series Analysis and Forecasting Project

This project explores time series analysis techniques on various datasets, including climate and solar power data. It covers data cleaning, feature engineering, visualization, frequency analysis using Fast Fourier Transform (FFT), and building forecasting models with TensorFlow.

## Project Structure

```
.
├── main.py             # Main script for data analysis and modeling
├── README.md           # This file
└── *.csv / *.csv.gz    # Data files used for analysis
```

- **`main.py`**: The core script containing the logic for:

  - Loading data from local files and remote URLs.
  - Cleaning and preprocessing data (handling NaNs, filling missing timestamps).
  - Performing feature engineering (e.g., converting wind data to vectors).
  - Visualizing data with Matplotlib and Seaborn.
  - Analyzing frequency components with Fast Fourier Transform (FFT).
  - Building and training time series forecasting models using TensorFlow/Keras.

- **`README.md`**: Provides an overview of the project structure and purpose.

- **Data Files (`.csv`, `.csv.gz`)**: Contains the raw time series data used in the analysis. The project demonstrates handling of both plain text and compressed CSV files.

## Key Techniques Demonstrated

- **Data Handling**: Loading and saving data with pandas, including from compressed formats and URLs.
- **Data Cleaning**: Identifying and filling missing timestamps, handling `NaN` values, and replacing anomalous zero-value data.
- **Feature Engineering**: Creating cyclical time features (e.g., from timestamps) and converting polar coordinates (wind speed/direction) to Cartesian vectors.
- **Time Series Analysis**: Using Fast Fourier Transform (FFT) to identify and visualize periodic patterns like daily and yearly cycles.
- **Machine Learning**: Implementing a custom `WindowGenerator` class to prepare time series data for supervised learning and building forecasting models
