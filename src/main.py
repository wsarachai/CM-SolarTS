try:
    from .data_loader import DataLoader
    from .frequency_analyzer import FrequencyAnalyzer
    from .window_generator import WindowGenerator
    from .model_trainer import ModelTrainer
except ImportError:
    # Fallback for running as a script directly (not as a module)
    from data_loader import DataLoader
    from frequency_analyzer import FrequencyAnalyzer
    from window_generator import WindowGenerator
    from model_trainer import ModelTrainer

import numpy as np

# Constants
DATASET_HOST = 'https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/'
DATASET_FILE = 'export_device_1_basic_aggregated_15minutes.csv.gz'
ALL_COLS = [
    'Grid Feed In', 'External Energy Supply', 'Internal Power Supply',
    'Current Power', 'Self Consumption', 'Ambient Temperature',
    'Module Temperature', 'Total Irradiation'
]
MAX_EPOCHS = 5

def main():
    # Data loading and preprocessing
    data_loader = DataLoader(DATASET_HOST, DATASET_FILE, ALL_COLS)
    df = data_loader.download_and_load()
    df = data_loader.preprocess()
    train_df, val_df, test_df = data_loader.split_and_normalize()

    print(f"Total rows: {df.shape[0]}")
    import pandas as pd
    if pd.api.types.is_datetime64_any_dtype(df.index):
        print(df.index.year.unique())
        print(df.index.year.value_counts())
    else:
        print("Warning: DataFrame index is not a DatetimeIndex.")

    # Frequency analysis on 'Current Power'
    if 'Current Power' not in df.columns:
        print("Error: 'Current Power' column not found in dataframe.")
        return

    n_samples = len(df['Current Power'])
    sample_period = 15 * 60  # 15 minutes in seconds

    # Handle missing values for FFT
    current_power = df['Current Power'].fillna(0).to_numpy()

    freq_analyzer = FrequencyAnalyzer(current_power, sample_period)
    peak_frequencies, peak_periods = freq_analyzer.analyze()

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

    # Prepare for window generation (example usage)
    window = WindowGenerator(
        input_width=24, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['Current Power']
    )

    # Prepare for model training (example usage)
    # model_trainer = ModelTrainer(max_epochs=MAX_EPOCHS)
    # history = model_trainer.compile_and_fit(model, window)

    # AdaBoost Training and Evaluation
    from adaboost_trainer import AdaBoostTrainer
    
    # Enable development mode for faster training during testing
    # Set dev_mode=False for production runs
    ada_trainer = AdaBoostTrainer(
        window,
        differencing_order=1,
        dev_mode=False,  # Enable for faster development/testing
        dev_sample_ratio=0.1  # Use 10% of data for quick testing
    )
    
    print(f"Training mode: {'Development' if ada_trainer.is_dev_mode() else 'Production'}")
    ada_trainer.fit(n_splits=5)
    predictions = ada_trainer.predict()
    r2_score = ada_trainer.score()
    rmse = ada_trainer.calculate_rmse()
    mae = ada_trainer.calculate_mae()
    print(f"AdaBoost R^2 Score: {r2_score:.4f}")
    print(f"AdaBoost RMSE: {rmse:.4f}")
    print(f"AdaBoost MAE: {mae:.4f}")

    # Plotting the results
    import matplotlib.pyplot as plt

    # Get the test data
    test_ds = window.test
    X_test, y_test = ada_trainer._extract_xy(test_ds)
    if ada_trainer.differencing_order > 0:
        y_test = y_test[ada_trainer.differencing_order:]
    #y_pred = ada_trainer.predict(test_ds)
    y_pred = ada_trainer.predict()

    # Plotting the last 100 actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[-100:], label='Actual')
    plt.plot(y_pred[-100:], label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Current Power')
    plt.title('Actual vs Predicted Current Power (Last 100 Values)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()