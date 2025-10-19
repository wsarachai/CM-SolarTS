#!/usr/bin/env python3
"""
Example script demonstrating development mode usage in AdaBoostTrainer.

This script shows how to:
1. Use development mode for fast testing/debugging
2. Switch to production mode for full training
3. Toggle modes dynamically
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from window_generator import WindowGenerator
from adaboost_trainer import AdaBoostTrainer

def main():
    print("=== AdaBoost Development Mode Example ===\n")
    
    # Load and prepare data (same as main.py)
    DATASET_HOST = 'https://itsci.mju.ac.th/downloads/watcharin/datasets/pv/'
    DATASET_FILE = 'export_device_1_basic_aggregated_15minutes.csv.gz'
    ALL_COLS = [
        'Grid Feed In', 'External Energy Supply', 'Internal Power Supply',
        'Current Power', 'Self Consumption', 'Ambient Temperature',
        'Module Temperature', 'Total Irradiation'
    ]
    
    print("Loading and preprocessing data...")
    data_loader = DataLoader(DATASET_HOST, DATASET_FILE, ALL_COLS)
    df = data_loader.download_and_load()
    df = data_loader.preprocess()
    train_df, val_df, test_df = data_loader.split_and_normalize()
    
    window = WindowGenerator(
        input_width=24, label_width=1, shift=1,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=['Current Power']
    )
    
    # Example 1: Initialize with development mode
    print("\n1. Training with Development Mode (fast):")
    print("-" * 50)
    ada_trainer_dev = AdaBoostTrainer(
        window,
        n_estimators=50,
        differencing_order=1,
        dev_mode=True,
        dev_sample_ratio=0.05  # Use only 5% of data for very fast training
    )
    
    import time
    start_time = time.time()
    ada_trainer_dev.fit()
    dev_time = time.time() - start_time
    
    dev_score = ada_trainer_dev.score()
    print(f"Development mode training time: {dev_time:.2f} seconds")
    print(f"Development mode R² score: {dev_score:.4f}")
    
    # Example 2: Switch to production mode
    print("\n2. Switching to Production Mode (full training):")
    print("-" * 50)
    ada_trainer_dev.set_dev_mode(False)  # Switch to production mode
    
    start_time = time.time()
    ada_trainer_dev.fit()
    prod_time = time.time() - start_time
    
    prod_score = ada_trainer_dev.score()
    print(f"Production mode training time: {prod_time:.2f} seconds")
    print(f"Production mode R² score: {prod_score:.4f}")
    
    # Example 3: Initialize directly in production mode
    print("\n3. Direct Production Mode Initialization:")
    print("-" * 50)
    ada_trainer_prod = AdaBoostTrainer(
        window,
        n_estimators=50,
        differencing_order=1,
        dev_mode=False  # Production mode from start
    )
    
    print(f"Mode check: {'Development' if ada_trainer_prod.is_dev_mode() else 'Production'}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"Development mode: {dev_time:.2f}s, R² = {dev_score:.4f}")
    print(f"Production mode:  {prod_time:.2f}s, R² = {prod_score:.4f}")
    print(f"Speed improvement: {prod_time/dev_time:.1f}x faster in dev mode")
    
    print("\n=== Usage Tips ===")
    print("• Use dev_mode=True for rapid prototyping and testing")
    print("• Use dev_mode=False for final model training")
    print("• Adjust dev_sample_ratio (0.01-0.2) based on your needs")
    print("• Use set_dev_mode() to toggle modes dynamically")

if __name__ == "__main__":
    main()