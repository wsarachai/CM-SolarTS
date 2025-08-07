#!/usr/bin/env python3
"""
Example usage of the unified forecasting system.

This script demonstrates how to use the unified forecasting system to train
and evaluate different models (SVM, TensorFlow, AdaBoost) on PV power data.
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.forecasting_system import ForecastingSystem

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('forecasting_example.log')
        ]
    )

def main():
    """Main example function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting unified forecasting system example")
    
    # Initialize the forecasting system with default configuration
    forecasting_system = ForecastingSystem()
    
    try:
        # Initialize data and window generator
        logger.info("Initializing data and window generator...")
        forecasting_system.initialize_data()
        
        # Get system status
        status = forecasting_system.get_status()
        logger.info(f"System status: {status}")
        
        # Example 1: Train SVM model
        logger.info("\n" + "="*50)
        logger.info("Example 1: Training SVM model")
        logger.info("="*50)
        
        svm_config = {
            'dev_mode': True,  # Use development mode for faster training
            'optimization_method': 'grid',
            'cv_folds': 3
        }
        
        svm_results = forecasting_system.train_model('svm', svm_config)
        logger.info(f"SVM training results: {svm_results}")
        
        # Evaluate SVM model
        svm_eval = forecasting_system.evaluate_model('svm')
        logger.info(f"SVM evaluation: {svm_eval}")
        
        # Example 2: Train TensorFlow model
        logger.info("\n" + "="*50)
        logger.info("Example 2: Training TensorFlow model")
        logger.info("="*50)
        
        tf_config = {
            'dev_mode': True,  # Use development mode for faster training
            'max_epochs': 10,
            'patience': 2
        }
        
        tf_results = forecasting_system.train_model('tensorflow', tf_config)
        logger.info(f"TensorFlow training results: {tf_results}")
        
        # Evaluate TensorFlow model
        tf_eval = forecasting_system.evaluate_model('tensorflow')
        logger.info(f"TensorFlow evaluation: {tf_eval}")
        
        # Example 3: Train AdaBoost model
        logger.info("\n" + "="*50)
        logger.info("Example 3: Training AdaBoost model")
        logger.info("="*50)
        
        ada_config = {
            'dev_mode': True,  # Use development mode for faster training
            'n_estimators': 20,
            'learning_rate': 1.0
        }
        
        ada_results = forecasting_system.train_model('adaboost', ada_config)
        logger.info(f"AdaBoost training results: {ada_results}")
        
        # Evaluate AdaBoost model
        ada_eval = forecasting_system.evaluate_model('adaboost')
        logger.info(f"AdaBoost evaluation: {ada_eval}")
        
        # Example 4: Compare all models
        logger.info("\n" + "="*50)
        logger.info("Example 4: Comparing all models")
        logger.info("="*50)
        
        comparison = forecasting_system.compare_models(['svm', 'tensorflow', 'adaboost'])
        logger.info("Model comparison results:")
        for model_name, metrics in comparison.items():
            logger.info(f"  {model_name}: {metrics}")
        
        # Example 5: Make predictions
        logger.info("\n" + "="*50)
        logger.info("Example 5: Making predictions")
        logger.info("="*50)
        
        # Make predictions with the best performing model
        best_model = min(comparison.keys(), key=lambda x: comparison[x].get('rmse', float('inf')))
        logger.info(f"Best model based on RMSE: {best_model}")
        
        predictions = forecasting_system.predict(best_model)
        logger.info(f"Generated {len(predictions)} predictions with {best_model} model")
        logger.info(f"Prediction sample (first 5): {predictions[:5].flatten()}")
        
        # Example 6: Save models
        logger.info("\n" + "="*50)
        logger.info("Example 6: Saving models")
        logger.info("="*50)
        
        for model_name in ['svm', 'tensorflow', 'adaboost']:
            try:
                saved_path = forecasting_system.save_model(model_name)
                logger.info(f"Saved {model_name} model to: {saved_path}")
            except Exception as e:
                logger.error(f"Failed to save {model_name} model: {e}")
        
        logger.info("\nExample completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in example execution: {e}")
        raise

if __name__ == "__main__":
    main()