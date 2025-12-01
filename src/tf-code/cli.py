import argparse
import json
import sys
import logging
from typing import List, Optional
import pandas as pd

from .forecasting_system import ForecastingSystem

def setup_cli() -> argparse.ArgumentParser:
    """
    Setup the command-line interface argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Unified Forecasting System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models
  python -m src.cli train

  # Train specific models
  python -m src.cli train --models svm tensorflow

  # Evaluate all trained models
  python -m src.cli evaluate

  # Compare model performance
  python -m src.cli compare

  # Make predictions
  python -m src.cli predict --model svm --steps 5

  # Get system status
  python -m src.cli status

  # List available models
  python -m src.cli list-models

  # Use custom configuration
  python -m src.cli train --config config.json
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file'
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train forecasting models')
    train_parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Models to train (default: all enabled models)'
    )
    train_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for training results'
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Models to evaluate (default: all trained models)'
    )
    eval_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for evaluation results'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model performance')
    compare_parser.add_argument(
        '--models', '-m',
        nargs='+',
        help='Models to compare (default: all trained models)'
    )
    compare_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for comparison results'
    )
    compare_parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        default='json',
        help='Output format (default: json)'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Model to use for prediction'
    )
    predict_parser.add_argument(
        '--steps', '-s',
        type=int,
        default=1,
        help='Number of steps to predict (default: 1)'
    )
    predict_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for predictions'
    )
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get system status')
    status_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for status information'
    )
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    
    return parser

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
        log_file: Path to log file
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

def handle_train_command(args, forecasting_system: ForecastingSystem) -> None:
    """
    Handle the train command.

    Args:
        args: Parsed command-line arguments
        forecasting_system: ForecastingSystem instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training")
    
    try:
        # Train models
        results = forecasting_system.train_models(args.models)
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Training results saved to {args.output}")
        
        # Print summary
        print("\nTraining Results:")
        print("=" * 50)
        for model_type, result in results.items():
            if 'error' in result:
                print(f"{model_type}: ERROR - {result['error']}")
            else:
                val_score = result.get('val_score', 'N/A')
                print(f"{model_type}: Validation Score = {val_score:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)

def handle_evaluate_command(args, forecasting_system: ForecastingSystem) -> None:
    """
    Handle the evaluate command.

    Args:
        args: Parsed command-line arguments
        forecasting_system: ForecastingSystem instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation")
    
    try:
        # Evaluate models
        results = forecasting_system.evaluate_models(args.models)
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {args.output}")
        
        # Print summary
        print("\nEvaluation Results:")
        print("=" * 50)
        for model_type, result in results.items():
            if 'error' in result:
                print(f"{model_type}: ERROR - {result['error']}")
            else:
                print(f"{model_type}:")
                print(f"  R² Score: {result['test_score']:.4f}")
                print(f"  RMSE: {result['rmse']:.4f}")
                print(f"  MAE: {result['mae']:.4f}")
                print(f"  MAPE: {result['mape']:.4f}%")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        sys.exit(1)

def handle_compare_command(args, forecasting_system: ForecastingSystem) -> None:
    """
    Handle the compare command.

    Args:
        args: Parsed command-line arguments
        forecasting_system: ForecastingSystem instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model comparison")
    
    try:
        # Compare models
        comparison_df = forecasting_system.compare_models(args.models)
        
        # Save results if output file specified
        if args.output:
            if args.format == 'csv':
                comparison_df.to_csv(args.output, index=False)
            else:
                comparison_df.to_json(args.output, orient='records', indent=2)
            logger.info(f"Comparison results saved to {args.output}")
        
        # Print comparison table
        print("\nModel Comparison:")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        sys.exit(1)

def handle_predict_command(args, forecasting_system: ForecastingSystem) -> None:
    """
    Handle the predict command.

    Args:
        args: Parsed command-line arguments
        forecasting_system: ForecastingSystem instance
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting prediction with {args.model} model")
    
    try:
        # Make predictions
        predictions = forecasting_system.predict(args.model, args.steps)
        
        # Save results if output file specified
        if args.output:
            if args.output.endswith('.json'):
                with open(args.output, 'w') as f:
                    json.dump({'predictions': predictions.tolist()}, f, indent=2)
            else:
                pd.DataFrame({'predictions': predictions}).to_csv(args.output, index=False)
            logger.info(f"Predictions saved to {args.output}")
        
        # Print predictions
        print(f"\nPredictions ({args.model} model, {args.steps} steps):")
        print("=" * 50)
        for i, pred in enumerate(predictions):
            print(f"Step {i+1}: {pred:.4f}")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        sys.exit(1)

def handle_status_command(args, forecasting_system: ForecastingSystem) -> None:
    """
    Handle the status command.

    Args:
        args: Parsed command-line arguments
        forecasting_system: ForecastingSystem instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Getting system status")
    
    try:
        # Get status
        status = forecasting_system.get_system_status()
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(status, f, indent=2)
            logger.info(f"Status saved to {args.output}")
        
        # Print status
        print("\nSystem Status:")
        print("=" * 50)
        print(f"Initialized: {status['initialized']}")
        print(f"Data Loaded: {status['data_loaded']}")
        print(f"Available Models: {', '.join(status['available_models'])}")
        print(f"Trained Models: {', '.join(status['trained_models'])}")
        
        # Print model info
        if status['model_info']:
            print("\nModel Information:")
            print("-" * 30)
            for model_type, info in status['model_info'].items():
                print(f"{model_type}:")
                print(f"  Trained: {info['is_trained']}")
                print(f"  Type: {info['model_type']}")
        
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        sys.exit(1)

def handle_list_models_command(args, forecasting_system: ForecastingSystem) -> None:
    """
    Handle the list-models command.

    Args:
        args: Parsed command-line arguments
        forecasting_system: ForecastingSystem instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Listing available models")
    
    try:
        # Get available models
        from .model_factory import ModelFactory
        available_models = ModelFactory.get_available_models()
        
        print("\nAvailable Models:")
        print("=" * 30)
        for model_type in available_models:
            config = ModelFactory.get_default_config(model_type)
            print(f"{model_type}:")
            for key, value in config.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
            print()
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    # Parse arguments
    parser = setup_cli()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Create forecasting system
    forecasting_system = ForecastingSystem(args.config)
    
    # Handle commands
    if args.command == 'train':
        handle_train_command(args, forecasting_system)
    elif args.command == 'evaluate':
        handle_evaluate_command(args, forecasting_system)
    elif args.command == 'compare':
        handle_compare_command(args, forecasting_system)
    elif args.command == 'predict':
        handle_predict_command(args, forecasting_system)
    elif args.command == 'status':
        handle_status_command(args, forecasting_system)
    elif args.command == 'list-models':
        handle_list_models_command(args, forecasting_system)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()