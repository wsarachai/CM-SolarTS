import logging
from typing import Optional

try:
    from .forecasting_system import ForecastingSystem
except ImportError:
    from forecasting_system import ForecastingSystem


def main():
    """
    Main function that runs the forecasting system.
    Can be used as a standalone script or imported as a module.
    """
    try:
        # Initialize forecasting system
        forecasting_system = ForecastingSystem()
        
        # Run the complete pipeline
        results = forecasting_system.run_full_pipeline()
        
        return results
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()