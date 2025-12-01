"""
Unified Forecasting System Package

This package provides a comprehensive forecasting system for photovoltaic (PV) 
power output prediction that consolidates multiple machine learning models 
into a single, cohesive framework.
"""

# Import main components for easy access
from .base_trainer import BaseTrainer
from .svm_trainer import SVMTrainer
from .tensorflow_trainer import TensorFlowTrainer
from .adaboost_trainer import AdaBoostTrainer
from .model_factory import ModelFactory
from .forecasting_system import ForecastingSystem
from .window_generator import WindowGenerator
from .data_loader import DataLoader

__version__ = "1.0.0"
__author__ = "Watcharin Sarachai"

__all__ = [
    'BaseTrainer',
    'SVMTrainer', 
    'TensorFlowTrainer',
    'AdaBoostTrainer',
    'ModelFactory',
    'ForecastingSystem',
    'WindowGenerator',
    'DataLoader'
]