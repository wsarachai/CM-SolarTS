#!/usr/bin/env python3
"""
Test script for AdaBoostTrainer checkpointing functionality.
"""

import sys
import os
import numpy as np
import tempfile
import shutil

# Add src to path
sys.path.append('src')

from adaboost_trainer import AdaBoostTrainer

def create_mock_window_generator():
    """Create a mock window generator for testing."""
    class MockDataset:
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __iter__(self):
            yield self.data, self.labels
    
    class MockWindowGenerator:
        def __init__(self, data, labels):
            self._train_data = data
            self._train_labels = labels
            self._test_data = data
            self._test_labels = labels
        
        @property
        def train(self):
            return MockDataset(self._train_data, self._train_labels)
        
        @property
        def test(self):
            return MockDataset(self._test_data, self._test_labels)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    data = np.random.randn(n_samples, n_features)
    labels = np.random.randn(n_samples, 1)
    
    return MockWindowGenerator(data, labels)

def test_checkpointing():
    """Test the checkpointing functionality."""
    print("Testing AdaBoostTrainer checkpointing functionality...")
    
    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "test_model.pkl")
    
    try:
        # Create mock window generator
        window_gen = create_mock_window_generator()
        
        # Test 1: Train with checkpointing enabled
        print("\n=== Test 1: Training with checkpointing enabled ===")
        trainer = AdaBoostTrainer(
            window_gen, 
            n_estimators=5, 
            dev_mode=True, 
            dev_sample_ratio=0.2,
            checkpoint_enabled=True,
            checkpoint_path=checkpoint_path
        )
        
        # Train the model
        trainer.fit(n_splits=2)  # Use fewer splits for faster testing
        
        # Check if checkpoint was created
        assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
        print("Checkpoint file created successfully")
        
        # Check best score
        best_score = trainer.get_best_score()
        assert best_score > -np.inf, "Best score was not updated"
        print(f"Best score recorded: {best_score:.4f}")
        
        # Test 2: Load existing model
        print("\n=== Test 2: Loading existing model ===")
        trainer2 = AdaBoostTrainer(
            window_gen,
            n_estimators=5,
            dev_mode=True,
            dev_sample_ratio=0.2,
            checkpoint_enabled=True,
            checkpoint_path=checkpoint_path,
            load_existing_model=True
        )
        
        # Check if model was loaded
        assert trainer2.best_model is not None, "Model was not loaded"
        print("Model loaded successfully")
        
        # Check if training is skipped
        trainer2.fit(n_splits=2)  # This should skip training
        print("Training skipped when loading existing model")
        
        # Test 3: Restore best model
        print("\n=== Test 3: Restoring best model ===")
        trainer3 = AdaBoostTrainer(
            window_gen,
            n_estimators=5,
            dev_mode=True,
            dev_sample_ratio=0.2,
            checkpoint_enabled=False  # Disable checkpointing for this test
        )
        
        # Train a new model
        trainer3.fit(n_splits=2)
        original_score = trainer3.get_best_score()
        
        # Now try to load and restore the best model
        trainer3.checkpoint_path = checkpoint_path
        trainer3.checkpoint_enabled = True
        trainer3._load_model()
        success = trainer3.restore_best_model()
        
        assert success, "Failed to restore best model"
        print("Best model restored successfully")
        
        # Test 4: Delete checkpoint
        print("\n=== Test 4: Deleting checkpoint ===")
        success = trainer.delete_checkpoint()
        assert success, "Failed to delete checkpoint"
        assert not os.path.exists(checkpoint_path), "Checkpoint file still exists"
        print("Checkpoint deleted successfully")
        
        # Test 5: Disable checkpointing
        print("\n=== Test 5: Testing with checkpointing disabled ===")
        trainer4 = AdaBoostTrainer(
            window_gen,
            n_estimators=5,
            dev_mode=True,
            dev_sample_ratio=0.2,
            checkpoint_enabled=False
        )
        
        trainer4.fit(n_splits=2)
        # Should not create checkpoint file
        assert not os.path.exists(checkpoint_path), "Checkpoint file created when disabled"
        print("Checkpointing disabled correctly")
        
        print("\nAll tests passed!")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Create mock window generator
    window_gen = create_mock_window_generator()
    
    # Test loading non-existent checkpoint
    trainer = AdaBoostTrainer(
        window_gen,
        checkpoint_enabled=True,
        checkpoint_path="non_existent_path.pkl",
        load_existing_model=True
    )
    
    # Should handle missing file gracefully
    assert trainer.best_model is None, "Loaded non-existent model"
    print("Handled non-existent checkpoint file")
    
    # Test invalid checkpoint path
    trainer = AdaBoostTrainer(
        window_gen,
        checkpoint_enabled=True,
        checkpoint_path="/invalid/path/model.pkl"
    )
    
    # Should handle invalid path gracefully
    trainer.fit(n_splits=2)  # Should not crash
    print("Handled invalid checkpoint path")
    
    print("Edge case tests passed")

if __name__ == "__main__":
    test_checkpointing()
    test_edge_cases()
    print("\nAll checkpointing tests completed successfully!")