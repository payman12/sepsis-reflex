"""
Transformer Training Bridge
============================

This module bridges the sepsis_simulation package with the MIMIC-sepsis/src
transformer model, enabling training on synthetic data when real MIMIC data
is not available.

Key Features:
- Generates synthetic training data compatible with TimeSeriesTransformer
- Uses the same data processing pipeline as MIMIC benchmark
- Supports training with configurable patient counts
- Includes evaluation metrics from MIMIC-sepsis/src/metrics.py

Cerebras Optimization Note:
---------------------------
The trained transformer model can then be deployed via Cerebras API for:
- Ultra-fast inference on streaming patient data
- Parallel processing of multiple patient sequences
- Real-time risk assessment with multi-cycle reasoning
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

# Add MIMIC-sepsis/src to path for imports
mimic_src_path = Path(__file__).parent.parent / "MIMIC-sepsis" / "src"
sys.path.insert(0, str(mimic_src_path))

# Import from our enhanced generator
from .enhanced_data_generator import EnhancedMIMICDataGenerator


@dataclass
class TrainingConfig:
    """Configuration for transformer training"""
    n_patients: int = 1000
    window_size: int = 6
    prediction_horizon: int = 6
    timestep_hours: int = 4
    train_ratio: float = 0.8
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    hidden_dim: int = 64
    num_layers: int = 2
    nhead: int = 8
    dropout: float = 0.1
    seed: Optional[int] = None  # None for different data each run
    

class TransformerTrainingBridge:
    """
    Bridge between synthetic data generation and transformer model training.
    
    This class:
    1. Generates synthetic patient data using EnhancedMIMICDataGenerator
    2. Formats data for TimeSeriesTransformer from MIMIC-sepsis/src
    3. Trains the model and evaluates performance
    4. Saves trained models for later use with Cerebras inference
    """
    
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize the training bridge.
        
        Args:
            config: Training configuration (uses defaults if None)
        """
        self.config = config or TrainingConfig()
        self.generator = EnhancedMIMICDataGenerator(seed=self.config.seed)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.training_history = []
        
    def generate_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic training and validation data.
        
        Returns:
            X_train: Training features (n_samples, window_size, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
        """
        print(f"Generating synthetic data for {self.config.n_patients} patients...")
        
        # Generate raw data
        X, y, feature_names = self.generator.generate_transformer_training_data(
            n_patients=self.config.n_patients,
            window_size=self.config.window_size,
            prediction_horizon=self.config.prediction_horizon,
            timestep_hours=self.config.timestep_hours
        )
        
        self.feature_names = feature_names
        
        print(f"Generated {len(X)} windows")
        print(f"Feature dimensions: {X.shape}")
        print(f"Positive class ratio: {y.mean():.2%}")
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            train_size=self.config.train_ratio,
            random_state=42,
            stratify=y
        )
        
        # Normalize features
        X_train, X_val = self._normalize_features(X_train, X_val)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val
    
    def _normalize_features(
        self, 
        X_train: np.ndarray, 
        X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features using training data statistics"""
        # Reshape to 2D for scaling
        train_shape = X_train.shape
        val_shape = X_val.shape
        
        X_train_2d = X_train.reshape(-1, train_shape[-1])
        X_val_2d = X_val.reshape(-1, val_shape[-1])
        
        # Fit on training data, transform both
        X_train_norm = self.scaler.fit_transform(X_train_2d)
        X_val_norm = self.scaler.transform(X_val_2d)
        
        # Reshape back
        X_train_norm = X_train_norm.reshape(train_shape)
        X_val_norm = X_val_norm.reshape(val_shape)
        
        return X_train_norm, X_val_norm
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Train the transformer model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Try to import the MIMIC transformer model
            from transformer_model import TimeSeriesTransformer
            print("Using MIMIC-sepsis TimeSeriesTransformer")
        except ImportError:
            # Fall back to a simple implementation
            print("MIMIC transformer not available, using built-in implementation")
            from .simple_transformer import SimpleTransformer as TimeSeriesTransformer
        
        # Initialize model
        self.model = TimeSeriesTransformer(
            input_dim=X_train.shape[2],
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            nhead=self.config.nhead,
            dropout=self.config.dropout,
            task_type='classification'
        )
        
        print(f"\nTraining transformer model...")
        print(f"  Hidden dim: {self.config.hidden_dim}")
        print(f"  Num layers: {self.config.num_layers}")
        print(f"  Attention heads: {self.config.nhead}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            validation_data=(X_val, y_val)
        )
        
        # Evaluate
        metrics = self.evaluate(X_train, y_train, X_val, y_val)
        
        return metrics
    
    def evaluate(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Returns comprehensive metrics including AUROC, AUPRC, etc.
        """
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            precision_recall_curve, accuracy_score,
            f1_score, confusion_matrix
        )
        
        # Get predictions
        train_preds = self.model.predict(X_train, batch_size=self.config.batch_size)
        val_preds = self.model.predict(X_val, batch_size=self.config.batch_size)
        
        # Calculate metrics
        metrics = {}
        
        # Training metrics
        metrics['train_auroc'] = roc_auc_score(y_train, train_preds)
        metrics['train_auprc'] = average_precision_score(y_train, train_preds)
        train_binary = (train_preds >= 0.5).astype(int)
        metrics['train_accuracy'] = accuracy_score(y_train, train_binary)
        metrics['train_f1'] = f1_score(y_train, train_binary)
        
        # Validation metrics
        metrics['val_auroc'] = roc_auc_score(y_val, val_preds)
        metrics['val_auprc'] = average_precision_score(y_val, val_preds)
        val_binary = (val_preds >= 0.5).astype(int)
        metrics['val_accuracy'] = accuracy_score(y_val, val_binary)
        metrics['val_f1'] = f1_score(y_val, val_binary)
        
        # Sensitivity/Specificity at optimal threshold
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_preds)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[min(optimal_idx, len(thresholds)-1)]
        
        val_optimal = (val_preds >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, val_optimal).ravel()
        
        metrics['val_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['val_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['val_ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['val_npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['optimal_threshold'] = optimal_threshold
        
        # Print results
        print("\n" + "=" * 50)
        print("MODEL EVALUATION RESULTS")
        print("=" * 50)
        print(f"\nTraining Metrics:")
        print(f"  AUROC:    {metrics['train_auroc']:.4f}")
        print(f"  AUPRC:    {metrics['train_auprc']:.4f}")
        print(f"  Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  F1:       {metrics['train_f1']:.4f}")
        
        print(f"\nValidation Metrics:")
        print(f"  AUROC:       {metrics['val_auroc']:.4f}")
        print(f"  AUPRC:       {metrics['val_auprc']:.4f}")
        print(f"  Accuracy:    {metrics['val_accuracy']:.4f}")
        print(f"  F1:          {metrics['val_f1']:.4f}")
        print(f"  Sensitivity: {metrics['val_sensitivity']:.4f}")
        print(f"  Specificity: {metrics['val_specificity']:.4f}")
        print(f"  PPV:         {metrics['val_ppv']:.4f}")
        print(f"  NPV:         {metrics['val_npv']:.4f}")
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print("=" * 50)
        
        return metrics
    
    def save_model(self, path: str = None):
        """Save the trained model and configuration"""
        if path is None:
            path = Path(__file__).parent / "trained_models"
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model weights
        import torch
        model_path = path / f"transformer_model_{timestamp}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save scaler
        import joblib
        scaler_path = path / f"scaler_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save config and feature names
        config_path = path / f"config_{timestamp}.json"
        config_data = {
            "training_config": {
                "n_patients": self.config.n_patients,
                "window_size": self.config.window_size,
                "prediction_horizon": self.config.prediction_horizon,
                "timestep_hours": self.config.timestep_hours,
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "nhead": self.config.nhead,
            },
            "feature_names": self.feature_names,
            "timestamp": timestamp
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Config saved to: {config_path}")
        
        return model_path, scaler_path, config_path
    
    def run_full_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        This is the main entry point for training a transformer model
        on synthetic data.
        
        Returns:
            Dictionary containing model, metrics, and paths
        """
        print("\n" + "=" * 60)
        print("TRANSFORMER TRAINING PIPELINE")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Patients: {self.config.n_patients}")
        print(f"  Window size: {self.config.window_size} timesteps")
        print(f"  Prediction horizon: {self.config.prediction_horizon} timesteps")
        print(f"  Random seed: {self.config.seed or 'None (random)'}")
        
        # Generate data
        X_train, y_train, X_val, y_val = self.generate_training_data()
        
        # Train model
        metrics = self.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model_path, scaler_path, config_path = self.save_model()
        
        return {
            "model": self.model,
            "metrics": metrics,
            "paths": {
                "model": str(model_path),
                "scaler": str(scaler_path),
                "config": str(config_path)
            }
        }


def train_on_synthetic_data(
    n_patients: int = 1000,
    epochs: int = 20,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to train a transformer model on synthetic data.
    
    Args:
        n_patients: Number of synthetic patients to generate
        epochs: Number of training epochs
        seed: Random seed (None for different data each run)
        
    Returns:
        Training results including model and metrics
    """
    config = TrainingConfig(
        n_patients=n_patients,
        epochs=epochs,
        seed=seed
    )
    
    bridge = TransformerTrainingBridge(config)
    return bridge.run_full_training_pipeline()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train transformer on synthetic data")
    parser.add_argument("--n_patients", type=int, default=500, 
                       help="Number of patients to generate")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (omit for different data each run)")
    parser.add_argument("--window_size", type=int, default=6,
                       help="Input window size in timesteps")
    parser.add_argument("--prediction_horizon", type=int, default=6,
                       help="Prediction horizon in timesteps")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        n_patients=args.n_patients,
        epochs=args.epochs,
        seed=args.seed,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon
    )
    
    bridge = TransformerTrainingBridge(config)
    results = bridge.run_full_training_pipeline()
    
    print("\nTraining complete!")
    print(f"Final validation AUROC: {results['metrics']['val_auroc']:.4f}")
