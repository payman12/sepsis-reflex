"""
Expert vs AI Comparison Module
================================

This module provides tools for comparing AI model predictions against
expert-labeled ground truth data. It generates comprehensive accuracy
metrics to evaluate model performance.

Key Features:
- Generates N patients with "expert" ground truth labels
- Runs AI inference (Bayesian risk tracker or Transformer model)
- Computes accuracy metrics at multiple time points
- Visualizes comparison results
- Supports variable patient counts per run

Cerebras Optimization Note:
---------------------------
The comparison can leverage Cerebras for:
- Parallel inference across all patients
- Multiple reasoning cycles per patient
- Fast batch processing of evaluation data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, confusion_matrix, accuracy_score
)


@dataclass
class ComparisonConfig:
    """Configuration for expert vs AI comparison"""
    n_patients: int = 20
    duration_hours: int = 48
    timestep_minutes: int = 60
    sepsis_ratio: float = 0.5
    evaluation_points: List[str] = field(default_factory=lambda: [
        "final", "at_onset", "6h_before", "12h_before"
    ])
    risk_threshold: float = 0.5  # Threshold for binary classification
    seed: Optional[int] = None  # None for different patients each run


@dataclass 
class ComparisonResult:
    """Results of expert vs AI comparison"""
    # Overall metrics
    auroc: float
    auprc: float
    accuracy: float
    f1: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    
    # Per-evaluation-point metrics
    point_metrics: Dict[str, Dict[str, float]]
    
    # Per-patient results
    patient_results: pd.DataFrame
    
    # Metadata
    n_patients: int
    n_sepsis: int
    n_non_sepsis: int
    config: ComparisonConfig
    timestamp: str


class ExpertAIComparison:
    """
    Compare AI predictions against expert-labeled ground truth.
    
    This class:
    1. Generates synthetic patient data with expert labels
    2. Runs AI inference (Bayesian tracker or Transformer)
    3. Computes comprehensive accuracy metrics
    4. Provides visualization data
    """
    
    def __init__(self, config: ComparisonConfig = None):
        """
        Initialize the comparison.
        
        Args:
            config: Comparison configuration
        """
        self.config = config or ComparisonConfig()
        
        # Initialize data generator
        from .enhanced_data_generator import EnhancedMIMICDataGenerator
        self.generator = EnhancedMIMICDataGenerator(seed=self.config.seed)
        
        # Initialize risk tracker (Bayesian approach)
        from .risk_state_tracker import BayesianRiskTracker
        from .physiological_features import PhysiologicalFeatureExtractor
        self.risk_tracker = BayesianRiskTracker()
        self.feature_extractor = PhysiologicalFeatureExtractor()
        
        # Store generated data
        self.patient_data: Optional[pd.DataFrame] = None
        self.expert_labels: Optional[pd.DataFrame] = None
        self.ai_predictions: Optional[Dict] = None
    
    def generate_comparison_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate patient data with expert labels.
        
        Returns:
            patient_data: Time series data
            expert_labels: Ground truth labels
        """
        print(f"\nGenerating {self.config.n_patients} patients for comparison...")
        print(f"  Duration: {self.config.duration_hours} hours")
        print(f"  Timestep: {self.config.timestep_minutes} minutes")
        print(f"  Sepsis ratio: {self.config.sepsis_ratio:.0%}")
        print(f"  Seed: {self.config.seed or 'None (random)'}")
        
        self.patient_data, self.expert_labels = self.generator.generate_patient_cohort(
            n_patients=self.config.n_patients,
            duration_hours=self.config.duration_hours,
            timestep_minutes=self.config.timestep_minutes,
            sepsis_ratio=self.config.sepsis_ratio,
            include_expert_labels=True
        )
        
        n_sepsis = int(self.expert_labels['develops_sepsis'].sum())
        print(f"  Generated {n_sepsis} sepsis, {self.config.n_patients - n_sepsis} non-sepsis")
        
        return self.patient_data, self.expert_labels
    
    def run_ai_inference(self) -> Dict[str, Dict]:
        """
        Run AI inference on all patients.
        
        Uses the Bayesian risk tracker to generate risk scores
        over time for each patient.
        
        Returns:
            Dictionary mapping patient_id to risk predictions
        """
        if self.patient_data is None:
            self.generate_comparison_data()
        
        print("\nRunning AI inference on all patients...")
        
        self.ai_predictions = {}
        
        for pid in self.patient_data['patient_id'].unique():
            patient_df = self.patient_data[
                self.patient_data['patient_id'] == pid
            ].sort_values('timestep')
            
            # Reset tracker and feature extractor for each patient
            self.risk_tracker.reset()
            self.feature_extractor.reset()
            
            # Track risk over time
            risk_history = []
            uncertainty_history = []
            
            for idx, row in patient_df.iterrows():
                # Create signal dictionary from row data
                signals = {}
                
                # Extract vital signs and other measurements
                signal_mapping = {
                    'heart_rate': 'heart_rate',
                    'map': 'map',
                    'respiratory_rate': 'respiratory_rate',
                    'spo2': 'spo2',
                    'temp_C': 'temperature',
                    'lactic_acid': 'lactate',
                    'wbc': 'wbc'
                }
                
                for col, signal_name in signal_mapping.items():
                    if col in row and not pd.isna(row[col]):
                        signals[signal_name] = row[col]
                
                # Extract features using the feature extractor
                timestamp = row.get('hours_from_start', idx) * 3600  # Convert hours to seconds
                features = self.feature_extractor.extract(timestamp, signals)
                
                # Update risk tracker with extracted features
                if features is not None:
                    belief = self.risk_tracker.update(features)
                    risk_history.append(belief.mean)
                    uncertainty_history.append(np.sqrt(belief.variance))
                else:
                    # Use previous values if no features
                    risk_history.append(risk_history[-1] if risk_history else 0.1)
                    uncertainty_history.append(uncertainty_history[-1] if uncertainty_history else 0.3)
            
            # Store predictions
            self.ai_predictions[pid] = {
                'risk_history': risk_history,
                'uncertainty_history': uncertainty_history,
                'final_risk': risk_history[-1] if risk_history else 0.5,
                'max_risk': max(risk_history) if risk_history else 0.5,
                'timesteps': list(range(len(risk_history)))
            }
        
        print(f"  Completed inference for {len(self.ai_predictions)} patients")
        
        return self.ai_predictions
    
    def compute_metrics(self) -> ComparisonResult:
        """
        Compute comprehensive comparison metrics.
        
        Returns:
            ComparisonResult with all metrics
        """
        if self.ai_predictions is None:
            self.run_ai_inference()
        
        print("\nComputing comparison metrics...")
        
        # Collect labels and predictions
        y_true = []
        y_pred = []
        patient_results = []
        
        for _, row in self.expert_labels.iterrows():
            pid = row['patient_id']
            true_sepsis = row['develops_sepsis']
            onset_timestep = row['sepsis_onset_timestep']
            
            pred = self.ai_predictions.get(pid, {})
            pred_risk = pred.get('max_risk', 0.5)
            final_risk = pred.get('final_risk', 0.5)
            
            y_true.append(1 if true_sepsis else 0)
            y_pred.append(pred_risk)
            
            # Per-patient results
            patient_results.append({
                'patient_id': pid,
                'true_sepsis': true_sepsis,
                'onset_timestep': onset_timestep,
                'ai_max_risk': pred_risk,
                'ai_final_risk': final_risk,
                'ai_prediction': pred_risk >= self.config.risk_threshold,
                'correct': (pred_risk >= self.config.risk_threshold) == true_sepsis
            })
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_binary = (y_pred >= self.config.risk_threshold).astype(int)
        
        # Overall metrics
        auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        auprc = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
        accuracy = accuracy_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Confusion matrix metrics
        if len(np.unique(y_true)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        else:
            sensitivity = specificity = ppv = npv = 0.0
        
        # Metrics at different evaluation points
        point_metrics = self._compute_point_metrics()
        
        # Create result
        result = ComparisonResult(
            auroc=auroc,
            auprc=auprc,
            accuracy=accuracy,
            f1=f1,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            point_metrics=point_metrics,
            patient_results=pd.DataFrame(patient_results),
            n_patients=self.config.n_patients,
            n_sepsis=int(y_true.sum()),
            n_non_sepsis=int((1 - y_true).sum()),
            config=self.config,
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    def _compute_point_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics at specific evaluation points"""
        point_metrics = {}
        
        for point in self.config.evaluation_points:
            y_true = []
            y_pred = []
            
            for _, row in self.expert_labels.iterrows():
                pid = row['patient_id']
                true_sepsis = row['develops_sepsis']
                onset_timestep = row['sepsis_onset_timestep']
                
                pred = self.ai_predictions.get(pid, {})
                risk_history = pred.get('risk_history', [0.5])
                
                # Determine evaluation timestep
                if point == "final":
                    eval_idx = -1
                elif point == "at_onset" and onset_timestep is not None and not pd.isna(onset_timestep):
                    eval_idx = min(int(onset_timestep), len(risk_history) - 1)
                elif point == "6h_before" and onset_timestep is not None and not pd.isna(onset_timestep):
                    hours_before = 6
                    timesteps_before = int(hours_before * 60 / self.config.timestep_minutes)
                    eval_idx = max(0, int(onset_timestep) - timesteps_before)
                elif point == "12h_before" and onset_timestep is not None and not pd.isna(onset_timestep):
                    hours_before = 12
                    timesteps_before = int(hours_before * 60 / self.config.timestep_minutes)
                    eval_idx = max(0, int(onset_timestep) - timesteps_before)
                else:
                    eval_idx = -1
                
                if eval_idx < len(risk_history):
                    y_true.append(1 if true_sepsis else 0)
                    y_pred.append(risk_history[eval_idx])
            
            if len(y_true) > 0 and len(np.unique(y_true)) > 1:
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                y_pred_binary = (y_pred >= self.config.risk_threshold).astype(int)
                
                point_metrics[point] = {
                    'auroc': roc_auc_score(y_true, y_pred),
                    'accuracy': accuracy_score(y_true, y_pred_binary),
                    'f1': f1_score(y_true, y_pred_binary, zero_division=0)
                }
            else:
                point_metrics[point] = {'auroc': 0.5, 'accuracy': 0.0, 'f1': 0.0}
        
        return point_metrics
    
    def run_comparison(self) -> ComparisonResult:
        """
        Run the full comparison pipeline.
        
        This is the main entry point for running expert vs AI comparison.
        
        Returns:
            ComparisonResult with all metrics
        """
        print("\n" + "=" * 60)
        print("EXPERT vs AI COMPARISON")
        print("=" * 60)
        
        # Generate data
        self.generate_comparison_data()
        
        # Run AI inference
        self.run_ai_inference()
        
        # Compute metrics
        result = self.compute_metrics()
        
        # Print results
        self._print_results(result)
        
        return result
    
    def _print_results(self, result: ComparisonResult):
        """Print formatted comparison results"""
        print("\n" + "-" * 50)
        print("COMPARISON RESULTS")
        print("-" * 50)
        
        print(f"\nPatients: {result.n_patients}")
        print(f"  Sepsis: {result.n_sepsis}")
        print(f"  Non-sepsis: {result.n_non_sepsis}")
        
        print(f"\nOverall Metrics:")
        print(f"  AUROC:       {result.auroc:.4f}")
        print(f"  AUPRC:       {result.auprc:.4f}")
        print(f"  Accuracy:    {result.accuracy:.4f}")
        print(f"  F1 Score:    {result.f1:.4f}")
        print(f"  Sensitivity: {result.sensitivity:.4f}")
        print(f"  Specificity: {result.specificity:.4f}")
        print(f"  PPV:         {result.ppv:.4f}")
        print(f"  NPV:         {result.npv:.4f}")
        
        print(f"\nMetrics by Evaluation Point:")
        for point, metrics in result.point_metrics.items():
            print(f"  {point}:")
            print(f"    AUROC: {metrics['auroc']:.4f}, "
                  f"Accuracy: {metrics['accuracy']:.4f}, "
                  f"F1: {metrics['f1']:.4f}")
        
        # Patient-level summary
        correct = result.patient_results['correct'].sum()
        print(f"\nPatient-level Accuracy: {correct}/{result.n_patients} "
              f"({correct/result.n_patients:.1%})")
        
        print("-" * 50)
    
    def save_results(self, path: str = None) -> str:
        """Save comparison results to file"""
        result = self.compute_metrics()
        
        if path is None:
            path = Path(__file__).parent / "comparison_results"
        path = Path(path)
        path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = path / f"comparison_summary_{timestamp}.json"
        summary = {
            'n_patients': result.n_patients,
            'n_sepsis': result.n_sepsis,
            'n_non_sepsis': result.n_non_sepsis,
            'auroc': result.auroc,
            'auprc': result.auprc,
            'accuracy': result.accuracy,
            'f1': result.f1,
            'sensitivity': result.sensitivity,
            'specificity': result.specificity,
            'ppv': result.ppv,
            'npv': result.npv,
            'point_metrics': result.point_metrics,
            'timestamp': result.timestamp
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save patient results
        patient_path = path / f"patient_results_{timestamp}.csv"
        result.patient_results.to_csv(patient_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"  Summary: {summary_path}")
        print(f"  Patient results: {patient_path}")
        
        return str(summary_path)


def run_expert_ai_comparison(
    n_patients: int = 20,
    seed: Optional[int] = None
) -> ComparisonResult:
    """
    Convenience function to run expert vs AI comparison.
    
    Args:
        n_patients: Number of patients to compare
        seed: Random seed (None for different patients each run)
        
    Returns:
        ComparisonResult with all metrics
    """
    config = ComparisonConfig(
        n_patients=n_patients,
        seed=seed
    )
    
    comparison = ExpertAIComparison(config)
    return comparison.run_comparison()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Expert vs AI comparison")
    parser.add_argument("--n_patients", type=int, default=20,
                       help="Number of patients to compare")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (omit for different patients each run)")
    parser.add_argument("--save", action="store_true",
                       help="Save results to file")
    
    args = parser.parse_args()
    
    config = ComparisonConfig(
        n_patients=args.n_patients,
        seed=args.seed
    )
    
    comparison = ExpertAIComparison(config)
    result = comparison.run_comparison()
    
    if args.save:
        comparison.save_results()
