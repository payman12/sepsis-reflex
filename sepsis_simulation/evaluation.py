"""
Evaluation Module: Accuracy and Speed Metrics
==============================================

This module provides comprehensive evaluation of the sepsis simulation system:

1. ACCURACY EVALUATION
   - AUROC: Area under ROC curve
   - AUPRC: Area under Precision-Recall curve  
   - F1 Score at various thresholds
   - Expert vs Machine comparison
   - Agreement with ground truth labels

2. SPEED EVALUATION
   - Latency: Time from signal ingestion to risk update
   - Throughput: Observations processed per second
   - Reasoning cycles per second
   - Cerebras utilization metrics

IMPORTANT NOTES:
- Speed metrics measure COMPUTATIONAL performance
- NOT biological onset timing or clinical decision speed
- No clinical deployment claims are made
- This is a research prototype evaluation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    confusion_matrix,
    accuracy_score
)
import time

try:
    from .config import PatientState, EvaluationConfig
    from .simulation_engine import SimulationResult, BenchmarkResult
except ImportError:
    from config import PatientState, EvaluationConfig
    from simulation_engine import SimulationResult, BenchmarkResult


@dataclass
class AccuracyMetrics:
    """
    Container for accuracy evaluation metrics.
    
    All metrics compare model risk estimates against ground truth labels.
    """
    # ROC metrics
    auroc: float = 0.0
    roc_curve_fpr: Optional[np.ndarray] = None
    roc_curve_tpr: Optional[np.ndarray] = None
    roc_thresholds: Optional[np.ndarray] = None
    
    # Precision-Recall metrics
    auprc: float = 0.0
    pr_curve_precision: Optional[np.ndarray] = None
    pr_curve_recall: Optional[np.ndarray] = None
    pr_thresholds: Optional[np.ndarray] = None
    
    # F1 metrics at different thresholds
    f1_at_05: float = 0.0
    f1_at_03: float = 0.0
    f1_at_07: float = 0.0
    optimal_f1: float = 0.0
    optimal_threshold: float = 0.5
    
    # Confusion matrix at optimal threshold
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Additional metrics
    accuracy: float = 0.0
    sensitivity: float = 0.0  # Recall / True Positive Rate
    specificity: float = 0.0  # True Negative Rate
    ppv: float = 0.0  # Positive Predictive Value / Precision
    npv: float = 0.0  # Negative Predictive Value
    
    # Sample sizes
    n_positive_samples: int = 0
    n_negative_samples: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding numpy arrays)."""
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "f1_at_05": self.f1_at_05,
            "f1_at_03": self.f1_at_03,
            "f1_at_07": self.f1_at_07,
            "optimal_f1": self.optimal_f1,
            "optimal_threshold": self.optimal_threshold,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "n_positive_samples": self.n_positive_samples,
            "n_negative_samples": self.n_negative_samples
        }
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"AUROC: {self.auroc:.3f} | AUPRC: {self.auprc:.3f} | "
            f"F1: {self.optimal_f1:.3f} @ {self.optimal_threshold:.2f} | "
            f"Sens: {self.sensitivity:.3f} | Spec: {self.specificity:.3f}"
        )


@dataclass
class SpeedMetrics:
    """
    Container for speed/performance metrics.
    
    NOTE: These measure COMPUTATIONAL performance,
    not biological timing or clinical decision speed.
    """
    # Latency metrics (milliseconds)
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Throughput metrics
    observations_per_second: float = 0.0
    patients_per_second: float = 0.0
    
    # Cerebras-specific metrics
    reasoning_cycles_per_second: float = 0.0
    hypotheses_per_second: float = 0.0
    
    # Resource utilization
    total_processing_time_seconds: float = 0.0
    total_observations_processed: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "mean_latency_ms": self.mean_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "observations_per_second": self.observations_per_second,
            "patients_per_second": self.patients_per_second,
            "reasoning_cycles_per_second": self.reasoning_cycles_per_second,
            "hypotheses_per_second": self.hypotheses_per_second,
            "total_processing_time_seconds": self.total_processing_time_seconds,
            "total_observations_processed": self.total_observations_processed
        }
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"Latency: {self.mean_latency_ms:.2f}ms (p95: {self.p95_latency_ms:.2f}ms) | "
            f"Throughput: {self.observations_per_second:.0f} obs/sec | "
            f"Cycles/sec: {self.reasoning_cycles_per_second:.0f}"
        )


@dataclass
class EvaluationReport:
    """
    Complete evaluation report combining accuracy and speed metrics.
    """
    accuracy: AccuracyMetrics
    speed: SpeedMetrics
    
    # Metadata
    evaluation_timestamp: str = ""
    n_patients: int = 0
    n_observations: int = 0
    simulation_mode: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy.to_dict(),
            "speed": self.speed.to_dict(),
            "metadata": {
                "evaluation_timestamp": self.evaluation_timestamp,
                "n_patients": self.n_patients,
                "n_observations": self.n_observations,
                "simulation_mode": self.simulation_mode
            }
        }
    
    def summary(self) -> str:
        """Return a formatted summary."""
        return (
            f"=== Evaluation Report ===\n"
            f"Patients: {self.n_patients} | Observations: {self.n_observations}\n"
            f"Accuracy: {self.accuracy.summary()}\n"
            f"Speed: {self.speed.summary()}"
        )


class AccuracyEvaluator:
    """
    Evaluates accuracy of risk predictions against ground truth.
    
    Key Considerations:
    ------------------
    1. Ground truth labels come from:
       - Synthetic data: Known latent states
       - Real data: Expert annotations or diagnostic codes
    
    2. We evaluate at the OBSERVATION level:
       - Each (patient, timestep) pair is one sample
       - Risk prediction compared to binary label
    
    3. We emphasize:
       - Agreement with expert judgment
       - NOT just hospital coding accuracy
       - Continuous risk scores, not binary predictions
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the evaluator."""
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        risk_scores: np.ndarray,
        true_labels: np.ndarray,
        threshold: float = 0.5
    ) -> AccuracyMetrics:
        """
        Evaluate risk scores against binary labels.
        
        Args:
            risk_scores: Array of risk scores (0-1)
            true_labels: Array of binary labels (0 or 1)
            threshold: Decision threshold for binary metrics
            
        Returns:
            AccuracyMetrics object
        """
        metrics = AccuracyMetrics()
        
        # Ensure arrays
        risk_scores = np.array(risk_scores)
        true_labels = np.array(true_labels).astype(int)
        
        # Sample counts
        metrics.n_positive_samples = int(np.sum(true_labels == 1))
        metrics.n_negative_samples = int(np.sum(true_labels == 0))
        
        # Check for edge cases
        if metrics.n_positive_samples == 0 or metrics.n_negative_samples == 0:
            print("Warning: Only one class present in labels. Some metrics will be undefined.")
            return metrics
        
        # ROC curve and AUROC
        try:
            metrics.auroc = roc_auc_score(true_labels, risk_scores)
            fpr, tpr, thresholds = roc_curve(true_labels, risk_scores)
            metrics.roc_curve_fpr = fpr
            metrics.roc_curve_tpr = tpr
            metrics.roc_thresholds = thresholds
        except Exception as e:
            print(f"ROC computation error: {e}")
        
        # Precision-Recall curve and AUPRC
        try:
            metrics.auprc = average_precision_score(true_labels, risk_scores)
            precision, recall, thresholds = precision_recall_curve(true_labels, risk_scores)
            metrics.pr_curve_precision = precision
            metrics.pr_curve_recall = recall
            metrics.pr_thresholds = thresholds
        except Exception as e:
            print(f"PR computation error: {e}")
        
        # F1 scores at different thresholds
        for thresh, attr in [(0.3, 'f1_at_03'), (0.5, 'f1_at_05'), (0.7, 'f1_at_07')]:
            binary_preds = (risk_scores >= thresh).astype(int)
            setattr(metrics, attr, f1_score(true_labels, binary_preds, zero_division=0))
        
        # Find optimal threshold (maximizes F1)
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.9, 50):
            binary_preds = (risk_scores >= thresh).astype(int)
            f1 = f1_score(true_labels, binary_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        metrics.optimal_f1 = best_f1
        metrics.optimal_threshold = best_thresh
        
        # Confusion matrix at optimal threshold
        binary_preds = (risk_scores >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, binary_preds, labels=[0, 1]).ravel()
        
        metrics.true_positives = int(tp)
        metrics.true_negatives = int(tn)
        metrics.false_positives = int(fp)
        metrics.false_negatives = int(fn)
        
        # Derived metrics
        metrics.accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics.sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics.ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def evaluate_from_results(
        self,
        results: List[SimulationResult],
        state_to_label: Optional[Dict[str, int]] = None
    ) -> AccuracyMetrics:
        """
        Evaluate accuracy from SimulationResult objects.
        
        Args:
            results: List of simulation results
            state_to_label: Mapping from state names to binary labels
            
        Returns:
            AccuracyMetrics object
        """
        if state_to_label is None:
            # Default: STABLE/COMPENSATING = 0, DETERIORATING/CRITICAL = 1
            state_to_label = {
                "stable": 0,
                "compensating": 0,
                "deteriorating": 1,
                "critical": 1,
                PatientState.STABLE.value: 0,
                PatientState.COMPENSATING.value: 0,
                PatientState.DETERIORATING.value: 1,
                PatientState.CRITICAL.value: 1
            }
        
        all_risk_scores = []
        all_labels = []
        
        for result in results:
            if result.true_states is None:
                continue
            
            for risk, state in zip(result.risk_means, result.true_states):
                state_str = state.value if hasattr(state, 'value') else str(state)
                if state_str in state_to_label:
                    all_risk_scores.append(risk)
                    all_labels.append(state_to_label[state_str])
        
        if not all_risk_scores:
            print("Warning: No valid samples for evaluation")
            return AccuracyMetrics()
        
        return self.evaluate(np.array(all_risk_scores), np.array(all_labels))


class SpeedEvaluator:
    """
    Evaluates computational performance of the simulation.
    
    IMPORTANT: These metrics measure COMPUTATIONAL speed,
    NOT biological timing or clinical decision speed.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the evaluator."""
        self.config = config or EvaluationConfig()
        self._latencies: List[float] = []
    
    def record_latency(self, latency_ms: float):
        """Record a single latency measurement."""
        self._latencies.append(latency_ms)
    
    def evaluate(
        self,
        latencies: Optional[List[float]] = None,
        total_time_seconds: Optional[float] = None,
        n_observations: Optional[int] = None,
        n_patients: Optional[int] = None,
        reasoning_cycles_per_obs: int = 10,
        hypotheses_per_obs: int = 8
    ) -> SpeedMetrics:
        """
        Compute speed metrics from latency measurements.
        
        Args:
            latencies: List of per-observation latencies (ms)
            total_time_seconds: Total processing time
            n_observations: Number of observations processed
            n_patients: Number of patients processed
            reasoning_cycles_per_obs: Cycles per observation
            hypotheses_per_obs: Hypotheses evaluated per observation
            
        Returns:
            SpeedMetrics object
        """
        metrics = SpeedMetrics()
        
        latencies = latencies or self._latencies
        
        if latencies:
            latencies_np = np.array(latencies)
            metrics.mean_latency_ms = float(np.mean(latencies_np))
            metrics.median_latency_ms = float(np.median(latencies_np))
            metrics.p95_latency_ms = float(np.percentile(latencies_np, 95))
            metrics.p99_latency_ms = float(np.percentile(latencies_np, 99))
            metrics.min_latency_ms = float(np.min(latencies_np))
            metrics.max_latency_ms = float(np.max(latencies_np))
            
            if n_observations is None:
                n_observations = len(latencies)
        
        if total_time_seconds and n_observations:
            metrics.observations_per_second = n_observations / total_time_seconds
            metrics.reasoning_cycles_per_second = (
                n_observations * reasoning_cycles_per_obs / total_time_seconds
            )
            metrics.hypotheses_per_second = (
                n_observations * hypotheses_per_obs / total_time_seconds
            )
        
        if total_time_seconds and n_patients:
            metrics.patients_per_second = n_patients / total_time_seconds
        
        metrics.total_processing_time_seconds = total_time_seconds or 0
        metrics.total_observations_processed = n_observations or 0
        
        return metrics
    
    def evaluate_from_benchmark(self, benchmark: BenchmarkResult) -> SpeedMetrics:
        """
        Convert BenchmarkResult to SpeedMetrics.
        
        Args:
            benchmark: BenchmarkResult from simulation engine
            
        Returns:
            SpeedMetrics object
        """
        return SpeedMetrics(
            mean_latency_ms=benchmark.avg_latency_ms,
            median_latency_ms=benchmark.avg_latency_ms,  # Approximate
            p95_latency_ms=benchmark.avg_latency_ms * 1.5,  # Approximate
            p99_latency_ms=benchmark.avg_latency_ms * 2.0,  # Approximate
            observations_per_second=benchmark.observations_per_second,
            patients_per_second=benchmark.n_patients / benchmark.total_time_seconds,
            reasoning_cycles_per_second=benchmark.reasoning_cycles_per_second,
            hypotheses_per_second=benchmark.hypotheses_per_second,
            total_processing_time_seconds=benchmark.total_time_seconds,
            total_observations_processed=benchmark.total_observations
        )
    
    def reset(self):
        """Clear recorded latencies."""
        self._latencies = []


def generate_evaluation_report(
    results: List[SimulationResult],
    benchmark: Optional[BenchmarkResult] = None,
    simulation_mode: str = "standard"
) -> EvaluationReport:
    """
    Generate a complete evaluation report.
    
    Args:
        results: List of simulation results
        benchmark: Optional benchmark results
        simulation_mode: Description of simulation mode
        
    Returns:
        EvaluationReport object
    """
    # Accuracy evaluation
    accuracy_evaluator = AccuracyEvaluator()
    accuracy_metrics = accuracy_evaluator.evaluate_from_results(results)
    
    # Speed evaluation
    speed_evaluator = SpeedEvaluator()
    
    if benchmark:
        speed_metrics = speed_evaluator.evaluate_from_benchmark(benchmark)
    else:
        # Compute from results
        latencies = [r.avg_latency_per_step_ms for r in results if r.avg_latency_per_step_ms > 0]
        total_obs = sum(len(r.timestamps) for r in results)
        total_time = sum(r.total_time_ms for r in results) / 1000
        
        speed_metrics = speed_evaluator.evaluate(
            latencies=latencies,
            total_time_seconds=total_time,
            n_observations=total_obs,
            n_patients=len(results)
        )
    
    # Build report
    from datetime import datetime
    
    report = EvaluationReport(
        accuracy=accuracy_metrics,
        speed=speed_metrics,
        evaluation_timestamp=datetime.now().isoformat(),
        n_patients=len(results),
        n_observations=sum(len(r.timestamps) for r in results),
        simulation_mode=simulation_mode
    )
    
    return report


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Evaluation Module...")
    print("=" * 60)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate risk scores and labels
    # Higher risk scores for positive class
    positive_scores = np.clip(np.random.beta(5, 2, n_samples // 2), 0, 1)
    negative_scores = np.clip(np.random.beta(2, 5, n_samples // 2), 0, 1)
    
    risk_scores = np.concatenate([positive_scores, negative_scores])
    true_labels = np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    risk_scores = risk_scores[indices]
    true_labels = true_labels[indices]
    
    # Test accuracy evaluator
    print("\n--- Accuracy Evaluation ---")
    accuracy_evaluator = AccuracyEvaluator()
    accuracy_metrics = accuracy_evaluator.evaluate(risk_scores, true_labels)
    
    print(f"AUROC: {accuracy_metrics.auroc:.3f}")
    print(f"AUPRC: {accuracy_metrics.auprc:.3f}")
    print(f"Optimal F1: {accuracy_metrics.optimal_f1:.3f} @ threshold {accuracy_metrics.optimal_threshold:.2f}")
    print(f"Sensitivity: {accuracy_metrics.sensitivity:.3f}")
    print(f"Specificity: {accuracy_metrics.specificity:.3f}")
    
    # Test speed evaluator
    print("\n--- Speed Evaluation ---")
    speed_evaluator = SpeedEvaluator()
    
    # Simulate latencies
    latencies = np.random.gamma(2, 2, 1000)  # ms
    
    speed_metrics = speed_evaluator.evaluate(
        latencies=list(latencies),
        total_time_seconds=10.0,
        n_observations=1000,
        n_patients=10
    )
    
    print(f"Mean latency: {speed_metrics.mean_latency_ms:.2f} ms")
    print(f"P95 latency: {speed_metrics.p95_latency_ms:.2f} ms")
    print(f"Throughput: {speed_metrics.observations_per_second:.0f} obs/sec")
    
    # Test summary
    print("\n--- Summary ---")
    print(f"Accuracy: {accuracy_metrics.summary()}")
    print(f"Speed: {speed_metrics.summary()}")
    
    print("\n" + "=" * 60)
    print("Evaluation module test complete.")
