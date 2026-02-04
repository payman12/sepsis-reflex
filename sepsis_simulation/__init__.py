"""
Sepsis Risk Simulation System
==============================

A real-time physiological simulation for early sepsis risk tracking,
optimized for Cerebras Cloud Compute.

This is a research prototype - NOT for clinical deployment.

Components:
- config: Configuration and constants
- synthetic_data_generator: High-resolution synthetic patient data
- enhanced_data_generator: MIMIC-based generator with 60+ measurements
- physiological_features: Deterministic feature extraction
- risk_state_tracker: Bayesian latent risk state
- cerebras_inference: Cerebras-optimized multi-cycle reasoning
- cerebras_cloud_integration: Full Cerebras Cloud API integration
- simulation_engine: Main orchestration
- evaluation: Accuracy and speed metrics
- dashboard: Streamlit visualization
- transformer_training_bridge: Train transformer on synthetic data
- expert_ai_comparison: Compare AI vs expert labels
- mimic_utilities_bridge: MIMIC-derived utilities for risk scoring
- unified_cli: Command-line interface for all components

Usage:
    from sepsis_simulation import SepsisSimulationEngine
    
    engine = SepsisSimulationEngine()
    septic, stable = engine.run_comparison(n_septic=10, n_stable=10)
    
    # Or use CLI:
    # python -m sepsis_simulation.unified_cli generate --n_patients 20
    # python -m sepsis_simulation.unified_cli train --n_patients 500
    # python -m sepsis_simulation.unified_cli compare --n_patients 20
"""

from .config import (
    CerebrasConfig,
    SimulationConfig,
    EvaluationConfig,
    PatientState,
    VITAL_SIGNS
)

from .synthetic_data_generator import (
    SyntheticDataGenerator,
    PatientTrajectory,
    generate_benchmark_dataset
)

from .physiological_features import (
    PhysiologicalFeatureExtractor,
    ExtractedFeatures
)

from .risk_state_tracker import (
    BayesianRiskTracker,
    RiskBelief,
    RiskRegime
)

from .cerebras_inference import (
    CerebrasClient,
    MultiCycleReasoner,
    ParallelPatientProcessor,
    CerebrasRiskEngine,
    CerebrasTransformerInference,
    CerebrasTransformerTraining,
    CerebrasTransformerInferenceCloud,
    CerebrasFeatureExtractor,
    CerebrasBatchProcessor,
    CerebrasMetrics,
    RiskAnalysisResult,
    InferenceMetrics,
    BatchInferenceResult,
    create_cerebras_engine
)

from .simulation_engine import (
    SepsisSimulationEngine,
    SimulationResult,
    BenchmarkResult
)

from .evaluation import (
    AccuracyEvaluator,
    SpeedEvaluator,
    AccuracyMetrics,
    SpeedMetrics,
    EvaluationReport,
    generate_evaluation_report
)

# New modules for enhanced functionality
from .enhanced_data_generator import (
    EnhancedMIMICDataGenerator,
    MeasurementSpec,
    generate_expert_vs_ai_comparison_data
)

from .transformer_training_bridge import (
    TransformerTrainingBridge,
    TrainingConfig,
    train_on_synthetic_data
)

from .expert_ai_comparison import (
    ExpertAIComparison,
    ComparisonConfig,
    ComparisonResult,
    run_expert_ai_comparison
)

from .mimic_utilities_bridge import (
    load_measurement_mappings,
    handle_outliers,
    calculate_sofa_score,
    calculate_sirs_score,
    calculate_shock_index,
    add_derived_features,
    EnhancedRiskScorer,
    calculate_classification_metrics
)

from .simple_transformer import SimpleTransformer

__version__ = "2.0.0"
__author__ = "Sepsis Simulation Team"

__all__ = [
    # Config
    "CerebrasConfig",
    "SimulationConfig", 
    "EvaluationConfig",
    "PatientState",
    "VITAL_SIGNS",
    
    # Data Generation (Original)
    "SyntheticDataGenerator",
    "PatientTrajectory",
    "generate_benchmark_dataset",
    
    # Data Generation (Enhanced MIMIC-based)
    "EnhancedMIMICDataGenerator",
    "MeasurementSpec",
    "generate_expert_vs_ai_comparison_data",
    
    # Features
    "PhysiologicalFeatureExtractor",
    "ExtractedFeatures",
    
    # Risk Tracking
    "BayesianRiskTracker",
    "RiskBelief",
    "RiskRegime",
    
    # Cerebras (unified)
    "CerebrasClient",
    "MultiCycleReasoner",
    "ParallelPatientProcessor",
    "CerebrasRiskEngine",
    "CerebrasTransformerInference",
    "CerebrasTransformerTraining",
    "CerebrasTransformerInferenceCloud",
    "CerebrasFeatureExtractor",
    "CerebrasBatchProcessor",
    "CerebrasMetrics",
    "RiskAnalysisResult",
    "InferenceMetrics",
    "BatchInferenceResult",
    "create_cerebras_engine",
    
    # Engine
    "SepsisSimulationEngine",
    "SimulationResult",
    "BenchmarkResult",
    
    # Evaluation
    "AccuracyEvaluator",
    "SpeedEvaluator",
    "AccuracyMetrics",
    "SpeedMetrics",
    "EvaluationReport",
    "generate_evaluation_report",
    
    # Transformer Training
    "TransformerTrainingBridge",
    "TrainingConfig",
    "train_on_synthetic_data",
    "SimpleTransformer",
    
    # Expert vs AI Comparison
    "ExpertAIComparison",
    "ComparisonConfig",
    "ComparisonResult",
    "run_expert_ai_comparison",
    
    # MIMIC Utilities
    "load_measurement_mappings",
    "handle_outliers",
    "calculate_sofa_score",
    "calculate_sirs_score",
    "calculate_shock_index",
    "add_derived_features",
    "EnhancedRiskScorer",
    "calculate_classification_metrics"
]
