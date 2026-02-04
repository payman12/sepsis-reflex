"""
Configuration and Constants for Sepsis Risk Simulation System
==============================================================

This module defines all configuration parameters, physiological constants,
and system-wide settings for the real-time sepsis risk tracking simulation.

Design Philosophy:
- Simulation-first: All parameters tuned for research prototype behavior
- Cerebras-optimized: Settings that leverage massive parallelism
- Uncertainty-aware: Parameters for probabilistic reasoning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class PatientState(Enum):
    """
    Patient physiological regime states for synthetic generation.
    These represent latent states, not diagnostic labels.
    """
    STABLE = "stable"
    COMPENSATING = "compensating"  # Early stress response
    DETERIORATING = "deteriorating"  # Progressive decline
    CRITICAL = "critical"  # Severe physiological derangement


@dataclass
class VitalSignConfig:
    """Configuration for a single vital sign."""
    name: str
    unit: str
    normal_mean: float
    normal_std: float
    critical_low: float
    critical_high: float
    sampling_noise_std: float  # Measurement noise
    physiological_inertia: float  # How slowly the signal changes (0-1)
    
    def is_abnormal(self, value: float) -> bool:
        """Check if value is outside normal range."""
        return value < self.critical_low or value > self.critical_high


@dataclass
class CerebrasConfig:
    """
    Cerebras Cloud Compute configuration.
    
    Why Cerebras for this task:
    - Single-cycle memory access: No GPU memory hierarchy bottlenecks
    - Massive parallelism: Run 10-20 reasoning cycles per timestep simultaneously
    - Low latency: Sub-millisecond inference enables real-time streaming
    - Wafer-scale: Process hundreds of patient streams in parallel
    """
    api_key: Optional[str] = None
    model_id: str = "llama3.1-8b"  # For optional LLM-based reasoning
    
    # Multi-cycle reasoning parameters
    # These settings exploit Cerebras' ability to run many forward passes cheaply
    reasoning_cycles_per_timestep: int = 10  # Number of belief updates per new observation
    parallel_hypotheses: int = 8  # Parallel "what-if" scenarios evaluated
    future_horizon_steps: int = 5  # Short-horizon future simulations
    
    # Batch processing for throughput
    max_batch_size: int = 128  # Patients processed in parallel
    
    # API settings
    base_url: str = "https://api.cerebras.ai/v1"
    timeout_seconds: float = 30.0
    
    # Why these settings benefit from Cerebras:
    # 1. reasoning_cycles_per_timestep: On GPUs, each cycle adds latency due to
    #    kernel launch overhead and memory transfers. Cerebras' dataflow architecture
    #    makes sequential cycles nearly free.
    # 2. parallel_hypotheses: GPUs struggle with small-batch parallel inference.
    #    Cerebras can evaluate 8 hypotheses as cheaply as 1.
    # 3. future_horizon_steps: Autoregressive rollouts are memory-bound on GPUs.
    #    Cerebras' on-chip SRAM eliminates this bottleneck.


@dataclass
class SimulationConfig:
    """
    Core simulation parameters.
    
    Design rationale:
    - High temporal resolution (second-level) to capture rapid physiological changes
    - 48-hour window matches typical ICU monitoring periods
    - Uncertainty tracking at every timestep
    """
    # Temporal resolution
    timestep_seconds: int = 60  # 1-minute resolution (can be 1 for second-level)
    simulation_window_hours: int = 48
    
    # Risk state parameters
    initial_risk_mean: float = 0.1  # Prior belief: most patients are low-risk
    initial_risk_std: float = 0.15  # Moderate initial uncertainty
    
    # Trend detection windows
    short_trend_window: int = 15  # 15 timesteps for acute changes
    medium_trend_window: int = 60  # 1 hour for gradual trends
    long_trend_window: int = 360  # 6 hours for sustained patterns
    
    # Regime shift detection
    regime_shift_threshold: float = 2.5  # Std deviations for regime change
    acceleration_threshold: float = 0.1  # Rate of change threshold
    
    # Uncertainty bounds
    min_uncertainty: float = 0.01  # Never be too certain
    max_uncertainty: float = 0.5  # Cap uncertainty for stability
    
    @property
    def total_timesteps(self) -> int:
        """Total number of timesteps in simulation window."""
        return (self.simulation_window_hours * 3600) // self.timestep_seconds


@dataclass
class EvaluationConfig:
    """
    Evaluation and benchmarking configuration.
    
    Note: Speed metrics measure COMPUTATIONAL performance,
    not biological onset timing or clinical decision speed.
    """
    # Accuracy metrics
    auroc_threshold: float = 0.75  # Minimum acceptable AUROC
    
    # Speed benchmarking
    warmup_iterations: int = 10  # Warmup before timing
    benchmark_iterations: int = 100  # Iterations for timing
    
    # Throughput targets (Cerebras should exceed these easily)
    target_latency_ms: float = 10.0  # Per-timestep inference
    target_throughput_patients: int = 1000  # Patients/second


# =============================================================================
# PHYSIOLOGICAL CONSTANTS
# =============================================================================

# Vital sign configurations based on clinical literature
VITAL_SIGNS: Dict[str, VitalSignConfig] = {
    "heart_rate": VitalSignConfig(
        name="Heart Rate",
        unit="bpm",
        normal_mean=75.0,
        normal_std=12.0,
        critical_low=40.0,
        critical_high=150.0,
        sampling_noise_std=2.0,
        physiological_inertia=0.85
    ),
    "map": VitalSignConfig(
        name="Mean Arterial Pressure",
        unit="mmHg",
        normal_mean=85.0,
        normal_std=10.0,
        critical_low=60.0,
        critical_high=120.0,
        sampling_noise_std=3.0,
        physiological_inertia=0.9
    ),
    "respiratory_rate": VitalSignConfig(
        name="Respiratory Rate",
        unit="breaths/min",
        normal_mean=16.0,
        normal_std=3.0,
        critical_low=8.0,
        critical_high=30.0,
        sampling_noise_std=1.0,
        physiological_inertia=0.8
    ),
    "spo2": VitalSignConfig(
        name="SpO2",
        unit="%",
        normal_mean=97.0,
        normal_std=1.5,
        critical_low=88.0,
        critical_high=100.0,
        sampling_noise_std=0.5,
        physiological_inertia=0.92
    ),
    "temperature": VitalSignConfig(
        name="Temperature",
        unit="°C",
        normal_mean=37.0,
        normal_std=0.3,
        critical_low=35.0,
        critical_high=39.5,
        sampling_noise_std=0.1,
        physiological_inertia=0.95
    ),
    "lactate": VitalSignConfig(
        name="Lactate",
        unit="mmol/L",
        normal_mean=1.0,
        normal_std=0.3,
        critical_low=0.3,
        critical_high=4.0,
        sampling_noise_std=0.1,
        physiological_inertia=0.88
    ),
    "wbc": VitalSignConfig(
        name="White Blood Cell Count",
        unit="K/uL",
        normal_mean=7.5,
        normal_std=2.0,
        critical_low=2.0,
        critical_high=20.0,
        sampling_noise_std=0.3,
        physiological_inertia=0.92
    )
}

# Sepsis-related physiological correlations
# These define how vital signs co-vary during deterioration
SEPSIS_CORRELATION_MATRIX = np.array([
    # HR    MAP    RR    SpO2   Temp  Lactate  WBC
    [1.0,  -0.4,  0.5,  -0.3,   0.3,   0.4,    0.3],  # HR
    [-0.4,  1.0, -0.3,   0.4,  -0.2,  -0.5,   -0.2],  # MAP
    [0.5,  -0.3,  1.0,  -0.5,   0.3,   0.4,    0.2],  # RR
    [-0.3,  0.4, -0.5,   1.0,  -0.2,  -0.4,   -0.1],  # SpO2
    [0.3,  -0.2,  0.3,  -0.2,   1.0,   0.3,    0.5],  # Temp
    [0.4,  -0.5,  0.4,  -0.4,   0.3,   1.0,    0.3],  # Lactate
    [0.3,  -0.2,  0.2,  -0.1,   0.5,   0.3,    1.0],  # WBC
])

# State transition probabilities for synthetic data generation
STATE_TRANSITION_PROBS = {
    PatientState.STABLE: {
        PatientState.STABLE: 0.95,
        PatientState.COMPENSATING: 0.04,
        PatientState.DETERIORATING: 0.01,
        PatientState.CRITICAL: 0.00
    },
    PatientState.COMPENSATING: {
        PatientState.STABLE: 0.10,
        PatientState.COMPENSATING: 0.70,
        PatientState.DETERIORATING: 0.18,
        PatientState.CRITICAL: 0.02
    },
    PatientState.DETERIORATING: {
        PatientState.STABLE: 0.02,
        PatientState.COMPENSATING: 0.08,
        PatientState.DETERIORATING: 0.65,
        PatientState.CRITICAL: 0.25
    },
    PatientState.CRITICAL: {
        PatientState.STABLE: 0.01,
        PatientState.COMPENSATING: 0.04,
        PatientState.DETERIORATING: 0.15,
        PatientState.CRITICAL: 0.80
    }
}


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

def get_default_config() -> Dict:
    """Return default configuration dictionary."""
    return {
        "cerebras": CerebrasConfig(),
        "simulation": SimulationConfig(),
        "evaluation": EvaluationConfig(),
        "vital_signs": VITAL_SIGNS
    }
