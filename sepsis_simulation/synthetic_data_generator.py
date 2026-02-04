"""
High-Resolution Synthetic Patient Data Generator
=================================================

This module generates realistic physiological time series for sepsis simulation.
It creates minute-level or second-level patient trajectories with:
- Stable patients
- Gradually deteriorating patients  
- Patients entering sepsis-like physiological regimes

Design Philosophy:
- Simulation-first: Data for testing reasoning depth, not classifier training
- Physiologically realistic: Based on clinical knowledge of sepsis progression
- High temporal resolution: Minute or second-level timestamps
- Stochastic: Incorporates measurement noise and physiological variability

Why Synthetic Data is Critical:
1. Stress-tests multi-cycle reasoning under dense temporal input
2. Allows controlled experiments with known ground truth
3. Enables speed benchmarking with arbitrary data volume
4. No privacy concerns for research prototyping
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Generator
from datetime import datetime, timedelta
import warnings

try:
    from .config import (
        PatientState, 
        VitalSignConfig, 
        VITAL_SIGNS,
        SEPSIS_CORRELATION_MATRIX,
        STATE_TRANSITION_PROBS,
        SimulationConfig
    )
except ImportError:
    from config import (
        PatientState, 
        VitalSignConfig, 
        VITAL_SIGNS,
        SEPSIS_CORRELATION_MATRIX,
        STATE_TRANSITION_PROBS,
        SimulationConfig
    )


@dataclass
class PatientTrajectory:
    """
    Container for a single patient's time series data.
    
    Attributes:
        patient_id: Unique identifier
        timestamps: Array of datetime objects
        vitals: Dict mapping vital sign names to value arrays
        latent_states: Ground truth latent states (for evaluation only)
        metadata: Additional patient information
    """
    patient_id: str
    timestamps: np.ndarray
    vitals: Dict[str, np.ndarray]
    latent_states: np.ndarray  # Ground truth for evaluation
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.timestamps)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trajectory to pandas DataFrame."""
        data = {"timestamp": self.timestamps, "patient_id": self.patient_id}
        data.update(self.vitals)
        data["latent_state"] = self.latent_states
        return pd.DataFrame(data)
    
    def get_window(self, start_idx: int, end_idx: int) -> "PatientTrajectory":
        """Extract a time window from the trajectory."""
        return PatientTrajectory(
            patient_id=self.patient_id,
            timestamps=self.timestamps[start_idx:end_idx],
            vitals={k: v[start_idx:end_idx] for k, v in self.vitals.items()},
            latent_states=self.latent_states[start_idx:end_idx],
            metadata=self.metadata
        )


class SyntheticDataGenerator:
    """
    Generates high-resolution synthetic patient vital sign trajectories.
    
    The generator creates physiologically realistic data by:
    1. Simulating latent patient states (stable → compensating → deteriorating → critical)
    2. Generating correlated vital signs based on the latent state
    3. Adding realistic measurement noise and physiological variability
    4. Supporting minute-level or second-level temporal resolution
    
    This is NOT for training classifiers - it's for:
    - Testing continuous inference systems
    - Benchmarking computational performance
    - Validating reasoning depth under dense input
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        vital_configs: Optional[Dict[str, VitalSignConfig]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            config: Simulation configuration
            vital_configs: Vital sign configurations (defaults to VITAL_SIGNS)
            random_seed: Random seed for reproducibility
        """
        self.config = config or SimulationConfig()
        self.vital_configs = vital_configs or VITAL_SIGNS
        self.rng = np.random.default_rng(random_seed)
        
        # Precompute Cholesky decomposition for correlated sampling
        self._correlation_cholesky = np.linalg.cholesky(
            self._ensure_positive_definite(SEPSIS_CORRELATION_MATRIX)
        )
        
        # Vital sign order for correlation matrix
        self._vital_order = ["heart_rate", "map", "respiratory_rate", 
                            "spo2", "temperature", "lactate", "wbc"]
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite."""
        # Add small diagonal term if needed
        min_eigenvalue = np.min(np.linalg.eigvalsh(matrix))
        if min_eigenvalue < 1e-6:
            matrix = matrix + (1e-6 - min_eigenvalue) * np.eye(matrix.shape[0])
        return matrix
    
    def generate_patient(
        self,
        patient_id: str,
        scenario: str = "random",
        duration_hours: float = 48,
        timestep_seconds: int = 60
    ) -> PatientTrajectory:
        """
        Generate a single patient trajectory.
        
        Args:
            patient_id: Unique patient identifier
            scenario: One of "stable", "deteriorating", "sepsis", "random"
            duration_hours: Length of trajectory in hours
            timestep_seconds: Time between samples in seconds
            
        Returns:
            PatientTrajectory with complete vital sign time series
        """
        n_timesteps = int((duration_hours * 3600) / timestep_seconds)
        
        # Generate timestamps
        start_time = datetime.now()
        timestamps = np.array([
            start_time + timedelta(seconds=i * timestep_seconds)
            for i in range(n_timesteps)
        ])
        
        # Generate latent state trajectory
        latent_states = self._generate_state_trajectory(n_timesteps, scenario)
        
        # Generate vital signs based on states
        vitals = self._generate_vitals(latent_states)
        
        # Create metadata
        metadata = {
            "scenario": scenario,
            "duration_hours": duration_hours,
            "timestep_seconds": timestep_seconds,
            "generation_time": datetime.now().isoformat()
        }
        
        return PatientTrajectory(
            patient_id=patient_id,
            timestamps=timestamps,
            vitals=vitals,
            latent_states=latent_states,
            metadata=metadata
        )
    
    def _generate_state_trajectory(
        self, 
        n_timesteps: int, 
        scenario: str
    ) -> np.ndarray:
        """
        Generate a sequence of latent patient states.
        
        The state trajectory determines the underlying physiological regime,
        which then influences all vital sign values.
        """
        states = np.empty(n_timesteps, dtype=object)
        
        if scenario == "stable":
            # Patient remains stable throughout
            states[:] = PatientState.STABLE
            # Small chance of brief compensating periods
            for i in range(n_timesteps):
                if self.rng.random() < 0.02:
                    duration = min(self.rng.integers(5, 30), n_timesteps - i)
                    states[i:i+duration] = PatientState.COMPENSATING
                    
        elif scenario == "deteriorating":
            # Gradual progression toward critical state
            transition_point = self.rng.integers(n_timesteps // 4, n_timesteps // 2)
            critical_point = self.rng.integers(2 * n_timesteps // 3, n_timesteps)
            
            states[:transition_point] = PatientState.STABLE
            states[transition_point:critical_point] = PatientState.DETERIORATING
            states[critical_point:] = PatientState.CRITICAL
            
            # Add some back-and-forth in the deteriorating phase
            for i in range(transition_point, critical_point):
                if self.rng.random() < 0.1:
                    states[i] = PatientState.COMPENSATING
                    
        elif scenario == "sepsis":
            # Sepsis-like trajectory: compensation → rapid deterioration
            compensation_start = self.rng.integers(n_timesteps // 6, n_timesteps // 3)
            deterioration_start = self.rng.integers(n_timesteps // 3, n_timesteps // 2)
            critical_start = self.rng.integers(n_timesteps // 2, 3 * n_timesteps // 4)
            
            states[:compensation_start] = PatientState.STABLE
            states[compensation_start:deterioration_start] = PatientState.COMPENSATING
            states[deterioration_start:critical_start] = PatientState.DETERIORATING
            states[critical_start:] = PatientState.CRITICAL
            
        else:  # random - use Markov chain
            states[0] = PatientState.STABLE
            for i in range(1, n_timesteps):
                current_state = states[i-1]
                transition_probs = STATE_TRANSITION_PROBS[current_state]
                next_states = list(transition_probs.keys())
                probs = list(transition_probs.values())
                states[i] = self.rng.choice(next_states, p=probs)
        
        return states
    
    def _generate_vitals(
        self, 
        latent_states: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate vital sign trajectories based on latent states.
        
        This creates physiologically realistic, correlated vital signs
        that reflect the underlying patient state.
        """
        n_timesteps = len(latent_states)
        vitals = {}
        
        # State-dependent vital sign modifiers
        state_modifiers = {
            PatientState.STABLE: {
                "heart_rate": (0, 1.0),      # (mean_shift, std_multiplier)
                "map": (0, 1.0),
                "respiratory_rate": (0, 1.0),
                "spo2": (0, 1.0),
                "temperature": (0, 1.0),
                "lactate": (0, 1.0),
                "wbc": (0, 1.0)
            },
            PatientState.COMPENSATING: {
                "heart_rate": (10, 1.2),      # Mild tachycardia
                "map": (-5, 1.1),             # Slight hypotension
                "respiratory_rate": (3, 1.3), # Tachypnea
                "spo2": (-1, 1.5),            # Mild desaturation
                "temperature": (0.5, 1.2),    # Low-grade fever
                "lactate": (0.5, 1.3),        # Mild elevation
                "wbc": (2, 1.4)               # Leukocytosis
            },
            PatientState.DETERIORATING: {
                "heart_rate": (25, 1.5),
                "map": (-15, 1.3),
                "respiratory_rate": (8, 1.5),
                "spo2": (-4, 2.0),
                "temperature": (1.5, 1.5),
                "lactate": (1.5, 1.5),
                "wbc": (5, 1.6)
            },
            PatientState.CRITICAL: {
                "heart_rate": (40, 2.0),
                "map": (-25, 1.5),
                "respiratory_rate": (12, 2.0),
                "spo2": (-8, 2.5),
                "temperature": (2.0, 2.0),
                "lactate": (3.0, 2.0),
                "wbc": (8, 2.0)
            }
        }
        
        # Generate base values with temporal correlation
        for vital_name, vital_config in self.vital_configs.items():
            values = np.zeros(n_timesteps)
            inertia = vital_config.physiological_inertia
            
            # Initialize with normal value
            values[0] = vital_config.normal_mean + self.rng.normal(0, vital_config.normal_std)
            
            for i in range(1, n_timesteps):
                state = latent_states[i]
                modifier = state_modifiers[state].get(vital_name, (0, 1.0))
                mean_shift, std_mult = modifier
                
                # Target value based on state
                target = vital_config.normal_mean + mean_shift
                target += self.rng.normal(0, vital_config.normal_std * std_mult)
                
                # Apply physiological inertia (smoothing)
                values[i] = inertia * values[i-1] + (1 - inertia) * target
                
                # Add measurement noise
                values[i] += self.rng.normal(0, vital_config.sampling_noise_std)
                
                # Clip to physiological bounds
                values[i] = np.clip(
                    values[i],
                    vital_config.critical_low * 0.8,
                    vital_config.critical_high * 1.2
                )
            
            vitals[vital_name] = values
        
        return vitals
    
    def generate_cohort(
        self,
        n_patients: int,
        scenario_distribution: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> List[PatientTrajectory]:
        """
        Generate a cohort of patients with specified scenario distribution.
        
        Args:
            n_patients: Number of patients to generate
            scenario_distribution: Dict mapping scenario names to proportions
            **kwargs: Additional arguments passed to generate_patient
            
        Returns:
            List of PatientTrajectory objects
        """
        if scenario_distribution is None:
            scenario_distribution = {
                "stable": 0.4,
                "deteriorating": 0.25,
                "sepsis": 0.25,
                "random": 0.1
            }
        
        # Normalize distribution
        total = sum(scenario_distribution.values())
        scenario_distribution = {k: v/total for k, v in scenario_distribution.items()}
        
        patients = []
        scenarios = list(scenario_distribution.keys())
        probs = list(scenario_distribution.values())
        
        for i in range(n_patients):
            scenario = self.rng.choice(scenarios, p=probs)
            patient = self.generate_patient(
                patient_id=f"SYN_{i:05d}",
                scenario=scenario,
                **kwargs
            )
            patients.append(patient)
        
        return patients
    
    def stream_patient_data(
        self,
        patient: PatientTrajectory,
        chunk_size: int = 1
    ) -> Generator[Tuple[np.ndarray, Dict[str, float]], None, None]:
        """
        Stream patient data one timestep at a time.
        
        This simulates real-time data ingestion for testing
        continuous inference systems.
        
        Yields:
            Tuple of (timestamp, dict of vital sign values)
        """
        for i in range(len(patient)):
            timestamp = patient.timestamps[i]
            vitals = {k: v[i] for k, v in patient.vitals.items()}
            yield timestamp, vitals


def generate_benchmark_dataset(
    n_patients: int = 100,
    duration_hours: float = 48,
    timestep_seconds: int = 60,
    output_path: Optional[str] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate a complete benchmark dataset for evaluation.
    
    Args:
        n_patients: Number of patients
        duration_hours: Duration per patient
        timestep_seconds: Temporal resolution
        output_path: Optional path to save CSV
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with all patient trajectories
    """
    generator = SyntheticDataGenerator(random_seed=random_seed)
    
    patients = generator.generate_cohort(
        n_patients=n_patients,
        duration_hours=duration_hours,
        timestep_seconds=timestep_seconds
    )
    
    # Combine all trajectories
    dfs = [p.to_dataframe() for p in patients]
    combined = pd.concat(dfs, ignore_index=True)
    
    # Add derived features
    combined["latent_state_numeric"] = combined["latent_state"].apply(
        lambda x: {
            PatientState.STABLE: 0,
            PatientState.COMPENSATING: 1,
            PatientState.DETERIORATING: 2,
            PatientState.CRITICAL: 3
        }.get(x, 0)
    )
    
    if output_path:
        combined.to_csv(output_path, index=False)
        print(f"Saved benchmark dataset to {output_path}")
    
    return combined


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate single patient
    patient = generator.generate_patient(
        patient_id="TEST_001",
        scenario="sepsis",
        duration_hours=24,
        timestep_seconds=60
    )
    
    print(f"\nGenerated patient: {patient.patient_id}")
    print(f"  Timesteps: {len(patient)}")
    print(f"  Duration: {patient.metadata['duration_hours']} hours")
    print(f"  Scenario: {patient.metadata['scenario']}")
    print(f"\nVital signs: {list(patient.vitals.keys())}")
    
    # Show sample data
    df = patient.to_dataframe()
    print(f"\nSample data (first 5 rows):")
    print(df.head())
    
    # Generate cohort
    print("\n\nGenerating patient cohort...")
    patients = generator.generate_cohort(n_patients=10, duration_hours=12)
    print(f"Generated {len(patients)} patients")
    
    # Scenario distribution
    scenarios = [p.metadata["scenario"] for p in patients]
    print(f"Scenarios: {dict(zip(*np.unique(scenarios, return_counts=True)))}")
