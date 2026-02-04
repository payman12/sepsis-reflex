"""
Main Simulation Orchestration Engine
=====================================

This module coordinates all components of the sepsis risk simulation:
- Data ingestion (synthetic or real datasets)
- Feature extraction
- Multi-cycle reasoning
- Risk state tracking
- Metrics collection

Design Philosophy:
- Simulation-first: No clinical deployment claims
- Streaming: Process data as it arrives
- Modular: Each component is independently testable
- Observable: Full metrics and state visibility

Key Features:
1. Real-time streaming simulation
2. Batch replay of historical trajectories
3. Comparison mode (septic vs non-septic)
4. Speed benchmarking mode
"""

import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Generator, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

try:
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
        InferenceClient,
        LocalSimulator,
        SepsisRiskAnalyzer,
        InferenceResult,
        LatencyMetrics,
        # Legacy compatibility
        CerebrasClient,
        CerebrasRiskEngine,
        MultiCycleReasoner,
        ParallelPatientProcessor,
        InferenceMetrics,
        RiskAnalysisResult
    )
except ImportError:
    from config import (
        CerebrasConfig, 
        SimulationConfig, 
        EvaluationConfig,
        PatientState,
        VITAL_SIGNS
    )
    from synthetic_data_generator import (
        SyntheticDataGenerator,
        PatientTrajectory,
        generate_benchmark_dataset
    )
    from physiological_features import (
        PhysiologicalFeatureExtractor,
        ExtractedFeatures
    )
    from risk_state_tracker import (
        BayesianRiskTracker,
        RiskBelief,
        RiskRegime
    )
    from cerebras_inference import (
        InferenceClient,
        LocalSimulator,
        SepsisRiskAnalyzer,
        InferenceResult,
        LatencyMetrics,
        # Legacy compatibility
        CerebrasClient,
        CerebrasRiskEngine,
        MultiCycleReasoner,
        ParallelPatientProcessor,
        InferenceMetrics,
        RiskAnalysisResult
    )


@dataclass
class SimulationResult:
    """Complete results from a simulation run."""
    patient_id: str
    
    # Time series data
    timestamps: List[float]
    risk_means: List[float]
    risk_stds: List[float]
    risk_regimes: List[str]
    
    # Features over time
    abnormality_scores: List[float]
    instability_scores: List[float]
    
    # Ground truth (if available)
    true_states: Optional[List[str]] = None
    
    # Metrics
    total_time_ms: float = 0.0
    avg_latency_per_step_ms: float = 0.0
    total_reasoning_cycles: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        data = {
            "timestamp": self.timestamps,
            "risk_mean": self.risk_means,
            "risk_std": self.risk_stds,
            "risk_regime": self.risk_regimes,
            "abnormality_score": self.abnormality_scores,
            "instability_score": self.instability_scores
        }
        if self.true_states:
            data["true_state"] = self.true_states
        
        df = pd.DataFrame(data)
        df["patient_id"] = self.patient_id
        return df


@dataclass
class BenchmarkResult:
    """Results from a speed benchmark."""
    n_patients: int
    n_timesteps_per_patient: int
    total_observations: int
    
    # Timing
    total_time_seconds: float
    observations_per_second: float
    avg_latency_ms: float
    
    # Cerebras-specific
    reasoning_cycles_per_second: float
    hypotheses_per_second: float
    
    def to_dict(self) -> Dict:
        return {
            "n_patients": self.n_patients,
            "n_timesteps_per_patient": self.n_timesteps_per_patient,
            "total_observations": self.total_observations,
            "total_time_seconds": self.total_time_seconds,
            "observations_per_second": self.observations_per_second,
            "avg_latency_ms": self.avg_latency_ms,
            "reasoning_cycles_per_second": self.reasoning_cycles_per_second,
            "hypotheses_per_second": self.hypotheses_per_second
        }


class SepsisSimulationEngine:
    """
    Main orchestrator for sepsis risk simulation.
    
    Modes of Operation:
    ------------------
    1. STREAMING: Process observations one at a time
    2. BATCH: Process entire trajectories at once
    3. COMPARISON: Run parallel septic/non-septic simulations
    4. BENCHMARK: Measure computational performance
    
    Usage:
        engine = SepsisSimulationEngine()
        
        # Streaming mode
        for obs in data_stream:
            result = engine.process_observation(patient_id, obs)
        
        # Batch mode
        results = engine.run_batch(patient_trajectories)
        
        # Benchmark mode
        benchmark = engine.run_speed_benchmark(n_patients=100)
    """
    
    def __init__(
        self,
        cerebras_config: Optional[CerebrasConfig] = None,
        simulation_config: Optional[SimulationConfig] = None,
        cerebras_api_key: Optional[str] = None,
        use_cerebras_llm: bool = True,
        model_tier: str = "balanced"
    ):
        """
        Initialize the simulation engine.
        
        ARCHITECTURE:
        -------------
        - ALL simulation logic runs LOCALLY (time-series, features, visualization)
        - Cerebras is used ONLY for inference (stateless API calls)
        - Clean separation: InferenceClient (remote) + LocalSimulator (local)
        
        Args:
            cerebras_config: Cerebras Cloud configuration
            simulation_config: Simulation parameters
            cerebras_api_key: API key for Cerebras Cloud (for inference only)
            use_cerebras_llm: Whether to use Cerebras for inference
            model_tier: Model tier ("fast", "balanced", "accurate")
        """
        self.cerebras_config = cerebras_config or CerebrasConfig()
        self.simulation_config = simulation_config or SimulationConfig()
        self._use_cerebras_llm = use_cerebras_llm
        self._model_tier = model_tier
        
        # Override API key if provided
        if cerebras_api_key:
            self.cerebras_config.api_key = cerebras_api_key
        
        # ==========================================================
        # NEW CLEAN ARCHITECTURE
        # ==========================================================
        
        # Inference Client - ONLY makes API calls to Cerebras
        # This is the ONLY thing that talks to Cerebras Cloud
        self._inference_client = InferenceClient(
            api_key=self.cerebras_config.api_key,
            model_tier=model_tier
        )
        
        # Local Simulator - ALL local computation
        # Time-series, windowing, normalization, aggregation
        self._local_simulator = LocalSimulator(
            window_size=10,
            time_resolution_seconds=60.0
        )
        
        # Combined analyzer for convenience
        self._risk_analyzer = SepsisRiskAnalyzer(
            api_key=self.cerebras_config.api_key,
            model_tier=model_tier,
            window_size=10
        )
        
        # ==========================================================
        # LOCAL COMPONENTS (never sent to Cerebras)
        # ==========================================================
        
        # Feature extractors - LOCAL
        self._feature_extractors: Dict[str, PhysiologicalFeatureExtractor] = {}
        
        # Risk trackers - LOCAL Bayesian state tracking
        self._risk_trackers: Dict[str, BayesianRiskTracker] = {}
        
        # Synthetic data generator - LOCAL
        self._data_generator = SyntheticDataGenerator(
            config=self.simulation_config
        )
        
        # Results storage - LOCAL
        self._simulation_results: Dict[str, SimulationResult] = {}
        
        # ==========================================================
        # METRICS (for performance evaluation)
        # ==========================================================
        self._total_observations_processed = 0
        self._total_processing_time_ms = 0.0
        self._total_cerebras_api_calls = 0
        self._cerebras_api_latency_ms = 0.0
        self._total_serialization_ms = 0.0
        self._total_parsing_ms = 0.0
        
        # Log initialization status
        if self._inference_client.is_available():
            print(f"[ENGINE] ✅ Cerebras inference enabled (model: {self._inference_client.model})")
            print(f"[ENGINE] Local simulation + Remote inference architecture active")
        else:
            print(f"[ENGINE] ⚠️ Running fully local (Cerebras API not available)")
            print(f"[ENGINE] Status: {self._inference_client.get_status()}")
    
    @property
    def is_using_cerebras_cloud(self) -> bool:
        """Check if the engine is using Cerebras Cloud API for inference."""
        return self._inference_client.is_available()
    
    @property
    def cerebras_mode_status(self) -> str:
        """Get a human-readable status of Cerebras connection."""
        return self._inference_client.get_status()
    
    @property
    def inference_client(self) -> InferenceClient:
        """Get the inference client for direct access."""
        return self._inference_client
    
    @property
    def local_simulator(self) -> LocalSimulator:
        """Get the local simulator for direct access."""
        return self._local_simulator
    
    def _get_or_create_extractor(self, patient_id: str) -> PhysiologicalFeatureExtractor:
        """Get or create feature extractor for a patient."""
        if patient_id not in self._feature_extractors:
            self._feature_extractors[patient_id] = PhysiologicalFeatureExtractor(
                config=self.simulation_config
            )
        return self._feature_extractors[patient_id]
    
    def _get_or_create_tracker(self, patient_id: str) -> BayesianRiskTracker:
        """Get or create risk tracker for a patient."""
        if patient_id not in self._risk_trackers:
            self._risk_trackers[patient_id] = BayesianRiskTracker(
                config=self.simulation_config
            )
        return self._risk_trackers[patient_id]
    
    def process_observation(
        self,
        patient_id: str,
        timestamp: float,
        vital_values: Dict[str, float],
        n_reasoning_cycles: Optional[int] = None,
        use_cerebras_api: bool = True
    ) -> Tuple[RiskBelief, ExtractedFeatures]:
        """
        Process a single observation for a patient.
        
        ARCHITECTURE:
        -------------
        1. Extract features from vital signs (LOCAL)
        2. Add to local trajectory (LOCAL)
        3. Extract window for inference (LOCAL)
        4. Call Cerebras for inference (REMOTE, if enabled)
        5. Update risk belief (LOCAL)
        6. Return updated state
        
        Args:
            patient_id: Patient identifier
            timestamp: Observation timestamp (real time, continuous axis)
            vital_values: Dict of vital sign values
            n_reasoning_cycles: Override default cycle count (legacy, ignored)
            use_cerebras_api: Whether to call Cerebras API for inference
            
        Returns:
            Tuple of (updated_belief, extracted_features)
        """
        start_time = time.perf_counter()
        
        # ==========================================================
        # STEP 1: LOCAL - Extract features from vital signs
        # ==========================================================
        extractor = self._get_or_create_extractor(patient_id)
        tracker = self._get_or_create_tracker(patient_id)
        features = extractor.extract(timestamp, vital_values)
        
        # ==========================================================
        # STEP 2: LOCAL - Update risk belief using Bayesian tracker
        # This properly tracks state over time with trends!
        # ==========================================================
        if use_cerebras_api and self._inference_client.is_available():
            # Call Cerebras API for inference
            patient_data = self._local_simulator.add_reading(
                patient_id=patient_id,
                timestamp=timestamp,
                vitals=vital_values
            )
            
            inference_result = self._inference_client.infer(patient_data)
            
            # Track latency metrics
            self._total_cerebras_api_calls += 1
            self._cerebras_api_latency_ms += inference_result.latency.api_roundtrip_ms
            self._total_serialization_ms += inference_result.latency.serialization_ms
            self._total_parsing_ms += inference_result.latency.parsing_ms
            
            # Update tracker with API result
            belief = RiskBelief(
                mean=inference_result.risk_score,
                variance=inference_result.uncertainty ** 2,
                trend=tracker.belief.trend,  # Keep tracked trend
                acceleration=tracker.belief.acceleration
            )
            tracker.belief = belief
        else:
            # LOCAL - Use Bayesian tracker with extracted features
            # This produces realistic, varying risk scores over time!
            n_cycles = self.cerebras_config.reasoning_cycles_per_timestep
            belief = tracker.update(features, n_cycles=n_cycles)
            
            # Also add to local simulator for visualization
            self._local_simulator.add_reading(
                patient_id=patient_id,
                timestamp=timestamp,
                vitals=vital_values
            )
        
        # Update metrics
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        self._total_observations_processed += 1
        self._total_processing_time_ms += processing_time_ms
        
        return belief, features
    
    def run_patient_trajectory(
        self,
        trajectory: PatientTrajectory,
        progress_callback: Optional[callable] = None,
        use_cerebras_final: bool = True
    ) -> SimulationResult:
        """
        Run simulation on a complete patient trajectory.
        
        FAST EXECUTION:
        - ALL timestep processing runs LOCALLY (instant)
        - ONE Cerebras API call at the END for final risk classification
        
        Args:
            trajectory: PatientTrajectory object
            progress_callback: Optional callback(step, total) for progress
            use_cerebras_final: Make ONE API call at end for final classification
            
        Returns:
            SimulationResult with complete time series
        """
        patient_id = trajectory.patient_id
        n_steps = len(trajectory)
        
        # Initialize result containers
        timestamps = []
        risk_means = []
        risk_stds = []
        risk_regimes = []
        abnormality_scores = []
        instability_scores = []
        true_states = []
        
        total_start = time.perf_counter()
        
        # ==========================================================
        # STEP 1: Process ALL timesteps LOCALLY (fast!)
        # ==========================================================
        for i in range(n_steps):
            vital_values = {k: v[i] for k, v in trajectory.vitals.items()}
            timestamp = float(i)
            
            # ALL processing is LOCAL - no API calls during loop
            belief, features = self.process_observation(
                patient_id,
                timestamp,
                vital_values,
                use_cerebras_api=False  # Never call API during loop
            )
            
            timestamps.append(timestamp)
            risk_means.append(belief.mean)
            risk_stds.append(np.sqrt(belief.variance))
            risk_regimes.append(belief.regime.value)
            abnormality_scores.append(features.abnormality_score)
            instability_scores.append(features.instability_score)
            
            if trajectory.latent_states is not None:
                state = trajectory.latent_states[i]
                true_states.append(state.value if hasattr(state, 'value') else str(state))
            
            if progress_callback:
                progress_callback(i + 1, n_steps)
        
        # ==========================================================
        # STEP 2: ONE Cerebras API call for final classification
        # ==========================================================
        if use_cerebras_final and self._inference_client.is_available():
            # Get final vitals and recent history for API call
            final_vitals = {k: v[-1] for k, v in trajectory.vitals.items()}
            recent_history = [
                {k: v[i] for k, v in trajectory.vitals.items()}
                for i in range(max(0, n_steps - 6), n_steps)
            ]
            
            # Make ONE API call with patient summary
            patient_data = {
                "patient_id": patient_id,
                "vitals": final_vitals,
                "window": recent_history,
                "metrics_available": list(final_vitals.keys())
            }
            
            api_start = time.perf_counter()
            inference_result = self._inference_client.infer(patient_data)
            api_latency = (time.perf_counter() - api_start) * 1000
            
            # Update final risk score with Cerebras result
            risk_means[-1] = inference_result.risk_score
            risk_stds[-1] = inference_result.uncertainty
            
            # Track API metrics
            self._total_cerebras_api_calls += 1
            self._cerebras_api_latency_ms += api_latency
            self._total_serialization_ms += inference_result.latency.serialization_ms
            self._total_parsing_ms += inference_result.latency.parsing_ms
        
        total_time_ms = (time.perf_counter() - total_start) * 1000
        
        result = SimulationResult(
            patient_id=patient_id,
            timestamps=timestamps,
            risk_means=risk_means,
            risk_stds=risk_stds,
            risk_regimes=risk_regimes,
            abnormality_scores=abnormality_scores,
            instability_scores=instability_scores,
            true_states=true_states if true_states else None,
            total_time_ms=total_time_ms,
            avg_latency_per_step_ms=total_time_ms / n_steps,
            total_reasoning_cycles=n_steps * self.cerebras_config.reasoning_cycles_per_timestep
        )
        
        self._simulation_results[patient_id] = result
        return result
    
    def get_cerebras_metrics(self) -> Dict[str, Any]:
        """
        Get Cerebras Cloud API usage metrics.
        
        Returns:
            Dictionary with API call statistics and latency breakdown.
        """
        n_calls = max(self._total_cerebras_api_calls, 1)
        
        return {
            # Connection status
            "mode": self.cerebras_mode_status,
            "using_cloud": self.is_using_cerebras_cloud,
            "model": self._inference_client.model,
            
            # API call counts
            "total_api_calls": self._total_cerebras_api_calls,
            "total_observations": self._total_observations_processed,
            "api_call_ratio": (
                self._total_cerebras_api_calls / self._total_observations_processed
                if self._total_observations_processed > 0 else 0.0
            ),
            
            # Latency breakdown (per requirements)
            "avg_serialization_ms": self._total_serialization_ms / n_calls,
            "avg_api_roundtrip_ms": self._cerebras_api_latency_ms / n_calls,
            "avg_parsing_ms": self._total_parsing_ms / n_calls,
            "avg_total_latency_ms": (
                self._total_serialization_ms + 
                self._cerebras_api_latency_ms + 
                self._total_parsing_ms
            ) / n_calls,
            
            # Totals
            "total_api_latency_ms": self._cerebras_api_latency_ms,
            "total_serialization_ms": self._total_serialization_ms,
            "total_parsing_ms": self._total_parsing_ms,
        }
    
    def run_batch(
        self,
        trajectories: List[PatientTrajectory],
        parallel: bool = True,
        use_cerebras_final: bool = True
    ) -> List[SimulationResult]:
        """
        Run simulation on multiple patient trajectories.
        
        FAST: All processing is local. ONE API call per patient at the end.
        
        Args:
            trajectories: List of patient trajectories
            parallel: Whether to process patients in parallel
            use_cerebras_final: Make ONE API call per patient for final classification
            
        Returns:
            List of SimulationResult objects
        """
        results = []
        
        for i, trajectory in enumerate(trajectories):
            print(f"Processing patient {i+1}/{len(trajectories)}: {trajectory.patient_id}")
            result = self.run_patient_trajectory(
                trajectory, 
                use_cerebras_final=use_cerebras_final
            )
            results.append(result)
        
        return results
    
    def run_comparison(
        self,
        n_septic: int = 10,
        n_stable: int = 10,
        duration_hours: float = 24,
        use_cerebras_final: bool = True
    ) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        """
        Run comparison simulation between septic and stable patients.
        
        FAST EXECUTION:
        - All timestep processing runs locally (instant)
        - ONE API call per patient at the end for final classification
        - Total API calls = n_septic + n_stable
        
        Args:
            n_septic: Number of septic patients to simulate
            n_stable: Number of stable patients to simulate
            duration_hours: Duration of each trajectory
            use_cerebras_final: Use Cerebras for final classification (1 call per patient)
            
        Returns:
            Tuple of (septic_results, stable_results)
        """
        print(f"Generating {n_septic} septic and {n_stable} stable patients...")
        if use_cerebras_final and self._use_cerebras_llm:
            print(f"Cerebras API: {n_septic + n_stable} calls total (1 per patient)")
        
        # Generate septic patients
        septic_trajectories = [
            self._data_generator.generate_patient(
                patient_id=f"SEPTIC_{i:03d}",
                scenario="sepsis",
                duration_hours=duration_hours
            )
            for i in range(n_septic)
        ]
        
        # Generate stable patients
        stable_trajectories = [
            self._data_generator.generate_patient(
                patient_id=f"STABLE_{i:03d}",
                scenario="stable",
                duration_hours=duration_hours
            )
            for i in range(n_stable)
        ]
        
        print("Running simulations...")
        
        # Run simulations - all local, ONE API call per patient at end
        septic_results = self.run_batch(
            septic_trajectories, 
            use_cerebras_final=use_cerebras_final
        )
        stable_results = self.run_batch(
            stable_trajectories, 
            use_cerebras_final=use_cerebras_final
        )
        
        return septic_results, stable_results
    
    def run_speed_benchmark(
        self,
        n_patients: int = 100,
        n_timesteps: int = 100,
        warmup_iterations: int = 10
    ) -> BenchmarkResult:
        """
        Run speed benchmark to measure computational performance.
        
        IMPORTANT: This measures COMPUTATIONAL performance,
        not biological onset timing or clinical decision speed.
        
        Args:
            n_patients: Number of patients to simulate
            n_timesteps: Timesteps per patient
            warmup_iterations: Warmup iterations before timing
            
        Returns:
            BenchmarkResult with performance metrics
        """
        print(f"Running speed benchmark: {n_patients} patients × {n_timesteps} timesteps")
        
        # Generate synthetic data
        trajectories = [
            self._data_generator.generate_patient(
                patient_id=f"BENCH_{i:05d}",
                scenario="random",
                duration_hours=n_timesteps / 60  # Assuming 1-min timesteps
            )
            for i in range(n_patients)
        ]
        
        # Warmup
        print(f"Warmup ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            trajectory = trajectories[0]
            for i in range(min(10, len(trajectory))):
                vital_values = {k: v[i] for k, v in trajectory.vitals.items()}
                self.process_observation(trajectory.patient_id, float(i), vital_values)
        
        # Reset state
        self._feature_extractors.clear()
        self._risk_trackers.clear()
        
        # Timed run
        print("Running timed benchmark...")
        total_observations = 0
        total_reasoning_cycles = 0
        
        start_time = time.perf_counter()
        
        for trajectory in trajectories:
            n_steps = min(n_timesteps, len(trajectory))
            for i in range(n_steps):
                vital_values = {k: v[i] for k, v in trajectory.vitals.items()}
                self.process_observation(trajectory.patient_id, float(i), vital_values)
                total_observations += 1
                total_reasoning_cycles += self.cerebras_config.reasoning_cycles_per_timestep
        
        total_time = time.perf_counter() - start_time
        
        # Compute metrics
        observations_per_second = total_observations / total_time
        avg_latency_ms = (total_time * 1000) / total_observations
        reasoning_cycles_per_second = total_reasoning_cycles / total_time
        hypotheses_per_second = (
            total_observations * self.cerebras_config.parallel_hypotheses / total_time
        )
        
        result = BenchmarkResult(
            n_patients=n_patients,
            n_timesteps_per_patient=n_timesteps,
            total_observations=total_observations,
            total_time_seconds=total_time,
            observations_per_second=observations_per_second,
            avg_latency_ms=avg_latency_ms,
            reasoning_cycles_per_second=reasoning_cycles_per_second,
            hypotheses_per_second=hypotheses_per_second
        )
        
        print(f"\nBenchmark Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Observations/sec: {observations_per_second:.0f}")
        print(f"  Avg latency: {avg_latency_ms:.2f}ms")
        print(f"  Reasoning cycles/sec: {reasoning_cycles_per_second:.0f}")
        
        return result
    
    def load_external_dataset(
        self,
        file_path: str,
        patient_id_col: str = "patient_id",
        timestamp_col: str = "timestamp",
        vital_cols: Optional[Dict[str, str]] = None
    ) -> List[PatientTrajectory]:
        """
        Load external dataset (e.g., Kaggle EHR data).
        
        Args:
            file_path: Path to CSV file
            patient_id_col: Column name for patient ID
            timestamp_col: Column name for timestamp
            vital_cols: Mapping from our vital names to dataset column names
            
        Returns:
            List of PatientTrajectory objects
        """
        print(f"Loading external dataset: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Default vital column mapping
        if vital_cols is None:
            vital_cols = {
                "heart_rate": "heart_rate",
                "map": "map",
                "respiratory_rate": "respiratory_rate",
                "spo2": "spo2",
                "temperature": "temperature"
            }
        
        trajectories = []
        
        for patient_id, group in df.groupby(patient_id_col):
            group = group.sort_values(timestamp_col)
            
            # Extract vitals
            vitals = {}
            for our_name, their_name in vital_cols.items():
                if their_name in group.columns:
                    vitals[our_name] = group[their_name].values
            
            # Create trajectory
            trajectory = PatientTrajectory(
                patient_id=str(patient_id),
                timestamps=group[timestamp_col].values,
                vitals=vitals,
                latent_states=np.array([PatientState.STABLE] * len(group)),  # Unknown
                metadata={"source": file_path}
            )
            
            trajectories.append(trajectory)
        
        print(f"Loaded {len(trajectories)} patient trajectories")
        return trajectories
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all simulation results as a DataFrame."""
        if not self._simulation_results:
            return pd.DataFrame()
        
        dfs = [r.to_dataframe() for r in self._simulation_results.values()]
        return pd.concat(dfs, ignore_index=True)
    
    def reset(self):
        """Reset all internal state."""
        self._feature_extractors.clear()
        self._risk_trackers.clear()
        self._simulation_results.clear()
        self._total_observations_processed = 0
        self._total_processing_time_ms = 0.0
    
    @property
    def average_latency_ms(self) -> float:
        """Get average processing latency."""
        if self._total_observations_processed == 0:
            return 0.0
        return self._total_processing_time_ms / self._total_observations_processed


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Simulation Engine...")
    print("=" * 60)
    
    # Initialize engine
    engine = SepsisSimulationEngine()
    
    # Test 1: Single patient trajectory
    print("\n--- Test 1: Single Patient Trajectory ---")
    generator = SyntheticDataGenerator(random_seed=42)
    patient = generator.generate_patient(
        patient_id="TEST_001",
        scenario="sepsis",
        duration_hours=6,
        timestep_seconds=60
    )
    
    result = engine.run_patient_trajectory(patient)
    print(f"Patient: {result.patient_id}")
    print(f"Timesteps: {len(result.timestamps)}")
    print(f"Final risk: {result.risk_means[-1]:.3f} ± {result.risk_stds[-1]:.3f}")
    print(f"Final regime: {result.risk_regimes[-1]}")
    print(f"Total time: {result.total_time_ms:.1f}ms")
    print(f"Avg latency: {result.avg_latency_per_step_ms:.2f}ms/step")
    
    # Test 2: Comparison mode
    print("\n--- Test 2: Comparison Mode ---")
    engine.reset()
    septic_results, stable_results = engine.run_comparison(
        n_septic=3,
        n_stable=3,
        duration_hours=4
    )
    
    print(f"\nSeptic patients final risk:")
    for r in septic_results:
        print(f"  {r.patient_id}: {r.risk_means[-1]:.3f}")
    
    print(f"\nStable patients final risk:")
    for r in stable_results:
        print(f"  {r.patient_id}: {r.risk_means[-1]:.3f}")
    
    # Test 3: Speed benchmark
    print("\n--- Test 3: Speed Benchmark ---")
    engine.reset()
    benchmark = engine.run_speed_benchmark(
        n_patients=10,
        n_timesteps=50,
        warmup_iterations=5
    )
    
    print(f"\nBenchmark complete.")
    print(json.dumps(benchmark.to_dict(), indent=2))
    
    print("\n" + "=" * 60)
    print("Simulation engine test complete.")
