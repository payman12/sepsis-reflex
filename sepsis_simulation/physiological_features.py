"""
Physiological Feature Extraction Module
========================================

This module extracts deterministic features from vital sign time series.
These features are used for:
- Trend detection (slopes, acceleration)
- Volatility quantification (rolling variance, entropy)
- Cross-signal correlations (vital sign interactions)
- Regime shift indicators

Design Philosophy:
- No supervised learning: All features are domain-driven and interpretable
- Real-time capable: Features computed incrementally as data arrives
- Multi-scale: Features at multiple time horizons (5min, 30min, 6hr)

Why Deterministic Features:
1. Interpretable: Clinicians can understand what the system "sees"
2. Robust: No overfitting to training data distribution
3. Fast: Pure numerical computation, ideal for Cerebras acceleration
4. Complementary: Combine with probabilistic risk updates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy import stats
from scipy.signal import find_peaks

try:
    from .config import VitalSignConfig, VITAL_SIGNS, SimulationConfig
except ImportError:
    from config import VitalSignConfig, VITAL_SIGNS, SimulationConfig


@dataclass
class FeatureWindow:
    """
    Sliding window buffer for feature computation.
    
    Maintains efficient rolling statistics without recomputing
    from scratch at each timestep.
    """
    max_size: int
    values: deque = field(default_factory=deque)
    _sum: float = 0.0
    _sum_sq: float = 0.0
    
    def __post_init__(self):
        self.values = deque(maxlen=self.max_size)
    
    def add(self, value: float) -> None:
        """Add a new value to the window."""
        if len(self.values) == self.max_size:
            old = self.values[0]
            self._sum -= old
            self._sum_sq -= old ** 2
        self.values.append(value)
        self._sum += value
        self._sum_sq += value ** 2
    
    @property
    def mean(self) -> float:
        if len(self.values) == 0:
            return 0.0
        return self._sum / len(self.values)
    
    @property
    def variance(self) -> float:
        n = len(self.values)
        if n < 2:
            return 0.0
        return (self._sum_sq - self._sum ** 2 / n) / (n - 1)
    
    @property
    def std(self) -> float:
        return np.sqrt(max(0, self.variance))
    
    def as_array(self) -> np.ndarray:
        return np.array(self.values)
    
    def is_full(self) -> bool:
        return len(self.values) == self.max_size


@dataclass
class ExtractedFeatures:
    """
    Container for all extracted features at a single timestep.
    
    Organized by category for interpretability.
    """
    timestamp: float  # Unix timestamp
    
    # Raw values
    raw_values: Dict[str, float] = field(default_factory=dict)
    
    # Trend features (per vital sign)
    slopes: Dict[str, float] = field(default_factory=dict)
    accelerations: Dict[str, float] = field(default_factory=dict)
    
    # Volatility features
    volatilities: Dict[str, float] = field(default_factory=dict)
    
    # Deviation from normal
    z_scores: Dict[str, float] = field(default_factory=dict)
    
    # Cross-signal features
    correlations: Dict[str, float] = field(default_factory=dict)
    
    # Regime indicators
    regime_shift_score: float = 0.0
    trend_acceleration_score: float = 0.0
    
    # Composite scores
    instability_score: float = 0.0
    abnormality_score: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert all features to a flat numpy array."""
        features = []
        features.extend(self.raw_values.values())
        features.extend(self.slopes.values())
        features.extend(self.accelerations.values())
        features.extend(self.volatilities.values())
        features.extend(self.z_scores.values())
        features.extend(self.correlations.values())
        features.extend([
            self.regime_shift_score,
            self.trend_acceleration_score,
            self.instability_score,
            self.abnormality_score
        ])
        return np.array(features)
    
    @property
    def feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        names = []
        for prefix, d in [
            ("raw", self.raw_values),
            ("slope", self.slopes),
            ("accel", self.accelerations),
            ("vol", self.volatilities),
            ("zscore", self.z_scores),
            ("corr", self.correlations)
        ]:
            names.extend([f"{prefix}_{k}" for k in d.keys()])
        names.extend([
            "regime_shift_score",
            "trend_acceleration_score",
            "instability_score",
            "abnormality_score"
        ])
        return names


class PhysiologicalFeatureExtractor:
    """
    Real-time feature extractor for vital sign streams.
    
    Maintains internal state for efficient incremental computation.
    Features are computed at multiple time scales to capture:
    - Acute changes (5-15 minutes)
    - Gradual trends (30-60 minutes)
    - Sustained patterns (6+ hours)
    
    Design for Cerebras Optimization:
    --------------------------------
    While this module does basic feature extraction, the extracted features
    are then passed to the Cerebras-accelerated reasoning engine for:
    - Multi-cycle belief updates per feature vector
    - Parallel hypothesis evaluation
    - Fast forward simulation
    
    The separation allows feature extraction on CPU while reserving
    Cerebras for the computationally intensive reasoning cycles.
    """
    
    def __init__(
        self,
        vital_configs: Optional[Dict[str, VitalSignConfig]] = None,
        config: Optional[SimulationConfig] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            vital_configs: Configuration for each vital sign
            config: Simulation configuration
        """
        self.vital_configs = vital_configs or VITAL_SIGNS
        self.config = config or SimulationConfig()
        
        # Time windows in timesteps
        self.window_sizes = {
            "short": self.config.short_trend_window,    # ~15 min
            "medium": self.config.medium_trend_window,  # ~60 min
            "long": self.config.long_trend_window       # ~6 hours
        }
        
        # Initialize buffers for each vital sign
        self.buffers: Dict[str, Dict[str, FeatureWindow]] = {}
        self._initialize_buffers()
        
        # Previous features for acceleration computation
        self._prev_slopes: Dict[str, float] = {}
        
        # Correlation buffer (stores recent vectors for cross-correlation)
        self._correlation_buffer: Optional[np.ndarray] = None
        self._correlation_buffer_size = 30  # ~30 min at 1-min resolution
        
    def _initialize_buffers(self) -> None:
        """Initialize sliding window buffers for each vital sign."""
        for vital_name in self.vital_configs.keys():
            self.buffers[vital_name] = {
                window_name: FeatureWindow(max_size=size)
                for window_name, size in self.window_sizes.items()
            }
    
    def reset(self) -> None:
        """Reset all buffers (e.g., for new patient)."""
        self._initialize_buffers()
        self._prev_slopes = {}
        self._correlation_buffer = None
    
    def extract(
        self, 
        timestamp: float,
        vital_values: Dict[str, float]
    ) -> ExtractedFeatures:
        """
        Extract features from a new set of vital sign readings.
        
        Args:
            timestamp: Unix timestamp of the reading
            vital_values: Dict mapping vital sign names to values
            
        Returns:
            ExtractedFeatures object with all computed features
        """
        features = ExtractedFeatures(timestamp=timestamp)
        
        # Update buffers and compute per-vital features
        for vital_name, value in vital_values.items():
            if vital_name not in self.vital_configs:
                continue
                
            config = self.vital_configs[vital_name]
            
            # Update all window buffers
            for window_name, buffer in self.buffers[vital_name].items():
                buffer.add(value)
            
            # Store raw value
            features.raw_values[vital_name] = value
            
            # Compute z-score (deviation from normal)
            z_score = (value - config.normal_mean) / config.normal_std
            features.z_scores[vital_name] = z_score
            
            # Compute trend (slope) using medium window
            slope = self._compute_slope(vital_name, "medium")
            features.slopes[vital_name] = slope
            
            # Compute acceleration (change in slope)
            if vital_name in self._prev_slopes:
                acceleration = slope - self._prev_slopes[vital_name]
            else:
                acceleration = 0.0
            features.accelerations[vital_name] = acceleration
            self._prev_slopes[vital_name] = slope
            
            # Compute volatility (rolling std normalized by mean)
            volatility = self._compute_volatility(vital_name, "short")
            features.volatilities[vital_name] = volatility
        
        # Compute cross-signal correlations
        features.correlations = self._compute_correlations(vital_values)
        
        # Compute composite scores
        features.regime_shift_score = self._compute_regime_shift_score(features)
        features.trend_acceleration_score = self._compute_acceleration_score(features)
        features.instability_score = self._compute_instability_score(features)
        features.abnormality_score = self._compute_abnormality_score(features)
        
        return features
    
    def _compute_slope(
        self, 
        vital_name: str, 
        window: str = "medium"
    ) -> float:
        """
        Compute trend slope using linear regression.
        
        Returns slope in units/timestep, normalized by typical range.
        """
        buffer = self.buffers[vital_name][window]
        if len(buffer.values) < 5:
            return 0.0
        
        y = buffer.as_array()
        x = np.arange(len(y))
        
        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize by typical range
        config = self.vital_configs[vital_name]
        typical_range = config.critical_high - config.critical_low
        normalized_slope = slope / typical_range * 100  # Per 100 timesteps
        
        return normalized_slope
    
    def _compute_volatility(
        self, 
        vital_name: str,
        window: str = "short"
    ) -> float:
        """
        Compute volatility as coefficient of variation.
        
        High volatility indicates physiological instability.
        """
        buffer = self.buffers[vital_name][window]
        if len(buffer.values) < 3:
            return 0.0
        
        mean = buffer.mean
        std = buffer.std
        
        if abs(mean) < 1e-6:
            return 0.0
        
        # Coefficient of variation (CV)
        cv = std / abs(mean)
        
        return cv
    
    def _compute_correlations(
        self, 
        vital_values: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute key cross-signal correlations.
        
        These correlations are clinically meaningful:
        - HR-MAP: Should be inversely related (baroreceptor reflex)
        - RR-SpO2: Should be inversely related (compensation)
        - HR-RR: Coupling indicates intact autonomic function
        """
        correlations = {}
        
        # Update correlation buffer
        vital_order = ["heart_rate", "map", "respiratory_rate", "spo2", "temperature"]
        current_vector = np.array([
            vital_values.get(v, np.nan) for v in vital_order
        ])
        
        if self._correlation_buffer is None:
            self._correlation_buffer = current_vector.reshape(1, -1)
        else:
            self._correlation_buffer = np.vstack([
                self._correlation_buffer,
                current_vector.reshape(1, -1)
            ])
            if len(self._correlation_buffer) > self._correlation_buffer_size:
                self._correlation_buffer = self._correlation_buffer[-self._correlation_buffer_size:]
        
        # Compute correlations if we have enough data
        if len(self._correlation_buffer) >= 10:
            try:
                # HR-MAP correlation
                hr_idx, map_idx = 0, 1
                hr_data = self._correlation_buffer[:, hr_idx]
                map_data = self._correlation_buffer[:, map_idx]
                if np.std(hr_data) > 0 and np.std(map_data) > 0:
                    correlations["hr_map"] = np.corrcoef(hr_data, map_data)[0, 1]
                
                # RR-SpO2 correlation
                rr_idx, spo2_idx = 2, 3
                rr_data = self._correlation_buffer[:, rr_idx]
                spo2_data = self._correlation_buffer[:, spo2_idx]
                if np.std(rr_data) > 0 and np.std(spo2_data) > 0:
                    correlations["rr_spo2"] = np.corrcoef(rr_data, spo2_data)[0, 1]
                
                # HR-RR coupling
                if np.std(hr_data) > 0 and np.std(rr_data) > 0:
                    correlations["hr_rr"] = np.corrcoef(hr_data, rr_data)[0, 1]
                    
            except Exception:
                pass
        
        return correlations
    
    def _compute_regime_shift_score(self, features: ExtractedFeatures) -> float:
        """
        Compute a score indicating potential regime shift.
        
        Based on:
        - Large z-scores (values far from normal)
        - Changes in cross-correlation structure
        - Sustained trend reversals
        """
        score = 0.0
        
        # Z-score contribution
        z_scores = list(features.z_scores.values())
        if z_scores:
            max_z = max(abs(z) for z in z_scores)
            avg_z = np.mean([abs(z) for z in z_scores])
            score += min(max_z / 3.0, 1.0) * 0.4  # Cap at 1.0
            score += min(avg_z / 2.0, 1.0) * 0.3
        
        # Correlation deviation contribution
        if "hr_map" in features.correlations:
            # Normally HR and MAP are inversely correlated (~-0.4)
            # Loss of this relationship indicates dysregulation
            expected_corr = -0.4
            actual_corr = features.correlations["hr_map"]
            if not np.isnan(actual_corr):
                corr_deviation = abs(actual_corr - expected_corr)
                score += min(corr_deviation / 0.8, 1.0) * 0.3
        
        return min(score, 1.0)
    
    def _compute_acceleration_score(self, features: ExtractedFeatures) -> float:
        """
        Compute trend acceleration score.
        
        High acceleration indicates rapid physiological change,
        potentially preceding a clinical event.
        """
        accelerations = list(features.accelerations.values())
        if not accelerations:
            return 0.0
        
        # Use max absolute acceleration, normalized
        max_accel = max(abs(a) for a in accelerations)
        avg_accel = np.mean([abs(a) for a in accelerations])
        
        # Threshold based on typical variation
        threshold = self.config.acceleration_threshold
        
        score = (
            min(max_accel / threshold, 1.0) * 0.6 +
            min(avg_accel / (threshold * 0.5), 1.0) * 0.4
        )
        
        return min(score, 1.0)
    
    def _compute_instability_score(self, features: ExtractedFeatures) -> float:
        """
        Compute overall physiological instability score.
        
        Based on volatility across vital signs.
        """
        volatilities = list(features.volatilities.values())
        if not volatilities:
            return 0.0
        
        # Typical CV for stable vitals is ~0.02-0.05
        # Higher values indicate instability
        avg_vol = np.mean(volatilities)
        max_vol = max(volatilities)
        
        score = (
            min(avg_vol / 0.1, 1.0) * 0.5 +
            min(max_vol / 0.15, 1.0) * 0.5
        )
        
        return min(score, 1.0)
    
    def _compute_abnormality_score(self, features: ExtractedFeatures) -> float:
        """
        Compute overall abnormality score.
        
        Combines multiple indicators into a single 0-1 score.
        """
        components = [
            features.regime_shift_score * 0.35,
            features.trend_acceleration_score * 0.25,
            features.instability_score * 0.20,
            min(np.mean([abs(z) for z in features.z_scores.values()]) / 2.0, 1.0) * 0.20
            if features.z_scores else 0.0
        ]
        
        return min(sum(components), 1.0)


def compute_batch_features(
    timestamps: np.ndarray,
    vital_data: Dict[str, np.ndarray],
    vital_configs: Optional[Dict[str, VitalSignConfig]] = None
) -> List[ExtractedFeatures]:
    """
    Compute features for an entire time series in batch.
    
    Args:
        timestamps: Array of timestamps
        vital_data: Dict mapping vital names to value arrays
        vital_configs: Vital sign configurations
        
    Returns:
        List of ExtractedFeatures, one per timestep
    """
    extractor = PhysiologicalFeatureExtractor(vital_configs=vital_configs)
    features_list = []
    
    for i in range(len(timestamps)):
        vitals = {k: v[i] for k, v in vital_data.items()}
        features = extractor.extract(timestamps[i], vitals)
        features_list.append(features)
    
    return features_list


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing physiological feature extractor...")
    
    # Create test data
    np.random.seed(42)
    n_timesteps = 100
    
    # Simulate a patient who deteriorates
    vitals = {
        "heart_rate": 75 + np.cumsum(np.random.randn(n_timesteps) * 0.5),
        "map": 85 - np.cumsum(np.random.randn(n_timesteps) * 0.3),
        "respiratory_rate": 16 + np.cumsum(np.random.randn(n_timesteps) * 0.2),
        "spo2": 97 - np.abs(np.cumsum(np.random.randn(n_timesteps) * 0.1)),
        "temperature": 37 + np.cumsum(np.random.randn(n_timesteps) * 0.02)
    }
    
    timestamps = np.arange(n_timesteps, dtype=float)
    
    # Extract features
    extractor = PhysiologicalFeatureExtractor()
    
    print("\nExtracting features for each timestep...")
    all_features = []
    for i in range(n_timesteps):
        vital_values = {k: v[i] for k, v in vitals.items()}
        features = extractor.extract(timestamps[i], vital_values)
        all_features.append(features)
    
    # Show sample output
    print("\nSample features (timestep 50):")
    f = all_features[50]
    print(f"  Raw values: HR={f.raw_values.get('heart_rate', 0):.1f}, MAP={f.raw_values.get('map', 0):.1f}")
    print(f"  Z-scores: HR={f.z_scores.get('heart_rate', 0):.2f}, MAP={f.z_scores.get('map', 0):.2f}")
    print(f"  Slopes: HR={f.slopes.get('heart_rate', 0):.3f}, MAP={f.slopes.get('map', 0):.3f}")
    print(f"  Regime shift score: {f.regime_shift_score:.3f}")
    print(f"  Instability score: {f.instability_score:.3f}")
    print(f"  Abnormality score: {f.abnormality_score:.3f}")
    
    print(f"\nFeature vector size: {len(f.to_vector())}")
    print(f"Feature names: {f.feature_names[:10]}...")  # First 10
