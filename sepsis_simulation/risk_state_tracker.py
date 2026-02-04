"""
Bayesian Latent Risk State Tracker
===================================

This module maintains and updates a probabilistic belief state about
each patient's sepsis risk over time. Unlike binary classifiers, it:

1. Maintains uncertainty in risk estimates
2. Tracks risk as a continuous latent state
3. Updates beliefs incrementally with each observation
4. Detects regime shifts and trend acceleration

Design Philosophy:
- Probabilistic: Always track uncertainty, never output hard decisions
- Continuous: Risk is a spectrum, not a binary label
- Adaptive: Learn observation noise and drift from data
- Interpretable: Bayesian updates are mathematically principled

Why Not a Classifier:
- Classifiers output point estimates without uncertainty
- They're trained on labeled data (which we're avoiding)
- They don't naturally handle temporal continuity
- They can't express "I don't know"

Mathematical Framework:
- State: r_t ~ Normal(μ_t, σ_t²)  (latent risk)
- Observation model: y_t = f(r_t) + ε  (features given risk)
- Transition model: r_t = r_{t-1} + drift + noise
- Update: Bayesian filtering (similar to Kalman filter)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from scipy import stats

try:
    from .config import SimulationConfig
    from .physiological_features import ExtractedFeatures
except ImportError:
    from config import SimulationConfig
    from physiological_features import ExtractedFeatures


class RiskRegime(Enum):
    """
    Qualitative risk regimes for interpretability.
    These are derived from the continuous risk state.
    """
    BASELINE = "baseline"        # Normal physiological state
    ELEVATED = "elevated"        # Some concerning signals
    HIGH = "high"                # Multiple abnormalities
    CRITICAL = "critical"        # Severe derangement


@dataclass
class RiskBelief:
    """
    Represents the current belief about a patient's risk state.
    
    Uses a Gaussian approximation to the true posterior.
    """
    mean: float              # Point estimate of risk (0-1 scale)
    variance: float          # Uncertainty in the estimate
    
    # Derived quantities
    regime: RiskRegime = RiskRegime.BASELINE
    confidence: float = 0.5  # 1 - normalized variance
    
    # Historical tracking
    trend: float = 0.0       # Recent direction of risk change
    acceleration: float = 0.0  # Change in trend
    
    # Uncertainty bounds
    lower_bound: float = 0.0  # 95% CI lower
    upper_bound: float = 1.0  # 95% CI upper
    
    def __post_init__(self):
        """Compute derived quantities."""
        self._update_derived()
    
    def _update_derived(self):
        """Update derived quantities from mean and variance."""
        # Clamp mean to [0, 1]
        self.mean = np.clip(self.mean, 0.0, 1.0)
        self.variance = np.clip(self.variance, 1e-6, 0.25)
        
        # Confidence interval (approximate 95%)
        std = np.sqrt(self.variance)
        self.lower_bound = np.clip(self.mean - 1.96 * std, 0.0, 1.0)
        self.upper_bound = np.clip(self.mean + 1.96 * std, 0.0, 1.0)
        
        # Confidence (inverse of uncertainty)
        self.confidence = 1.0 - np.clip(std / 0.5, 0.0, 1.0)
        
        # Regime classification
        if self.mean < 0.25:
            self.regime = RiskRegime.BASELINE
        elif self.mean < 0.5:
            self.regime = RiskRegime.ELEVATED
        elif self.mean < 0.75:
            self.regime = RiskRegime.HIGH
        else:
            self.regime = RiskRegime.CRITICAL
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "variance": self.variance,
            "std": np.sqrt(self.variance),
            "regime": self.regime.value,
            "confidence": self.confidence,
            "trend": self.trend,
            "acceleration": self.acceleration,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound
        }


@dataclass
class HypothesisEvaluation:
    """
    Result of evaluating a specific hypothesis about patient state.
    
    Used for parallel hypothesis evaluation on Cerebras.
    """
    hypothesis_name: str
    prior_probability: float
    likelihood: float
    posterior_probability: float
    evidence_strength: float  # Log-likelihood ratio


class BayesianRiskTracker:
    """
    Maintains and updates a Bayesian belief state about sepsis risk.
    
    Core Algorithm:
    1. Receive new observation (features)
    2. Predict next state (with drift model)
    3. Compute observation likelihood
    4. Update belief (Bayesian filter step)
    5. Optionally run multiple reasoning cycles
    
    Multi-Cycle Reasoning:
    ---------------------
    The tracker supports running multiple update cycles per timestep,
    which is where Cerebras acceleration provides major benefits:
    
    - Cycle 1: Initial Bayesian update with raw observation
    - Cycle 2-N: Incorporate derived features (trends, correlations)
    - Each cycle refines the posterior with different evidence types
    
    On standard GPUs, each cycle adds ~10-50ms latency.
    On Cerebras, cycles are nearly free due to on-chip memory.
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        initial_risk_mean: float = 0.1,
        initial_risk_std: float = 0.15
    ):
        """
        Initialize the risk tracker.
        
        Args:
            config: Simulation configuration
            initial_risk_mean: Prior mean for risk
            initial_risk_std: Prior std for risk
        """
        self.config = config or SimulationConfig()
        
        # Current belief state
        self.belief = RiskBelief(
            mean=initial_risk_mean,
            variance=initial_risk_std ** 2
        )
        
        # Model parameters
        self.process_noise = 0.001    # State transition noise
        self.observation_noise = 0.1  # Measurement noise
        
        # Feature-to-risk mapping weights (domain-driven, not learned)
        # These define how each feature type influences risk
        self.feature_weights = {
            "z_scores": 0.25,           # Deviation from normal
            "slopes": 0.20,             # Trend direction
            "accelerations": 0.15,      # Trend acceleration
            "volatilities": 0.15,       # Physiological instability
            "regime_shift": 0.15,       # Regime change indicator
            "abnormality": 0.10         # Composite abnormality
        }
        
        # History for trend computation
        self._belief_history: List[RiskBelief] = []
        self._max_history = 100
        
        # Hypothesis tracking for parallel evaluation
        self._active_hypotheses: List[str] = [
            "stable",           # Patient is physiologically stable
            "compensating",     # Mild stress, body is compensating
            "deteriorating",    # Progressive decline
            "recovering",       # Improving from elevated risk
            "noise"             # Observations are just noise
        ]
    
    def reset(self, initial_mean: Optional[float] = None):
        """Reset the tracker for a new patient."""
        mean = initial_mean if initial_mean is not None else self.config.initial_risk_mean
        self.belief = RiskBelief(
            mean=mean,
            variance=self.config.initial_risk_std ** 2
        )
        self._belief_history = []
    
    def update(
        self,
        features: ExtractedFeatures,
        n_cycles: int = 1
    ) -> RiskBelief:
        """
        Update risk belief with new observation.
        
        This is the main entry point for single-observation updates.
        For multi-cycle reasoning, set n_cycles > 1.
        
        Args:
            features: Extracted features from current timestep
            n_cycles: Number of reasoning cycles (Cerebras optimization)
            
        Returns:
            Updated RiskBelief
        """
        # Store previous belief for trend computation
        prev_mean = self.belief.mean
        
        # Run multiple reasoning cycles
        for cycle in range(n_cycles):
            self._single_update_cycle(features, cycle)
        
        # Update trend and acceleration
        self._update_trends(prev_mean)
        
        # Store in history
        self._belief_history.append(RiskBelief(
            mean=self.belief.mean,
            variance=self.belief.variance,
            trend=self.belief.trend,
            acceleration=self.belief.acceleration
        ))
        if len(self._belief_history) > self._max_history:
            self._belief_history.pop(0)
        
        return self.belief
    
    def _single_update_cycle(
        self,
        features: ExtractedFeatures,
        cycle_idx: int
    ) -> None:
        """
        Execute a single Bayesian update cycle.
        
        Each cycle incorporates different types of evidence:
        - Cycle 0: Raw vital sign deviations
        - Cycle 1: Trend information
        - Cycle 2: Cross-correlations
        - Cycle 3+: Refine with composite scores
        
        Why Multiple Cycles on Cerebras:
        --------------------------------
        On GPUs, running 10 update cycles adds significant latency due to:
        - Kernel launch overhead (~10μs per cycle)
        - Memory transfers between GPU and CPU
        - Small batch inefficiency
        
        Cerebras' dataflow architecture eliminates these bottlenecks:
        - No kernel launches (static compute graph)
        - All data stays on-chip (850,000 cores, 40GB SRAM)
        - Uniform efficiency regardless of batch size
        
        This makes deep multi-cycle reasoning practical in real-time.
        """
        # === PREDICTION STEP ===
        # State evolves with small drift toward uncertainty
        predicted_mean = self.belief.mean
        predicted_variance = self.belief.variance + self.process_noise
        
        # === COMPUTE OBSERVATION LIKELIHOOD ===
        # Map features to expected risk level
        if cycle_idx == 0:
            # Cycle 0: Use z-scores (direct abnormality)
            observed_risk = self._compute_risk_from_zscores(features)
        elif cycle_idx == 1:
            # Cycle 1: Use trends
            observed_risk = self._compute_risk_from_trends(features)
        elif cycle_idx == 2:
            # Cycle 2: Use composite scores
            observed_risk = self._compute_risk_from_composites(features)
        else:
            # Additional cycles: Ensemble refinement
            observed_risk = self._compute_ensemble_risk(features)
        
        # Observation likelihood (Gaussian)
        # Higher observation noise → less trust in this observation
        cycle_obs_noise = self.observation_noise * (1 + 0.1 * cycle_idx)
        
        # === BAYESIAN UPDATE STEP ===
        # Kalman-style update (conjugate Gaussian)
        kalman_gain = predicted_variance / (predicted_variance + cycle_obs_noise)
        
        updated_mean = predicted_mean + kalman_gain * (observed_risk - predicted_mean)
        updated_variance = (1 - kalman_gain) * predicted_variance
        
        # Apply bounds
        updated_mean = np.clip(updated_mean, 0.0, 1.0)
        updated_variance = np.clip(
            updated_variance,
            self.config.min_uncertainty ** 2,
            self.config.max_uncertainty ** 2
        )
        
        # Update belief
        self.belief = RiskBelief(
            mean=updated_mean,
            variance=updated_variance,
            trend=self.belief.trend,
            acceleration=self.belief.acceleration
        )
    
    def _compute_risk_from_zscores(self, features: ExtractedFeatures) -> float:
        """Compute risk from z-score magnitudes."""
        if not features.z_scores:
            return self.belief.mean
        
        z_values = list(features.z_scores.values())
        
        # Use max absolute z-score with diminishing returns
        max_z = max(abs(z) for z in z_values)
        avg_z = np.mean([abs(z) for z in z_values])
        
        # Sigmoid-like mapping: z=0 → risk=0.1, z=3 → risk=0.8
        risk = 0.1 + 0.7 * (1 - np.exp(-0.5 * (0.6 * max_z + 0.4 * avg_z)))
        
        return np.clip(risk, 0.0, 1.0)
    
    def _compute_risk_from_trends(self, features: ExtractedFeatures) -> float:
        """Compute risk from trend indicators."""
        if not features.slopes:
            return self.belief.mean
        
        # Unfavorable trends increase risk
        # HR increasing, MAP decreasing, RR increasing, SpO2 decreasing → bad
        risk_contributions = []
        
        trend_directions = {
            "heart_rate": 1,        # Increasing is bad
            "map": -1,              # Decreasing is bad
            "respiratory_rate": 1,  # Increasing is bad
            "spo2": -1,             # Decreasing is bad
            "temperature": 1,       # Increasing (fever) is bad
            "lactate": 1,           # Increasing is bad
        }
        
        for vital, direction in trend_directions.items():
            if vital in features.slopes:
                slope = features.slopes[vital]
                # Positive contribution if slope is in "bad" direction
                contribution = max(0, direction * slope)
                risk_contributions.append(contribution)
        
        if not risk_contributions:
            return self.belief.mean
        
        # Combine contributions
        avg_contribution = np.mean(risk_contributions)
        max_contribution = max(risk_contributions)
        
        # Map to risk (0-1)
        risk = 0.15 + 0.7 * np.tanh(0.3 * avg_contribution + 0.2 * max_contribution)
        
        return np.clip(risk, 0.0, 1.0)
    
    def _compute_risk_from_composites(self, features: ExtractedFeatures) -> float:
        """Compute risk from composite feature scores."""
        # Weighted combination of composite scores
        risk = (
            0.35 * features.abnormality_score +
            0.25 * features.regime_shift_score +
            0.20 * features.instability_score +
            0.20 * features.trend_acceleration_score
        )
        
        # Anchor toward current belief (regularization)
        risk = 0.3 * self.belief.mean + 0.7 * risk
        
        return np.clip(risk, 0.0, 1.0)
    
    def _compute_ensemble_risk(self, features: ExtractedFeatures) -> float:
        """Combine multiple risk estimates (for additional cycles)."""
        estimates = [
            self._compute_risk_from_zscores(features),
            self._compute_risk_from_trends(features),
            self._compute_risk_from_composites(features)
        ]
        
        # Weighted average with emphasis on current belief
        ensemble = 0.2 * self.belief.mean + 0.8 * np.median(estimates)
        
        return np.clip(ensemble, 0.0, 1.0)
    
    def _update_trends(self, prev_mean: float) -> None:
        """Update trend and acceleration estimates."""
        # Current trend
        current_trend = self.belief.mean - prev_mean
        
        # Smooth with history
        if len(self._belief_history) > 0:
            recent_trends = [
                self._belief_history[i].mean - self._belief_history[i-1].mean
                for i in range(1, min(10, len(self._belief_history)))
            ]
            if recent_trends:
                smoothed_trend = 0.5 * current_trend + 0.5 * np.mean(recent_trends)
            else:
                smoothed_trend = current_trend
        else:
            smoothed_trend = current_trend
        
        # Acceleration (change in trend)
        prev_trend = self.belief.trend
        acceleration = smoothed_trend - prev_trend
        
        # Update belief
        self.belief.trend = smoothed_trend
        self.belief.acceleration = acceleration
        self.belief._update_derived()
    
    def evaluate_hypotheses(
        self,
        features: ExtractedFeatures
    ) -> List[HypothesisEvaluation]:
        """
        Evaluate multiple hypotheses in parallel.
        
        This is designed for Cerebras parallelization:
        Each hypothesis can be evaluated independently,
        making it ideal for massive parallelism.
        
        Returns list of hypothesis evaluations with posteriors.
        """
        evaluations = []
        
        # Prior probabilities based on current belief
        risk = self.belief.mean
        priors = {
            "stable": max(0.1, 1 - risk),
            "compensating": 0.2 if 0.2 < risk < 0.5 else 0.1,
            "deteriorating": min(0.5, risk),
            "recovering": 0.1 if self.belief.trend < -0.01 else 0.05,
            "noise": 0.1
        }
        
        # Normalize priors
        total = sum(priors.values())
        priors = {k: v/total for k, v in priors.items()}
        
        # Compute likelihoods
        likelihoods = {}
        
        # Stable: Low z-scores, low volatility
        z_magnitude = np.mean([abs(z) for z in features.z_scores.values()]) if features.z_scores else 0
        likelihoods["stable"] = np.exp(-z_magnitude) * np.exp(-features.instability_score * 2)
        
        # Compensating: Elevated but not extreme, some instability
        likelihoods["compensating"] = (
            np.exp(-abs(z_magnitude - 1.5)) *
            np.exp(-abs(features.abnormality_score - 0.4) * 2)
        )
        
        # Deteriorating: High abnormality, positive risk trend
        deteriorating_signal = features.abnormality_score + max(0, self.belief.trend) * 5
        likelihoods["deteriorating"] = 1 - np.exp(-deteriorating_signal)
        
        # Recovering: Negative risk trend, decreasing abnormality
        recovering_signal = max(0, -self.belief.trend) * 5 + max(0, -features.trend_acceleration_score)
        likelihoods["recovering"] = 1 - np.exp(-recovering_signal * 2)
        
        # Noise: Observations don't match any clear pattern
        likelihoods["noise"] = 0.1  # Small constant
        
        # Compute posteriors (Bayes' rule)
        unnormalized = {h: priors[h] * likelihoods[h] for h in self._active_hypotheses}
        total = sum(unnormalized.values())
        
        for hypothesis in self._active_hypotheses:
            posterior = unnormalized[hypothesis] / total if total > 0 else priors[hypothesis]
            
            # Evidence strength (log-likelihood ratio vs uniform)
            evidence = np.log(likelihoods[hypothesis] + 1e-10) - np.log(0.2)
            
            evaluations.append(HypothesisEvaluation(
                hypothesis_name=hypothesis,
                prior_probability=priors[hypothesis],
                likelihood=likelihoods[hypothesis],
                posterior_probability=posterior,
                evidence_strength=evidence
            ))
        
        return evaluations
    
    def get_state_summary(self) -> Dict:
        """Get a summary of the current risk state."""
        return {
            "risk_belief": self.belief.to_dict(),
            "history_length": len(self._belief_history),
            "regime": self.belief.regime.value,
            "trend_direction": "increasing" if self.belief.trend > 0.005 else 
                              "decreasing" if self.belief.trend < -0.005 else "stable"
        }


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Bayesian risk tracker...")
    
    from physiological_features import PhysiologicalFeatureExtractor
    
    # Create test data (simulating deterioration)
    np.random.seed(42)
    n_timesteps = 50
    
    # Simulate vital signs that gradually deteriorate
    timestamps = np.arange(n_timesteps, dtype=float)
    vitals = {
        "heart_rate": 75 + np.linspace(0, 30, n_timesteps) + np.random.randn(n_timesteps) * 2,
        "map": 85 - np.linspace(0, 20, n_timesteps) + np.random.randn(n_timesteps) * 3,
        "respiratory_rate": 16 + np.linspace(0, 10, n_timesteps) + np.random.randn(n_timesteps),
        "spo2": 97 - np.linspace(0, 5, n_timesteps) + np.random.randn(n_timesteps) * 0.5,
        "temperature": 37 + np.linspace(0, 1.5, n_timesteps) + np.random.randn(n_timesteps) * 0.1
    }
    
    # Initialize components
    feature_extractor = PhysiologicalFeatureExtractor()
    risk_tracker = BayesianRiskTracker()
    
    print("\nSimulating patient trajectory with deterioration...")
    print("-" * 60)
    
    for i in range(n_timesteps):
        # Extract features
        vital_values = {k: v[i] for k, v in vitals.items()}
        features = feature_extractor.extract(timestamps[i], vital_values)
        
        # Update risk (with 3 reasoning cycles)
        belief = risk_tracker.update(features, n_cycles=3)
        
        # Print every 10 timesteps
        if i % 10 == 0:
            print(f"t={i:3d}: Risk={belief.mean:.3f} ± {np.sqrt(belief.variance):.3f} "
                  f"[{belief.regime.value:12s}] trend={belief.trend:+.4f}")
    
    print("-" * 60)
    print(f"\nFinal state: {risk_tracker.get_state_summary()}")
    
    # Test hypothesis evaluation
    print("\n\nHypothesis evaluation at final timestep:")
    final_features = feature_extractor.extract(
        timestamps[-1],
        {k: v[-1] for k, v in vitals.items()}
    )
    hypotheses = risk_tracker.evaluate_hypotheses(final_features)
    for h in hypotheses:
        print(f"  {h.hypothesis_name:15s}: prior={h.prior_probability:.3f} "
              f"→ posterior={h.posterior_probability:.3f}")
