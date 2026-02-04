"""
Cerebras Inference Client - Stateless LLM Inference Only
=========================================================

ARCHITECTURE OVERVIEW:
----------------------
This module provides a CLEAN ABSTRACTION for Cerebras Cloud LLM inference.
Cerebras is used STRICTLY for high-throughput AI inference.
ALL simulation logic runs locally.

WHAT RUNS ON CEREBRAS (Remote API):
-----------------------------------
- Sepsis risk classification from patient data windows
- Return: probability, confidence, reasoning
- Stateless requests (each call independent)
- Optimized for batch throughput and low latency

WHAT RUNS LOCALLY (NOT sent to Cerebras):
-----------------------------------------
- Time-series generation of patient vital signals
- Patient trajectory simulation
- Data preprocessing, normalization, windowing
- Feature extraction
- Visualization and dashboards
- Aggregation of model outputs over time

DATA FLOW:
----------
1. Local simulator generates continuous patient vital signal trajectories
2. Sliding/fixed windows extracted from time series
3. Windows serialized into structured JSON payload
4. Payloads sent to Cerebras inference endpoint
5. Model outputs returned and stored locally
6. Outputs plotted against time on same x-axis as original signal

PERFORMANCE LOGGING:
--------------------
- Request serialization time
- API round-trip time
- Response parsing time

USAGE:
------
    # Create inference client (swappable with GPU model later)
    client = InferenceClient(api_key="your-key")
    
    # Single patient inference
    result = client.infer(patient_window)
    
    # Batch inference
    results = client.infer_batch(patient_windows)
    
    # Check latency metrics
    metrics = client.get_latency_metrics()
"""

import os
import numpy as np
import time
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol
from abc import ABC, abstractmethod
import threading

# =============================================================================
# CEREBRAS SDK IMPORT
# =============================================================================

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    print("[INFO] Cerebras SDK not installed. Install with: pip install cerebras-cloud-sdk")


# =============================================================================
# DATA CLASSES - INFERENCE RESULTS
# =============================================================================

@dataclass
class LatencyMetrics:
    """
    Detailed latency breakdown for performance evaluation.
    
    Per requirements:
    - Measure input → output latency only
    - Log serialization, round-trip, and parsing times
    - Changes below 0.1 ms are not clinically meaningful
    """
    serialization_ms: float = 0.0  # Time to serialize payload
    api_roundtrip_ms: float = 0.0  # Cerebras API call time
    parsing_ms: float = 0.0        # Time to parse response
    total_ms: float = 0.0          # Total end-to-end latency
    
    def __post_init__(self):
        self.total_ms = self.serialization_ms + self.api_roundtrip_ms + self.parsing_ms


@dataclass
class InferenceResult:
    """
    Result from Cerebras inference call.
    
    Contains only what Cerebras returns:
    - Sepsis classification probability
    - Risk score
    - Confidence/uncertainty metric
    """
    risk_probability: float      # 0-1 sepsis probability
    risk_score: float            # 0-1 overall risk score
    confidence: float            # 0-1 confidence in prediction
    uncertainty: float           # Standard deviation estimate
    key_factors: List[str]       # Contributing factors
    reasoning: str               # LLM's explanation
    latency: LatencyMetrics      # Timing breakdown
    model_used: str              # Model identifier
    from_api: bool               # True if from Cerebras, False if local fallback


@dataclass
class BatchInferenceResult:
    """Results from batch inference call."""
    results: List[InferenceResult]
    total_latency_ms: float
    throughput_per_second: float
    batch_size: int


# =============================================================================
# CLINICAL PROMPT TEMPLATE
# =============================================================================

CLINICAL_PROMPT_TEMPLATE = """You are a medical AI specialized in early sepsis detection.
Analyze the patient data and provide a sepsis risk assessment.

PATIENT DATA WINDOW:
{patient_data}

METRICS AVAILABLE:
{metrics_list}

Sepsis indicators to consider:
- Heart Rate (HR): Tachycardia (>90 bpm) is early sign
- Mean Arterial Pressure (MAP): Hypotension (<65 mmHg) indicates shock
- Respiratory Rate (RR): Tachypnea (>20/min) indicates compensation
- SpO2: Hypoxemia (<92%) indicates respiratory compromise
- Temperature: Fever (>38°C) or hypothermia (<36°C)
- Lactate: Elevated (>2 mmol/L) indicates tissue hypoperfusion
- White Blood Cell (WBC): Leukocytosis (>12K) or leukopenia (<4K)

TASK: Analyze the data and respond ONLY with valid JSON:
{{
    "risk_probability": <0.0-1.0>,
    "risk_score": <0.0-1.0>,
    "confidence": <0.0-1.0>,
    "key_factors": ["factor1", "factor2"],
    "reasoning": "<brief explanation>"
}}
"""


# =============================================================================
# INFERENCE CLIENT PROTOCOL (for swappability)
# =============================================================================

class InferenceClientProtocol(Protocol):
    """Protocol for inference clients (enables swapping Cerebras with GPU model)."""
    
    def infer(self, patient_data: Dict[str, Any]) -> InferenceResult:
        """Run inference on single patient window."""
        ...
    
    def infer_batch(self, patient_data_list: List[Dict[str, Any]]) -> BatchInferenceResult:
        """Run inference on batch of patient windows."""
        ...
    
    def is_available(self) -> bool:
        """Check if inference backend is available."""
        ...
    
    def get_latency_metrics(self) -> Dict[str, float]:
        """Get aggregate latency metrics."""
        ...


# =============================================================================
# INFERENCE CLIENT - CEREBRAS IMPLEMENTATION
# =============================================================================

class InferenceClient:
    """
    Cerebras Cloud Inference Client.
    
    This is the ONLY interface to Cerebras Cloud.
    All it does is:
    1. Accept patient data windows
    2. Send to Cerebras LLM for inference
    3. Return classification results
    
    NO simulation logic. NO data generation. NO visualization.
    
    Can be swapped with a GPU-based model by implementing the same interface.
    """
    
    # Available models
    MODELS = {
        "fast": "llama-4-scout-17b-16e-instruct",
        "balanced": "llama3.3-70b",
        "accurate": "llama3.3-70b",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_tier: str = "balanced",
        max_tokens: int = 500,
        temperature: float = 0.1
    ):
        """
        Initialize Cerebras inference client.
        
        Args:
            api_key: Cerebras API key (or set CEREBRAS_API_KEY env var)
            model_tier: "fast", "balanced", or "accurate"
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.api_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        self.model = self.MODELS.get(model_tier, self.MODELS["balanced"])
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Client state
        self._client = None
        self._connected = False
        self._use_local_fallback = True
        
        # Initialize connection
        if CEREBRAS_AVAILABLE and self.api_key:
            try:
                self._client = Cerebras(api_key=self.api_key)
                self._connected = True
                self._use_local_fallback = False
                print(f"[INFERENCE] ✅ Connected to Cerebras Cloud (model: {self.model})")
            except Exception as e:
                print(f"[INFERENCE] ❌ Cerebras connection failed: {e}")
                self._use_local_fallback = True
        else:
            if not CEREBRAS_AVAILABLE:
                print("[INFERENCE] Cerebras SDK not installed - using local fallback")
            elif not self.api_key:
                print("[INFERENCE] No API key - using local fallback")
        
        # Metrics tracking
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_serialization_ms = 0.0
        self._total_api_roundtrip_ms = 0.0
        self._total_parsing_ms = 0.0
    
    def infer(self, patient_data: Dict[str, Any]) -> InferenceResult:
        """
        Run inference on a single patient data window.
        
        This is the CORE method. It:
        1. Serializes patient data into prompt (LOCAL)
        2. Calls Cerebras API (REMOTE)
        3. Parses response (LOCAL)
        
        Args:
            patient_data: Dictionary containing:
                - vitals: Dict[str, float] - current vital signs
                - window: Optional[List[Dict]] - time window of readings
                - metrics_available: List[str] - which metrics are present
        
        Returns:
            InferenceResult with risk assessment
        """
        # ================================================
        # STEP 1: SERIALIZE (LOCAL)
        # ================================================
        serialize_start = time.perf_counter()
        prompt = self._build_prompt(patient_data)
        serialization_ms = (time.perf_counter() - serialize_start) * 1000
        
        # ================================================
        # STEP 2: CALL CEREBRAS API (REMOTE)
        # ================================================
        api_start = time.perf_counter()
        
        if self._use_local_fallback:
            response_text = self._local_inference(patient_data)
            from_api = False
        else:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                response_text = response.choices[0].message.content
                from_api = True
            except Exception as e:
                print(f"[INFERENCE] API error, using local fallback: {e}")
                response_text = self._local_inference(patient_data)
                from_api = False
        
        api_roundtrip_ms = (time.perf_counter() - api_start) * 1000
        
        # ================================================
        # STEP 3: PARSE RESPONSE (LOCAL)
        # ================================================
        parse_start = time.perf_counter()
        result = self._parse_response(response_text, from_api)
        parsing_ms = (time.perf_counter() - parse_start) * 1000
        
        # Build latency metrics
        result.latency = LatencyMetrics(
            serialization_ms=serialization_ms,
            api_roundtrip_ms=api_roundtrip_ms,
            parsing_ms=parsing_ms
        )
        
        # Update aggregate metrics
        with self._lock:
            self._total_requests += 1
            self._total_serialization_ms += serialization_ms
            self._total_api_roundtrip_ms += api_roundtrip_ms
            self._total_parsing_ms += parsing_ms
        
        return result
    
    def infer_batch(
        self,
        patient_data_list: List[Dict[str, Any]],
        parallel: bool = True
    ) -> BatchInferenceResult:
        """
        Run inference on a batch of patient windows.
        
        Optimized for high throughput on Cerebras.
        
        Args:
            patient_data_list: List of patient data dictionaries
            parallel: Whether to process in parallel (default True)
        
        Returns:
            BatchInferenceResult with all results
        """
        start_time = time.perf_counter()
        
        if parallel and len(patient_data_list) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = [None] * len(patient_data_list)
            with ThreadPoolExecutor(max_workers=min(10, len(patient_data_list))) as executor:
                futures = {
                    executor.submit(self.infer, data): i 
                    for i, data in enumerate(patient_data_list)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = self._error_result(str(e))
        else:
            results = [self.infer(data) for data in patient_data_list]
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        return BatchInferenceResult(
            results=results,
            total_latency_ms=total_ms,
            throughput_per_second=len(patient_data_list) / (total_ms / 1000) if total_ms > 0 else 0,
            batch_size=len(patient_data_list)
        )
    
    def _build_prompt(self, patient_data: Dict[str, Any]) -> str:
        """Build prompt from patient data."""
        vitals = patient_data.get("vitals", {})
        window = patient_data.get("window", [])
        metrics_available = patient_data.get("metrics_available", list(vitals.keys()))
        
        # Format current vitals
        vitals_str = ""
        for name, value in vitals.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                vitals_str += f"  {name}: {value:.2f}\n"
        
        # Format window if available
        if window:
            vitals_str += "\nRecent readings:\n"
            for i, reading in enumerate(window[-6:]):
                vitals_str += f"  T-{len(window)-i}: "
                vitals_str += ", ".join([f"{k}={v:.1f}" for k, v in reading.items() 
                                         if v is not None and not (isinstance(v, float) and np.isnan(v))])
                vitals_str += "\n"
        
        return CLINICAL_PROMPT_TEMPLATE.format(
            patient_data=vitals_str,
            metrics_list=", ".join(metrics_available)
        )
    
    def _parse_response(self, response_text: str, from_api: bool) -> InferenceResult:
        """Parse LLM response into InferenceResult."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return InferenceResult(
                    risk_probability=float(parsed.get("risk_probability", 0.5)),
                    risk_score=float(parsed.get("risk_score", 0.5)),
                    confidence=float(parsed.get("confidence", 0.5)),
                    uncertainty=1.0 - float(parsed.get("confidence", 0.5)),
                    key_factors=parsed.get("key_factors", []),
                    reasoning=parsed.get("reasoning", ""),
                    latency=LatencyMetrics(),
                    model_used=self.model if from_api else "local_fallback",
                    from_api=from_api
                )
        except Exception:
            pass
        
        # Fallback parsing
        risk = 0.5
        if "high risk" in response_text.lower():
            risk = 0.8
        elif "low risk" in response_text.lower():
            risk = 0.2
        
        return InferenceResult(
            risk_probability=risk,
            risk_score=risk,
            confidence=0.5,
            uncertainty=0.5,
            key_factors=[],
            reasoning=response_text[:200] if response_text else "",
            latency=LatencyMetrics(),
            model_used=self.model if from_api else "local_fallback",
            from_api=from_api
        )
    
    def _local_inference(self, patient_data: Dict[str, Any]) -> str:
        """
        Local fallback inference when Cerebras is unavailable.
        
        Uses rule-based risk calculation.
        """
        vitals = patient_data.get("vitals", {})
        
        risk = 0.1
        factors = []
        
        # Heart rate
        hr = vitals.get("heart_rate", vitals.get("hr", 75))
        if hr > 100:
            risk += 0.15
            factors.append("Tachycardia")
        elif hr > 90:
            risk += 0.05
        
        # Respiratory rate
        rr = vitals.get("respiratory_rate", vitals.get("rr", 16))
        if rr > 22:
            risk += 0.15
            factors.append("Tachypnea")
        elif rr > 20:
            risk += 0.05
        
        # Temperature
        temp = vitals.get("temperature", vitals.get("temp", 37))
        if temp > 38.3 or temp < 36:
            risk += 0.2
            factors.append("Abnormal temperature")
        
        # MAP
        map_val = vitals.get("map", vitals.get("mean_arterial_pressure", 85))
        if map_val < 65:
            risk += 0.25
            factors.append("Hypotension")
        elif map_val < 70:
            risk += 0.1
        
        # SpO2
        spo2 = vitals.get("spo2", vitals.get("oxygen_saturation", 97))
        if spo2 < 92:
            risk += 0.2
            factors.append("Hypoxemia")
        elif spo2 < 95:
            risk += 0.05
        
        # Lactate
        lactate = vitals.get("lactate", 1.0)
        if lactate > 4:
            risk += 0.3
            factors.append("Severely elevated lactate")
        elif lactate > 2:
            risk += 0.15
            factors.append("Elevated lactate")
        
        # WBC
        wbc = vitals.get("wbc", vitals.get("white_blood_cell", 8))
        if wbc > 12 or wbc < 4:
            risk += 0.1
            factors.append("Abnormal WBC")
        
        risk = min(risk, 0.95)
        confidence = 0.7 if len(factors) >= 2 else 0.5
        
        return json.dumps({
            "risk_probability": risk,
            "risk_score": risk,
            "confidence": confidence,
            "key_factors": factors,
            "reasoning": f"Local analysis: {len(factors)} risk factors identified"
        })
    
    def _error_result(self, error_msg: str) -> InferenceResult:
        """Return error result."""
        return InferenceResult(
            risk_probability=0.5,
            risk_score=0.5,
            confidence=0.0,
            uncertainty=1.0,
            key_factors=["Error"],
            reasoning=f"Inference failed: {error_msg}",
            latency=LatencyMetrics(),
            model_used="error",
            from_api=False
        )
    
    def is_available(self) -> bool:
        """Check if Cerebras API is available."""
        return self._connected and not self._use_local_fallback
    
    def get_latency_metrics(self) -> Dict[str, float]:
        """Get aggregate latency metrics."""
        with self._lock:
            n = max(self._total_requests, 1)
            return {
                "total_requests": self._total_requests,
                "avg_serialization_ms": self._total_serialization_ms / n,
                "avg_api_roundtrip_ms": self._total_api_roundtrip_ms / n,
                "avg_parsing_ms": self._total_parsing_ms / n,
                "avg_total_ms": (self._total_serialization_ms + self._total_api_roundtrip_ms + self._total_parsing_ms) / n,
                "connected": self._connected,
                "using_local_fallback": self._use_local_fallback
            }
    
    def get_status(self) -> str:
        """Get human-readable status."""
        if self._connected and not self._use_local_fallback:
            return f"✅ Cerebras Cloud ({self.model})"
        elif not CEREBRAS_AVAILABLE:
            return "⚠️ Local Mode (SDK not installed)"
        elif not self.api_key:
            return "⚠️ Local Mode (No API key)"
        else:
            return "⚠️ Local Mode (Connection failed)"


# =============================================================================
# LOCAL SIMULATOR - ALL SIMULATION LOGIC (NOT sent to Cerebras)
# =============================================================================

class LocalSimulator:
    """
    Local simulation engine for patient vital signal trajectories.
    
    THIS CLASS HANDLES ALL LOCAL COMPUTATION:
    - Time-series generation
    - Data preprocessing, normalization, windowing
    - Feature extraction
    - Metric selection and RX tagging
    - Aggregation of model outputs
    
    NONE of this is sent to Cerebras.
    """
    
    # Core ICU-relevant metrics (7 as specified)
    ICU_METRICS = {
        "heart_rate": {"unit": "bpm", "normal_range": (60, 100), "rx_actionable": True},
        "map": {"unit": "mmHg", "normal_range": (70, 100), "rx_actionable": True},
        "respiratory_rate": {"unit": "/min", "normal_range": (12, 20), "rx_actionable": True},
        "spo2": {"unit": "%", "normal_range": (95, 100), "rx_actionable": True},
        "temperature": {"unit": "°C", "normal_range": (36.5, 37.5), "rx_actionable": True},
        "lactate": {"unit": "mmol/L", "normal_range": (0.5, 2.0), "rx_actionable": True},
        "wbc": {"unit": "K/μL", "normal_range": (4, 11), "rx_actionable": True},
    }
    
    def __init__(
        self,
        window_size: int = 10,
        step_size: int = 1,
        time_resolution_seconds: float = 60.0
    ):
        """
        Initialize local simulator.
        
        Args:
            window_size: Number of readings per window
            step_size: Step size for sliding window
            time_resolution_seconds: Time between readings (continuous time axis)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.time_resolution = time_resolution_seconds
        
        # Patient data storage (local only)
        self._patient_trajectories: Dict[str, Dict] = {}
        self._inference_results: Dict[str, List[Tuple[float, InferenceResult]]] = {}
    
    def add_reading(
        self,
        patient_id: str,
        timestamp: float,
        vitals: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Add a vital sign reading to patient trajectory.
        
        ALL LOCAL - not sent to Cerebras.
        
        Args:
            patient_id: Patient identifier
            timestamp: Real time (seconds from start)
            vitals: Dictionary of vital sign values
        
        Returns:
            Preprocessed patient data window (ready for inference)
        """
        if patient_id not in self._patient_trajectories:
            self._patient_trajectories[patient_id] = {
                "timestamps": [],
                "readings": [],
                "normalized": []
            }
        
        traj = self._patient_trajectories[patient_id]
        traj["timestamps"].append(timestamp)
        traj["readings"].append(vitals)
        
        # Normalize (LOCAL)
        normalized = self._normalize_vitals(vitals)
        traj["normalized"].append(normalized)
        
        # Extract window
        return self._extract_window(patient_id)
    
    def _normalize_vitals(self, vitals: Dict[str, float]) -> Dict[str, float]:
        """Normalize vital signs to 0-1 range based on clinical ranges."""
        normalized = {}
        for metric, value in vitals.items():
            if metric in self.ICU_METRICS:
                low, high = self.ICU_METRICS[metric]["normal_range"]
                # Normalize to 0-1 where 0.5 is normal midpoint
                midpoint = (low + high) / 2
                range_size = high - low
                normalized[metric] = 0.5 + (value - midpoint) / (range_size * 2)
                normalized[metric] = np.clip(normalized[metric], 0, 1)
            else:
                normalized[metric] = value
        return normalized
    
    def _extract_window(self, patient_id: str) -> Dict[str, Any]:
        """Extract sliding window for inference."""
        traj = self._patient_trajectories[patient_id]
        
        if len(traj["readings"]) < self.window_size:
            # Pad with first reading
            pad_count = self.window_size - len(traj["readings"])
            window = [traj["readings"][0]] * pad_count + traj["readings"]
        else:
            window = traj["readings"][-self.window_size:]
        
        # Latest vitals
        current_vitals = traj["readings"][-1]
        
        return {
            "patient_id": patient_id,
            "vitals": current_vitals,
            "window": window,
            "metrics_available": [m for m in current_vitals.keys() if m in self.ICU_METRICS],
            "timestamp": traj["timestamps"][-1]
        }
    
    def store_inference_result(
        self,
        patient_id: str,
        timestamp: float,
        result: InferenceResult
    ):
        """Store inference result for later visualization."""
        if patient_id not in self._inference_results:
            self._inference_results[patient_id] = []
        self._inference_results[patient_id].append((timestamp, result))
    
    def get_risk_trajectory(self, patient_id: str) -> Tuple[List[float], List[float]]:
        """
        Get risk trajectory for visualization.
        
        Returns:
            Tuple of (timestamps, risk_scores)
        """
        if patient_id not in self._inference_results:
            return [], []
        
        results = self._inference_results[patient_id]
        timestamps = [r[0] for r in results]
        risks = [r[1].risk_score for r in results]
        return timestamps, risks
    
    def get_vital_trajectory(
        self,
        patient_id: str,
        metric: str
    ) -> Tuple[List[float], List[float]]:
        """
        Get vital sign trajectory for visualization.
        
        Args:
            patient_id: Patient identifier
            metric: Metric name (e.g., "heart_rate")
        
        Returns:
            Tuple of (timestamps, values)
        """
        if patient_id not in self._patient_trajectories:
            return [], []
        
        traj = self._patient_trajectories[patient_id]
        timestamps = traj["timestamps"]
        values = [r.get(metric, np.nan) for r in traj["readings"]]
        return timestamps, values
    
    def get_metrics_info(self) -> Dict[str, Dict]:
        """Get information about available metrics including RX tags."""
        return self.ICU_METRICS.copy()
    
    def clear_patient(self, patient_id: str):
        """Clear patient data."""
        self._patient_trajectories.pop(patient_id, None)
        self._inference_results.pop(patient_id, None)


# =============================================================================
# SEPSIS RISK ANALYZER - COMBINES INFERENCE + LOCAL SIMULATION
# =============================================================================

class SepsisRiskAnalyzer:
    """
    High-level interface for sepsis risk analysis.
    
    Combines:
    - InferenceClient (Cerebras API calls)
    - LocalSimulator (all local computation)
    
    This is the main entry point for the dashboard.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_tier: str = "balanced",
        window_size: int = 10
    ):
        """
        Initialize analyzer.
        
        Args:
            api_key: Cerebras API key
            model_tier: Model tier ("fast", "balanced", "accurate")
            window_size: Window size for time series
        """
        self.inference_client = InferenceClient(
            api_key=api_key,
            model_tier=model_tier
        )
        self.local_simulator = LocalSimulator(window_size=window_size)
        
        # Metrics
        self._total_analyses = 0
        self._api_calls = 0
    
    def analyze_reading(
        self,
        patient_id: str,
        timestamp: float,
        vitals: Dict[str, float],
        call_api: bool = True
    ) -> InferenceResult:
        """
        Analyze a single patient reading.
        
        1. Add reading to local trajectory (LOCAL)
        2. Extract window (LOCAL)
        3. Call Cerebras for inference (REMOTE, if enabled)
        4. Store result (LOCAL)
        
        Args:
            patient_id: Patient identifier
            timestamp: Real time (seconds)
            vitals: Vital sign values
            call_api: Whether to call Cerebras API
        
        Returns:
            InferenceResult with risk assessment
        """
        # Step 1-2: Local processing
        patient_data = self.local_simulator.add_reading(patient_id, timestamp, vitals)
        
        # Step 3: Inference (remote or local)
        if call_api:
            result = self.inference_client.infer(patient_data)
            self._api_calls += 1
        else:
            # Use local fallback directly
            response = self.inference_client._local_inference(patient_data)
            result = self.inference_client._parse_response(response, from_api=False)
            result.latency = LatencyMetrics(serialization_ms=0.1, api_roundtrip_ms=0, parsing_ms=0.1)
        
        # Step 4: Store result locally
        self.local_simulator.store_inference_result(patient_id, timestamp, result)
        self._total_analyses += 1
        
        return result
    
    def analyze_batch(
        self,
        readings: List[Dict[str, Any]],
        call_api: bool = True
    ) -> List[InferenceResult]:
        """
        Analyze multiple readings.
        
        Args:
            readings: List of dicts with patient_id, timestamp, vitals
            call_api: Whether to call Cerebras API
        
        Returns:
            List of InferenceResults
        """
        results = []
        for reading in readings:
            result = self.analyze_reading(
                patient_id=reading["patient_id"],
                timestamp=reading["timestamp"],
                vitals=reading["vitals"],
                call_api=call_api
            )
            results.append(result)
        return results
    
    def get_risk_trajectory(self, patient_id: str) -> Tuple[List[float], List[float]]:
        """Get risk trajectory for visualization."""
        return self.local_simulator.get_risk_trajectory(patient_id)
    
    def get_vital_trajectory(self, patient_id: str, metric: str) -> Tuple[List[float], List[float]]:
        """Get vital sign trajectory for visualization."""
        return self.local_simulator.get_vital_trajectory(patient_id, metric)
    
    def get_metrics_info(self) -> Dict[str, Dict]:
        """Get available metrics with RX tags."""
        return self.local_simulator.get_metrics_info()
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status."""
        return {
            "inference_status": self.inference_client.get_status(),
            "is_using_cerebras": self.inference_client.is_available(),
            "latency_metrics": self.inference_client.get_latency_metrics(),
            "total_analyses": self._total_analyses,
            "api_calls": self._api_calls,
            "model": self.inference_client.model
        }
    
    def clear_patient(self, patient_id: str):
        """Clear patient data."""
        self.local_simulator.clear_patient(patient_id)


# =============================================================================
# LEGACY COMPATIBILITY - Keep old class names working
# =============================================================================

# For backward compatibility with existing code
CerebrasClient = InferenceClient
CerebrasRiskEngine = SepsisRiskAnalyzer


@dataclass
class CerebrasMetrics:
    """Legacy metrics class for backward compatibility."""
    latency_ms: float
    tokens_input: int
    tokens_output: int
    tokens_per_second: float
    reasoning_cycles: int
    model_used: str


@dataclass
class RiskAnalysisResult:
    """Legacy result class for backward compatibility."""
    risk_score: float
    confidence: float
    uncertainty: float
    trend_direction: str
    trend_acceleration: float
    key_factors: List[str]
    reasoning: str
    metrics: CerebrasMetrics
    regime_shift_detected: bool = False
    regime_description: str = ""
    correlation_abnormality: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)


@dataclass
class InferenceMetrics:
    """Legacy metrics class for backward compatibility."""
    latency_ms: float
    reasoning_cycles: int
    hypotheses_evaluated: int
    throughput_cycles_per_second: float
    tokens_processed: int = 0
    tokens_per_second: float = 0.0


# Legacy class stubs for imports that expect them
class MultiCycleReasoner:
    """Legacy stub - functionality now in InferenceClient."""
    def __init__(self, client, config=None):
        self.client = client
        self.config = config


class ParallelPatientProcessor:
    """Legacy stub - functionality now in InferenceClient.infer_batch."""
    def __init__(self, client):
        self.client = client


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # New clean architecture
    "InferenceClient",
    "LocalSimulator", 
    "SepsisRiskAnalyzer",
    "InferenceResult",
    "BatchInferenceResult",
    "LatencyMetrics",
    # Legacy compatibility
    "CerebrasClient",
    "CerebrasRiskEngine",
    "RiskAnalysisResult",
    "CerebrasMetrics",
    "InferenceMetrics",
    "MultiCycleReasoner",
    "ParallelPatientProcessor",
    "CEREBRAS_AVAILABLE",
]
