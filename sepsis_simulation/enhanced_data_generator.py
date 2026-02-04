"""
Enhanced Synthetic Data Generator for Sepsis Prediction
========================================================

This module generates high-fidelity synthetic patient data using ALL 60+ clinical
measurements from the MIMIC measurement_mappings.json file.

Key Features:
- Uses MIMIC measurement definitions for realistic value ranges
- Generates N different patients each run (randomized)
- Creates both simulation data AND expert labels for accuracy comparison
- Supports minute-level or hour-level timestamps
- Generates correlated physiological signals (not independent)

Cerebras Optimization Note:
---------------------------
This generator produces high-frequency time series data that benefits from:
- Cerebras' ability to process long sequences without memory constraints
- Parallel batch processing of multiple patient streams
- Fast inference for continuous risk assessment
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import random
from datetime import datetime, timedelta


@dataclass
class MeasurementSpec:
    """Specification for a clinical measurement based on MIMIC mappings"""
    name: str
    display_name: str
    unit: str
    category: str
    codes: List[str]
    hold_time: int  # hours
    # Physiological ranges (added based on clinical knowledge)
    normal_mean: float = 0.0
    normal_std: float = 1.0
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None
    sepsis_shift_mean: float = 0.0  # How much sepsis shifts the mean
    sepsis_shift_std: float = 0.0   # How much sepsis increases variability
    sampling_noise: float = 0.02    # Measurement noise as fraction of normal_std


# Clinical reference ranges for all measurements
# Based on medical literature and MIMIC-III/IV statistics
CLINICAL_RANGES: Dict[str, Dict] = {
    # Vital Signs
    "heart_rate": {"mean": 75, "std": 12, "low": 40, "high": 150, "sepsis_shift": 25, "sepsis_std": 8},
    "sbp_arterial": {"mean": 120, "std": 15, "low": 70, "high": 200, "sepsis_shift": -25, "sepsis_std": 10},
    "dbp_arterial": {"mean": 75, "std": 10, "low": 40, "high": 120, "sepsis_shift": -15, "sepsis_std": 8},
    "map": {"mean": 85, "std": 10, "low": 50, "high": 140, "sepsis_shift": -20, "sepsis_std": 8},
    "respiratory_rate": {"mean": 16, "std": 4, "low": 8, "high": 40, "sepsis_shift": 8, "sepsis_std": 4},
    "spo2": {"mean": 97, "std": 2, "low": 85, "high": 100, "sepsis_shift": -5, "sepsis_std": 3},
    "temp_C": {"mean": 37.0, "std": 0.5, "low": 35, "high": 41, "sepsis_shift": 1.5, "sepsis_std": 0.8},
    "temp_F": {"mean": 98.6, "std": 0.9, "low": 95, "high": 106, "sepsis_shift": 2.7, "sepsis_std": 1.4},
    
    # Hemodynamic
    "cvp": {"mean": 8, "std": 4, "low": 0, "high": 20, "sepsis_shift": 4, "sepsis_std": 3},
    "pap_systolic": {"mean": 25, "std": 6, "low": 15, "high": 60, "sepsis_shift": 10, "sepsis_std": 5},
    "pap_diastolic": {"mean": 12, "std": 4, "low": 5, "high": 30, "sepsis_shift": 5, "sepsis_std": 3},
    "pap_mean": {"mean": 16, "std": 4, "low": 10, "high": 35, "sepsis_shift": 6, "sepsis_std": 3},
    "cardiac_index": {"mean": 3.0, "std": 0.5, "low": 1.5, "high": 5.0, "sepsis_shift": -0.5, "sepsis_std": 0.3},
    
    # Respiratory
    "fio2": {"mean": 30, "std": 10, "low": 21, "high": 100, "sepsis_shift": 25, "sepsis_std": 15},
    "peep": {"mean": 5, "std": 2, "low": 0, "high": 24, "sepsis_shift": 4, "sepsis_std": 3},
    "tidal_volume": {"mean": 500, "std": 100, "low": 200, "high": 800, "sepsis_shift": 0, "sepsis_std": 50},
    "minute_volume": {"mean": 8, "std": 2, "low": 4, "high": 20, "sepsis_shift": 3, "sepsis_std": 2},
    "oxygen_flow": {"mean": 2, "std": 2, "low": 0, "high": 15, "sepsis_shift": 5, "sepsis_std": 3},
    "mean_airway_pressure": {"mean": 12, "std": 4, "low": 5, "high": 30, "sepsis_shift": 5, "sepsis_std": 3},
    "peak_inspiratory_pressure": {"mean": 20, "std": 5, "low": 10, "high": 40, "sepsis_shift": 8, "sepsis_std": 4},
    "plateau_pressure": {"mean": 18, "std": 4, "low": 10, "high": 35, "sepsis_shift": 6, "sepsis_std": 3},
    
    # Laboratory - Electrolytes
    "potassium": {"mean": 4.0, "std": 0.5, "low": 3.0, "high": 6.0, "sepsis_shift": 0.5, "sepsis_std": 0.3},
    "sodium": {"mean": 140, "std": 4, "low": 125, "high": 155, "sepsis_shift": -3, "sepsis_std": 3},
    "chloride": {"mean": 102, "std": 4, "low": 90, "high": 115, "sepsis_shift": 3, "sepsis_std": 3},
    "glucose": {"mean": 110, "std": 30, "low": 50, "high": 400, "sepsis_shift": 60, "sepsis_std": 40},
    "magnesium": {"mean": 2.0, "std": 0.3, "low": 1.2, "high": 3.0, "sepsis_shift": -0.3, "sepsis_std": 0.2},
    "calcium_total": {"mean": 9.0, "std": 0.8, "low": 7.0, "high": 11.0, "sepsis_shift": -0.8, "sepsis_std": 0.4},
    "calcium_ionized": {"mean": 1.15, "std": 0.1, "low": 0.9, "high": 1.4, "sepsis_shift": -0.1, "sepsis_std": 0.05},
    
    # Laboratory - Renal
    "creatinine": {"mean": 1.0, "std": 0.3, "low": 0.5, "high": 10.0, "sepsis_shift": 1.5, "sepsis_std": 1.0},
    "urea_nitrogen": {"mean": 15, "std": 6, "low": 5, "high": 100, "sepsis_shift": 20, "sepsis_std": 15},
    
    # Laboratory - Liver
    "ast": {"mean": 30, "std": 15, "low": 10, "high": 2000, "sepsis_shift": 100, "sepsis_std": 80},
    "alt": {"mean": 25, "std": 12, "low": 5, "high": 2000, "sepsis_shift": 80, "sepsis_std": 60},
    "bilirubin_total": {"mean": 0.8, "std": 0.4, "low": 0.1, "high": 20, "sepsis_shift": 2.0, "sepsis_std": 1.5},
    "bilirubin_direct": {"mean": 0.2, "std": 0.1, "low": 0, "high": 10, "sepsis_shift": 1.0, "sepsis_std": 0.8},
    "albumin": {"mean": 3.8, "std": 0.5, "low": 2.0, "high": 5.0, "sepsis_shift": -0.8, "sepsis_std": 0.3},
    "total_protein": {"mean": 6.5, "std": 0.8, "low": 4.0, "high": 9.0, "sepsis_shift": -0.5, "sepsis_std": 0.3},
    
    # Laboratory - Hematology
    "hemoglobin": {"mean": 12.5, "std": 2.0, "low": 7, "high": 18, "sepsis_shift": -1.5, "sepsis_std": 1.0},
    "hematocrit": {"mean": 38, "std": 5, "low": 20, "high": 55, "sepsis_shift": -4, "sepsis_std": 3},
    "wbc": {"mean": 8.0, "std": 3.0, "low": 2.0, "high": 40, "sepsis_shift": 10, "sepsis_std": 6},
    "platelets": {"mean": 250, "std": 80, "low": 50, "high": 500, "sepsis_shift": -100, "sepsis_std": 50},
    "red_blood_cells": {"mean": 4.5, "std": 0.6, "low": 3.0, "high": 6.0, "sepsis_shift": -0.5, "sepsis_std": 0.3},
    
    # Laboratory - Coagulation
    "pt": {"mean": 12, "std": 2, "low": 9, "high": 30, "sepsis_shift": 5, "sepsis_std": 4},
    "ptt": {"mean": 30, "std": 5, "low": 20, "high": 80, "sepsis_shift": 15, "sepsis_std": 10},
    "inr": {"mean": 1.0, "std": 0.2, "low": 0.8, "high": 5.0, "sepsis_shift": 0.8, "sepsis_std": 0.5},
    
    # Laboratory - Cardiac
    "troponin": {"mean": 0.01, "std": 0.02, "low": 0, "high": 10, "sepsis_shift": 0.5, "sepsis_std": 0.4},
    "crp": {"mean": 5, "std": 5, "low": 0, "high": 300, "sepsis_shift": 80, "sepsis_std": 50},
    
    # Blood Gas
    "ph_arterial": {"mean": 7.40, "std": 0.03, "low": 7.0, "high": 7.6, "sepsis_shift": -0.1, "sepsis_std": 0.05},
    "arterial_o2_pressure": {"mean": 90, "std": 15, "low": 50, "high": 500, "sepsis_shift": -20, "sepsis_std": 15},
    "arterial_co2_pressure": {"mean": 40, "std": 5, "low": 20, "high": 80, "sepsis_shift": -8, "sepsis_std": 5},
    "arterial_base_excess": {"mean": 0, "std": 3, "low": -15, "high": 15, "sepsis_shift": -5, "sepsis_std": 3},
    "lactic_acid": {"mean": 1.0, "std": 0.5, "low": 0.5, "high": 15, "sepsis_shift": 4, "sepsis_std": 3},
    "lactate": {"mean": 1.0, "std": 0.5, "low": 0.5, "high": 15, "sepsis_shift": 4, "sepsis_std": 3},
    "hco3": {"mean": 24, "std": 3, "low": 12, "high": 35, "sepsis_shift": -5, "sepsis_std": 3},
    "etco2": {"mean": 38, "std": 4, "low": 20, "high": 60, "sepsis_shift": -6, "sepsis_std": 4},
    "central_venous_o2_saturation": {"mean": 70, "std": 8, "low": 50, "high": 85, "sepsis_shift": -15, "sepsis_std": 8},
    "total_co2": {"mean": 25, "std": 3, "low": 15, "high": 35, "sepsis_shift": -4, "sepsis_std": 3},
    
    # Neurological
    "gcs": {"mean": 15, "std": 0.5, "low": 3, "high": 15, "sepsis_shift": -3, "sepsis_std": 2},
    "richmond_ras": {"mean": 0, "std": 1, "low": -5, "high": 4, "sepsis_shift": -2, "sepsis_std": 1},
    
    # Demographics (static)
    "height_cm": {"mean": 170, "std": 10, "low": 140, "high": 210, "sepsis_shift": 0, "sepsis_std": 0},
    "height_inch": {"mean": 67, "std": 4, "low": 55, "high": 82, "sepsis_shift": 0, "sepsis_std": 0},
    "weight_kg": {"mean": 75, "std": 15, "low": 40, "high": 200, "sepsis_shift": 0, "sepsis_std": 0},
    "weight_lb": {"mean": 165, "std": 33, "low": 88, "high": 440, "sepsis_shift": 0, "sepsis_std": 0},
}


class EnhancedMIMICDataGenerator:
    """
    Generates realistic synthetic patient data based on MIMIC measurement definitions.
    
    This generator creates physiologically correlated time series that can be used to:
    1. Train transformer models for sepsis prediction
    2. Generate simulation data for real-time risk assessment
    3. Create expert-labeled data for accuracy comparison
    
    Cerebras Benefits:
    -----------------
    - Large batch generation can be parallelized
    - Long sequences (48h+ at minute resolution) fit in Cerebras memory
    - Multiple patient streams can be processed simultaneously
    """
    
    def __init__(
        self,
        measurement_mappings_path: str = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the generator with MIMIC measurement definitions.
        
        Args:
            measurement_mappings_path: Path to measurement_mappings.json
            seed: Random seed for reproducibility (None for different data each run)
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Load measurement mappings
        if measurement_mappings_path is None:
            # Default path relative to this file
            base_path = Path(__file__).parent.parent
            measurement_mappings_path = base_path / "MIMIC-sepsis" / "src" / "ReferenceFiles" / "measurement_mappings.json"
        
        self.measurements = self._load_measurements(measurement_mappings_path)
        self.measurement_specs = self._create_measurement_specs()
        
        # Define measurement categories for correlation modeling
        self.vital_signs = ["heart_rate", "sbp_arterial", "dbp_arterial", "map", 
                          "respiratory_rate", "spo2", "temp_C"]
        self.lab_values = ["potassium", "sodium", "chloride", "glucose", "creatinine",
                         "hemoglobin", "wbc", "platelets", "lactic_acid", "crp"]
        self.blood_gas = ["ph_arterial", "arterial_o2_pressure", "arterial_co2_pressure",
                        "arterial_base_excess", "hco3"]
    
    def _load_measurements(self, path: str) -> Dict:
        """Load measurement mappings from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {path}, using default measurements")
            return {}
    
    def _create_measurement_specs(self) -> Dict[str, MeasurementSpec]:
        """Create MeasurementSpec objects from loaded mappings with clinical ranges"""
        specs = {}
        for name, info in self.measurements.items():
            clinical = CLINICAL_RANGES.get(name, {})
            specs[name] = MeasurementSpec(
                name=name,
                display_name=info.get("display_name", name),
                unit=info.get("unit", ""),
                category=info.get("category", "unknown"),
                codes=info.get("codes", []),
                hold_time=info.get("hold_time", 4),
                normal_mean=clinical.get("mean", 0),
                normal_std=clinical.get("std", 1),
                critical_low=clinical.get("low"),
                critical_high=clinical.get("high"),
                sepsis_shift_mean=clinical.get("sepsis_shift", 0),
                sepsis_shift_std=clinical.get("sepsis_std", 0)
            )
        return specs
    
    def generate_patient_cohort(
        self,
        n_patients: int,
        duration_hours: int = 48,
        timestep_minutes: int = 60,
        sepsis_ratio: float = 0.5,
        include_expert_labels: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a cohort of N patients with realistic vital sign trajectories.
        
        This is the main entry point for generating training/evaluation data.
        Each call with seed=None produces DIFFERENT patients.
        
        Args:
            n_patients: Number of patients to generate
            duration_hours: Duration of each patient trajectory (default 48h)
            timestep_minutes: Time resolution in minutes (60 = hourly, 1 = minute-level)
            sepsis_ratio: Fraction of patients who will develop sepsis
            include_expert_labels: Whether to include ground truth sepsis labels
            
        Returns:
            patient_data: DataFrame with all patient time series
            expert_labels: DataFrame with expert sepsis labels (onset time, severity, etc.)
        """
        all_patient_data = []
        all_expert_labels = []
        
        n_sepsis = int(n_patients * sepsis_ratio)
        n_stable = n_patients - n_sepsis
        
        # Generate patient IDs
        patient_ids = [f"P{i+1:04d}" for i in range(n_patients)]
        
        # Randomly assign sepsis status
        sepsis_patients = set(random.sample(patient_ids, n_sepsis))
        
        for pid in patient_ids:
            is_sepsis = pid in sepsis_patients
            
            # Generate individual patient trajectory
            patient_df, expert_label = self._generate_single_patient(
                patient_id=pid,
                duration_hours=duration_hours,
                timestep_minutes=timestep_minutes,
                develops_sepsis=is_sepsis,
                include_expert_labels=include_expert_labels
            )
            
            all_patient_data.append(patient_df)
            if include_expert_labels:
                all_expert_labels.append(expert_label)
        
        # Combine all patients
        patient_data = pd.concat(all_patient_data, ignore_index=True)
        
        if include_expert_labels:
            expert_labels = pd.DataFrame(all_expert_labels)
        else:
            expert_labels = pd.DataFrame()
        
        return patient_data, expert_labels
    
    def _generate_single_patient(
        self,
        patient_id: str,
        duration_hours: int,
        timestep_minutes: int,
        develops_sepsis: bool,
        include_expert_labels: bool
    ) -> Tuple[pd.DataFrame, Dict]:
        """Generate time series for a single patient"""
        
        n_timesteps = (duration_hours * 60) // timestep_minutes
        
        # Generate timestamps
        base_time = datetime.now() - timedelta(hours=duration_hours)
        timestamps = [base_time + timedelta(minutes=i * timestep_minutes) 
                     for i in range(n_timesteps)]
        
        # Initialize data dictionary
        data = {
            "patient_id": [patient_id] * n_timesteps,
            "timestamp": timestamps,
            "timestep": list(range(n_timesteps)),
            "hours_from_start": [i * timestep_minutes / 60 for i in range(n_timesteps)]
        }
        
        # Determine sepsis parameters if applicable
        if develops_sepsis:
            # Sepsis onset between 25% and 75% through the recording
            onset_fraction = np.random.uniform(0.25, 0.75)
            sepsis_onset_timestep = int(n_timesteps * onset_fraction)
            sepsis_onset_hours = sepsis_onset_timestep * timestep_minutes / 60
            
            # Severity: mild (1), moderate (2), severe (3)
            severity = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            
            # Rate of deterioration
            deterioration_rate = np.random.uniform(0.5, 2.0) * severity
        else:
            sepsis_onset_timestep = n_timesteps + 100  # Never triggers
            sepsis_onset_hours = None
            severity = 0
            deterioration_rate = 0
        
        # Generate correlated physiological signals
        # First generate underlying latent "health state" that evolves over time
        health_state = self._generate_health_state(
            n_timesteps, develops_sepsis, sepsis_onset_timestep, severity
        )
        
        # Generate each measurement based on health state
        for name, spec in self.measurement_specs.items():
            if spec.normal_mean == 0 and spec.normal_std == 1:
                # Skip measurements without defined ranges
                continue
            
            values = self._generate_measurement_series(
                spec=spec,
                n_timesteps=n_timesteps,
                health_state=health_state,
                develops_sepsis=develops_sepsis,
                sepsis_onset=sepsis_onset_timestep,
                severity=severity,
                timestep_minutes=timestep_minutes
            )
            data[name] = values
        
        # Add derived clinical scores (SOFA, SIRS approximations)
        data = self._add_clinical_scores(data, n_timesteps)
        
        # Create DataFrame
        patient_df = pd.DataFrame(data)
        
        # Expert label
        expert_label = {
            "patient_id": patient_id,
            "develops_sepsis": develops_sepsis,
            "sepsis_onset_hours": sepsis_onset_hours,
            "sepsis_onset_timestep": sepsis_onset_timestep if develops_sepsis else None,
            "severity": severity,
            "duration_hours": duration_hours,
            "total_timesteps": n_timesteps
        }
        
        return patient_df, expert_label
    
    def _generate_health_state(
        self,
        n_timesteps: int,
        develops_sepsis: bool,
        sepsis_onset: int,
        severity: int
    ) -> np.ndarray:
        """
        Generate a latent health state trajectory.
        
        This underlying state influences all measurements, creating
        physiologically realistic correlations between signals.
        
        Returns:
            Array of health states from 0 (healthy) to 1 (severely ill)
        """
        health = np.zeros(n_timesteps)
        
        if develops_sepsis:
            for t in range(n_timesteps):
                if t < sepsis_onset:
                    # Pre-sepsis: slight random fluctuations
                    health[t] = 0.05 + 0.05 * np.sin(t * 0.1) + np.random.normal(0, 0.02)
                else:
                    # Post-onset: progressive deterioration with some variability
                    time_since_onset = t - sepsis_onset
                    # Sigmoid-like increase
                    max_severity = 0.3 + 0.2 * severity  # 0.5 to 0.9
                    rate = 0.05 * severity
                    base_health = max_severity * (1 - np.exp(-rate * time_since_onset))
                    health[t] = base_health + np.random.normal(0, 0.05)
        else:
            # Stable patient with minor fluctuations
            for t in range(n_timesteps):
                health[t] = 0.05 + 0.05 * np.sin(t * 0.05) + np.random.normal(0, 0.02)
        
        return np.clip(health, 0, 1)
    
    def _generate_measurement_series(
        self,
        spec: MeasurementSpec,
        n_timesteps: int,
        health_state: np.ndarray,
        develops_sepsis: bool,
        sepsis_onset: int,
        severity: int,
        timestep_minutes: int
    ) -> np.ndarray:
        """
        Generate a time series for a single measurement.
        
        The measurement is influenced by:
        1. The latent health state
        2. Physiological inertia (values don't change instantly)
        3. Measurement noise
        4. Category-specific sampling rates (labs less frequent than vitals)
        """
        values = np.zeros(n_timesteps)
        
        # Determine sampling frequency based on category
        if spec.category in ["vital", "hemodynamic"]:
            sample_every = 1  # Every timestep
        elif spec.category in ["respiratory"]:
            sample_every = max(1, 15 // timestep_minutes)  # Every 15 min minimum
        elif spec.category in ["laboratory", "blood_gas"]:
            sample_every = max(1, 240 // timestep_minutes)  # Every 4 hours
        else:
            sample_every = max(1, 60 // timestep_minutes)
        
        # Generate values
        last_value = spec.normal_mean
        for t in range(n_timesteps):
            # Base value from normal distribution
            base = spec.normal_mean
            
            # Apply sepsis shift based on health state
            if develops_sepsis:
                sepsis_effect = health_state[t] * spec.sepsis_shift_mean
                noise_increase = health_state[t] * spec.sepsis_shift_std
                base += sepsis_effect
                current_std = spec.normal_std + noise_increase
            else:
                current_std = spec.normal_std
            
            # Add physiological noise
            noise = np.random.normal(0, current_std * spec.sampling_noise)
            
            # Apply inertia (values change gradually)
            inertia = 0.9 if spec.category == "vital" else 0.95
            target = base + noise
            new_value = inertia * last_value + (1 - inertia) * target
            
            # Only record if this is a sample point
            if t % sample_every == 0:
                # Add measurement noise
                measurement_noise = np.random.normal(0, current_std * 0.02)
                values[t] = new_value + measurement_noise
                last_value = new_value
            else:
                # Carry forward last value (sample and hold)
                values[t] = last_value
            
            # Clip to valid ranges
            if spec.critical_low is not None:
                values[t] = max(values[t], spec.critical_low * 0.8)
            if spec.critical_high is not None:
                values[t] = min(values[t], spec.critical_high * 1.2)
        
        return values
    
    def _add_clinical_scores(self, data: Dict, n_timesteps: int) -> Dict:
        """Add derived clinical scores like SOFA and SIRS approximations"""
        
        # Initialize scores
        sofa_scores = np.zeros(n_timesteps)
        sirs_scores = np.zeros(n_timesteps)
        
        for t in range(n_timesteps):
            # SIRS criteria (simplified)
            sirs = 0
            if "temp_C" in data and (data["temp_C"][t] > 38 or data["temp_C"][t] < 36):
                sirs += 1
            if "heart_rate" in data and data["heart_rate"][t] > 90:
                sirs += 1
            if "respiratory_rate" in data and data["respiratory_rate"][t] > 20:
                sirs += 1
            if "wbc" in data and (data["wbc"][t] > 12 or data["wbc"][t] < 4):
                sirs += 1
            sirs_scores[t] = sirs
            
            # SOFA score approximation (simplified)
            sofa = 0
            
            # Respiratory (PaO2/FiO2 proxy via SpO2)
            if "spo2" in data:
                if data["spo2"][t] < 90:
                    sofa += 2
                elif data["spo2"][t] < 95:
                    sofa += 1
            
            # Cardiovascular (MAP)
            if "map" in data:
                if data["map"][t] < 65:
                    sofa += 2
                elif data["map"][t] < 70:
                    sofa += 1
            
            # Renal (Creatinine)
            if "creatinine" in data:
                if data["creatinine"][t] > 3.5:
                    sofa += 3
                elif data["creatinine"][t] > 2.0:
                    sofa += 2
                elif data["creatinine"][t] > 1.2:
                    sofa += 1
            
            # Coagulation (Platelets)
            if "platelets" in data:
                if data["platelets"][t] < 50:
                    sofa += 3
                elif data["platelets"][t] < 100:
                    sofa += 2
                elif data["platelets"][t] < 150:
                    sofa += 1
            
            # Liver (Bilirubin)
            if "bilirubin_total" in data:
                if data["bilirubin_total"][t] > 6:
                    sofa += 3
                elif data["bilirubin_total"][t] > 2:
                    sofa += 2
                elif data["bilirubin_total"][t] > 1.2:
                    sofa += 1
            
            # CNS (GCS)
            if "gcs" in data:
                if data["gcs"][t] < 6:
                    sofa += 4
                elif data["gcs"][t] < 10:
                    sofa += 3
                elif data["gcs"][t] < 13:
                    sofa += 2
                elif data["gcs"][t] < 15:
                    sofa += 1
            
            sofa_scores[t] = sofa
        
        data["sirs_score"] = sirs_scores
        data["sofa_score"] = sofa_scores
        
        # Add sepsis flag based on Sepsis-3 criteria (SOFA >= 2)
        data["sepsis_flag"] = [1 if s >= 2 else 0 for s in sofa_scores]
        
        return data
    
    def generate_transformer_training_data(
        self,
        n_patients: int,
        window_size: int = 6,
        prediction_horizon: int = 6,
        timestep_hours: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate data specifically formatted for transformer model training.
        
        This produces windowed time series data compatible with the
        TimeSeriesTransformer from MIMIC-sepsis/src/transformer_model.py
        
        Args:
            n_patients: Number of patients to generate
            window_size: Number of timesteps in each input window
            prediction_horizon: How far ahead to predict (in timesteps)
            timestep_hours: Hours per timestep (default 4 like MIMIC benchmark)
            
        Returns:
            X: Features array of shape (n_samples, window_size, n_features)
            y: Labels array of shape (n_samples,)
            feature_names: List of feature column names
        """
        # Generate raw patient data
        patient_data, expert_labels = self.generate_patient_cohort(
            n_patients=n_patients,
            duration_hours=96,  # 4 days to have enough windows
            timestep_minutes=timestep_hours * 60,
            sepsis_ratio=0.5,
            include_expert_labels=True
        )
        
        # Get feature columns (exclude metadata)
        exclude_cols = ["patient_id", "timestamp", "timestep", "hours_from_start", 
                       "sepsis_flag", "sirs_score", "sofa_score"]
        feature_cols = [c for c in patient_data.columns if c not in exclude_cols 
                       and patient_data[c].dtype in [np.float64, np.int64, float, int]]
        
        # Create sliding windows
        X_list = []
        y_list = []
        
        for pid in patient_data["patient_id"].unique():
            patient_df = patient_data[patient_data["patient_id"] == pid].sort_values("timestep")
            features = patient_df[feature_cols].values
            sepsis_flags = patient_df["sepsis_flag"].values
            
            # Create windows
            for i in range(len(patient_df) - window_size - prediction_horizon + 1):
                window = features[i:i + window_size]
                
                # Target: does sepsis occur in prediction horizon?
                future = sepsis_flags[i + window_size:i + window_size + prediction_horizon]
                target = 1 if np.max(future) > 0 else 0
                
                X_list.append(window)
                y_list.append(target)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y, feature_cols


def generate_expert_vs_ai_comparison_data(
    n_patients: int = 20,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Generate data for comparing expert labels vs AI predictions.
    
    This function creates:
    1. Simulation data (what the AI sees)
    2. Expert labels (ground truth)
    3. Comparison metrics structure
    
    Args:
        n_patients: Number of patients to generate
        seed: Random seed (None for different data each run)
        
    Returns:
        simulation_data: Time series data for AI processing
        expert_labels: Ground truth labels from "expert"
        comparison_config: Configuration for accuracy metrics
    """
    generator = EnhancedMIMICDataGenerator(seed=seed)
    
    # Generate patient cohort
    patient_data, expert_labels = generator.generate_patient_cohort(
        n_patients=n_patients,
        duration_hours=48,
        timestep_minutes=60,  # Hourly
        sepsis_ratio=0.5,
        include_expert_labels=True
    )
    
    # Create simulation data (hide certain expert columns from AI)
    simulation_columns = [c for c in patient_data.columns 
                         if c not in ["sepsis_flag"]]  # Hide ground truth
    simulation_data = patient_data[simulation_columns].copy()
    
    # Comparison configuration
    comparison_config = {
        "n_patients": n_patients,
        "sepsis_patients": int(expert_labels["develops_sepsis"].sum()),
        "non_sepsis_patients": n_patients - int(expert_labels["develops_sepsis"].sum()),
        "metrics_to_compute": ["auroc", "auprc", "sensitivity", "specificity", "f1"],
        "evaluation_points": ["at_onset", "6h_before", "12h_before"],
        "seed": seed,
        "generated_at": datetime.now().isoformat()
    }
    
    return simulation_data, expert_labels, comparison_config


if __name__ == "__main__":
    # Demo: Generate different data each run
    print("=" * 60)
    print("Enhanced MIMIC Data Generator Demo")
    print("=" * 60)
    
    # Generate 10 patients with random seed (different each time)
    generator = EnhancedMIMICDataGenerator(seed=None)
    
    patient_data, expert_labels = generator.generate_patient_cohort(
        n_patients=10,
        duration_hours=48,
        timestep_minutes=60,
        sepsis_ratio=0.5
    )
    
    print(f"\nGenerated {len(patient_data['patient_id'].unique())} patients")
    print(f"Total timesteps: {len(patient_data)}")
    print(f"Features: {len([c for c in patient_data.columns if c not in ['patient_id', 'timestamp', 'timestep']])}")
    print(f"\nSepsis patients: {expert_labels['develops_sepsis'].sum()}")
    print(f"Non-sepsis patients: {(~expert_labels['develops_sepsis']).sum()}")
    
    print("\nSample patient data:")
    print(patient_data.head())
    
    print("\nExpert labels:")
    print(expert_labels)
