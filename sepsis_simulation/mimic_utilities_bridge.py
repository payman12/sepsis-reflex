"""
MIMIC Utilities Bridge
=======================

This module bridges useful code from the MIMIC-sepsis/src folder to enhance
the risk scoring and vital signal analysis capabilities of the simulation.

Key Components Bridged:
- Measurement definitions from measurement_mappings.json
- Outlier handling from format_traj.py
- Clinical score calculations (SOFA, SIRS) from format_traj.py
- Metrics calculation from metrics.py
- Data processing utilities from data_processor.py

Cerebras Optimization Note:
---------------------------
These utilities help create more realistic data that better matches
MIMIC distributions, enabling:
- More accurate model training
- Better generalization to real clinical data
- Consistent feature engineering across training and inference
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


# Path to MIMIC-sepsis source
MIMIC_SRC_PATH = Path(__file__).parent.parent / "MIMIC-sepsis" / "src"
REFERENCE_FILES_PATH = MIMIC_SRC_PATH / "ReferenceFiles"


def load_measurement_mappings() -> Tuple[Dict, Dict, Dict]:
    """
    Load the measurement mappings from MIMIC reference files.
    
    This function replicates the load_measurement_mappings from format_traj.py
    
    Returns:
        measurements: Dictionary of measurement definitions
        code_to_concept: Mapping from MIMIC codes to concept names
        hold_times: Dictionary of hold times for each measurement
    """
    mapping_path = REFERENCE_FILES_PATH / "measurement_mappings.json"
    
    try:
        with open(mapping_path, 'r') as f:
            measurements = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {mapping_path}")
        return {}, {}, {}
    
    # Create reverse mapping (code to concept)
    code_to_concept = {}
    for concept, info in measurements.items():
        for code in info.get('codes', []):
            code_to_concept[code] = concept
    
    # Create hold times mapping
    hold_times = {}
    for concept, info in measurements.items():
        if 'hold_time' in info:
            hold_times[concept] = info['hold_time']
    
    return measurements, code_to_concept, hold_times


@dataclass
class ClinicalThresholds:
    """Clinical thresholds for outlier detection and scoring"""
    # Vital signs
    heart_rate_max: float = 250
    sbp_max: float = 300
    map_min: float = 0
    map_max: float = 200
    dbp_min: float = 0
    dbp_max: float = 200
    respiratory_rate_max: float = 80
    spo2_max: float = 100
    
    # Laboratory values
    potassium_min: float = 1
    potassium_max: float = 15
    sodium_min: float = 95
    sodium_max: float = 178
    chloride_min: float = 70
    chloride_max: float = 150
    glucose_min: float = 1
    glucose_max: float = 1000
    creatinine_max: float = 150
    
    # Blood gas
    ph_min: float = 6.7
    ph_max: float = 8.0
    lactic_acid_max: float = 30
    
    # Hematology
    wbc_max: float = 500
    platelets_max: float = 2000
    hemoglobin_max: float = 20
    hematocrit_max: float = 65
    
    # Coagulation
    inr_max: float = 20


def handle_outliers(df: pd.DataFrame, thresholds: ClinicalThresholds = None) -> pd.DataFrame:
    """
    Handle outliers in patient timeseries data based on clinical thresholds.
    
    This function replicates the handle_outliers from format_traj.py
    
    Args:
        df: DataFrame containing patient measurements
        thresholds: Clinical thresholds (uses defaults if None)
        
    Returns:
        DataFrame with outliers handled
    """
    if thresholds is None:
        thresholds = ClinicalThresholds()
    
    df = df.copy()
    
    # Heart Rate
    if 'heart_rate' in df.columns:
        df.loc[df['heart_rate'] > thresholds.heart_rate_max, 'heart_rate'] = np.nan
    
    # Blood Pressure
    if 'sbp_arterial' in df.columns:
        df.loc[df['sbp_arterial'] > thresholds.sbp_max, 'sbp_arterial'] = np.nan
    if 'map' in df.columns:
        df.loc[df['map'] < thresholds.map_min, 'map'] = np.nan
        df.loc[df['map'] > thresholds.map_max, 'map'] = np.nan
    if 'dbp_arterial' in df.columns:
        df.loc[df['dbp_arterial'] < thresholds.dbp_min, 'dbp_arterial'] = np.nan
        df.loc[df['dbp_arterial'] > thresholds.dbp_max, 'dbp_arterial'] = np.nan
    
    # Respiratory
    if 'respiratory_rate' in df.columns:
        df.loc[df['respiratory_rate'] > thresholds.respiratory_rate_max, 'respiratory_rate'] = np.nan
    if 'spo2' in df.columns:
        df.loc[df['spo2'] > 100, 'spo2'] = 100
    
    # Laboratory values
    if 'potassium' in df.columns:
        df.loc[df['potassium'] < thresholds.potassium_min, 'potassium'] = np.nan
        df.loc[df['potassium'] > thresholds.potassium_max, 'potassium'] = np.nan
    
    if 'sodium' in df.columns:
        df.loc[df['sodium'] < thresholds.sodium_min, 'sodium'] = np.nan
        df.loc[df['sodium'] > thresholds.sodium_max, 'sodium'] = np.nan
    
    if 'glucose' in df.columns:
        df.loc[df['glucose'] < thresholds.glucose_min, 'glucose'] = np.nan
        df.loc[df['glucose'] > thresholds.glucose_max, 'glucose'] = np.nan
    
    if 'creatinine' in df.columns:
        df.loc[df['creatinine'] > thresholds.creatinine_max, 'creatinine'] = np.nan
    
    # Blood gas
    if 'ph_arterial' in df.columns:
        df.loc[df['ph_arterial'] < thresholds.ph_min, 'ph_arterial'] = np.nan
        df.loc[df['ph_arterial'] > thresholds.ph_max, 'ph_arterial'] = np.nan
    
    if 'lactic_acid' in df.columns:
        df.loc[df['lactic_acid'] > thresholds.lactic_acid_max, 'lactic_acid'] = np.nan
    
    # Hematology
    if 'wbc' in df.columns:
        df.loc[df['wbc'] > thresholds.wbc_max, 'wbc'] = np.nan
    if 'platelets' in df.columns:
        df.loc[df['platelets'] > thresholds.platelets_max, 'platelets'] = np.nan
    if 'hemoglobin' in df.columns:
        df.loc[df['hemoglobin'] > thresholds.hemoglobin_max, 'hemoglobin'] = np.nan
    if 'hematocrit' in df.columns:
        df.loc[df['hematocrit'] > thresholds.hematocrit_max, 'hematocrit'] = np.nan
    
    # Coagulation
    if 'inr' in df.columns:
        df.loc[df['inr'] > thresholds.inr_max, 'inr'] = np.nan
    
    return df


def calculate_sofa_score(row: pd.Series) -> int:
    """
    Calculate SOFA score for a single timestep.
    
    Based on Sepsis-3 definition from MIMIC-sepsis/src/format_traj.py
    
    Args:
        row: Series containing patient measurements
        
    Returns:
        SOFA score (0-24)
    """
    sofa = 0
    
    # Respiratory (using SpO2 as proxy for PaO2/FiO2)
    if 'spo2' in row and not pd.isna(row.get('spo2')):
        spo2 = row['spo2']
        if spo2 < 85:
            sofa += 4
        elif spo2 < 90:
            sofa += 3
        elif spo2 < 95:
            sofa += 2
        elif spo2 < 97:
            sofa += 1
    
    # If we have actual PaO2/FiO2 ratio
    if 'pf_ratio' in row and not pd.isna(row.get('pf_ratio')):
        pf = row['pf_ratio']
        if pf < 100:
            sofa += 4
        elif pf < 200:
            sofa += 3
        elif pf < 300:
            sofa += 2
        elif pf < 400:
            sofa += 1
    
    # Coagulation (Platelets)
    if 'platelets' in row and not pd.isna(row.get('platelets')):
        plt = row['platelets']
        if plt < 20:
            sofa += 4
        elif plt < 50:
            sofa += 3
        elif plt < 100:
            sofa += 2
        elif plt < 150:
            sofa += 1
    
    # Liver (Bilirubin)
    if 'bilirubin_total' in row and not pd.isna(row.get('bilirubin_total')):
        bili = row['bilirubin_total']
        if bili >= 12.0:
            sofa += 4
        elif bili >= 6.0:
            sofa += 3
        elif bili >= 2.0:
            sofa += 2
        elif bili >= 1.2:
            sofa += 1
    
    # Cardiovascular (MAP)
    if 'map' in row and not pd.isna(row.get('map')):
        map_val = row['map']
        if map_val < 55:
            sofa += 4
        elif map_val < 60:
            sofa += 3
        elif map_val < 65:
            sofa += 2
        elif map_val < 70:
            sofa += 1
    
    # CNS (GCS)
    if 'gcs' in row and not pd.isna(row.get('gcs')):
        gcs = row['gcs']
        if gcs < 6:
            sofa += 4
        elif gcs < 10:
            sofa += 3
        elif gcs < 13:
            sofa += 2
        elif gcs < 15:
            sofa += 1
    
    # Renal (Creatinine)
    if 'creatinine' in row and not pd.isna(row.get('creatinine')):
        cr = row['creatinine']
        if cr >= 5.0:
            sofa += 4
        elif cr >= 3.5:
            sofa += 3
        elif cr >= 2.0:
            sofa += 2
        elif cr >= 1.2:
            sofa += 1
    
    return sofa


def calculate_sirs_score(row: pd.Series) -> int:
    """
    Calculate SIRS score for a single timestep.
    
    SIRS criteria from MIMIC-sepsis/src/format_traj.py
    
    Args:
        row: Series containing patient measurements
        
    Returns:
        SIRS score (0-4)
    """
    sirs = 0
    
    # Temperature criterion
    if 'temp_C' in row and not pd.isna(row.get('temp_C')):
        temp = row['temp_C']
        if temp >= 38 or temp <= 36:
            sirs += 1
    elif 'temp_F' in row and not pd.isna(row.get('temp_F')):
        temp_f = row['temp_F']
        if temp_f >= 100.4 or temp_f <= 96.8:
            sirs += 1
    
    # Heart rate criterion
    if 'heart_rate' in row and not pd.isna(row.get('heart_rate')):
        hr = row['heart_rate']
        if hr > 90:
            sirs += 1
    
    # Respiratory criterion
    if 'respiratory_rate' in row and not pd.isna(row.get('respiratory_rate')):
        rr = row['respiratory_rate']
        if rr >= 20:
            sirs += 1
    elif 'arterial_co2_pressure' in row and not pd.isna(row.get('arterial_co2_pressure')):
        paco2 = row['arterial_co2_pressure']
        if paco2 <= 32:
            sirs += 1
    
    # WBC criterion
    if 'wbc' in row and not pd.isna(row.get('wbc')):
        wbc = row['wbc']
        if wbc >= 12 or wbc < 4:
            sirs += 1
    
    return sirs


def calculate_shock_index(row: pd.Series) -> float:
    """
    Calculate Shock Index (HR/SBP).
    
    Args:
        row: Series containing patient measurements
        
    Returns:
        Shock index value
    """
    hr = row.get('heart_rate')
    sbp = row.get('sbp_arterial')
    
    if pd.isna(hr) or pd.isna(sbp) or sbp == 0:
        return np.nan
    
    return hr / sbp


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived clinical features to the dataframe.
    
    Features added:
    - SOFA score
    - SIRS score
    - Shock index
    - Trend indicators (if time series)
    
    Args:
        df: DataFrame with patient measurements
        
    Returns:
        DataFrame with added derived features
    """
    df = df.copy()
    
    # Calculate scores row by row
    df['sofa_score_calc'] = df.apply(calculate_sofa_score, axis=1)
    df['sirs_score_calc'] = df.apply(calculate_sirs_score, axis=1)
    df['shock_index'] = df.apply(calculate_shock_index, axis=1)
    
    # Calculate trends if we have a patient_id column
    if 'patient_id' in df.columns and 'timestep' in df.columns:
        df = df.sort_values(['patient_id', 'timestep'])
        
        # Add trend features for key vitals
        for col in ['heart_rate', 'map', 'respiratory_rate', 'temp_C']:
            if col in df.columns:
                # Calculate rolling mean and trend
                df[f'{col}_trend'] = df.groupby('patient_id')[col].transform(
                    lambda x: x.diff()
                )
                df[f'{col}_rolling_mean'] = df.groupby('patient_id')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
    
    return df


class EnhancedRiskScorer:
    """
    Enhanced risk scoring using MIMIC-derived features and algorithms.
    
    This class combines:
    - SOFA score (Sepsis-3 definition)
    - SIRS score (traditional sepsis criteria)
    - Shock index
    - Trend analysis
    - Bayesian risk updating
    
    To provide a more clinically-grounded risk assessment.
    """
    
    def __init__(self):
        """Initialize the enhanced risk scorer"""
        self.measurements, self.code_to_concept, self.hold_times = load_measurement_mappings()
        self.thresholds = ClinicalThresholds()
        self.risk_history: List[float] = []
        self.sofa_history: List[int] = []
        self.sirs_history: List[int] = []
        
    def reset(self):
        """Reset scorer state for new patient"""
        self.risk_history = []
        self.sofa_history = []
        self.sirs_history = []
    
    def calculate_risk(self, row: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score for a single measurement.
        
        Args:
            row: Series containing patient measurements
            
        Returns:
            Dictionary with risk score and components
        """
        # Calculate component scores
        sofa = calculate_sofa_score(row)
        sirs = calculate_sirs_score(row)
        shock_idx = calculate_shock_index(row)
        
        # Store history
        self.sofa_history.append(sofa)
        self.sirs_history.append(sirs)
        
        # Calculate SOFA trend
        sofa_trend = 0
        if len(self.sofa_history) >= 2:
            sofa_trend = sofa - self.sofa_history[-2]
        
        # Calculate composite risk score
        # Weights based on clinical significance
        base_risk = 0.0
        
        # SOFA contribution (0-24 range, mapped to 0-0.4)
        base_risk += min(sofa / 24, 1.0) * 0.4
        
        # SIRS contribution (0-4 range, mapped to 0-0.2)
        base_risk += min(sirs / 4, 1.0) * 0.2
        
        # Shock index contribution (normal ~0.5-0.7, elevated >1.0)
        if not np.isnan(shock_idx):
            shock_contrib = max(0, (shock_idx - 0.7) / 0.6)  # Normalized
            base_risk += min(shock_contrib, 1.0) * 0.2
        
        # Trend contribution (worsening SOFA is concerning)
        if sofa_trend > 0:
            base_risk += min(sofa_trend / 4, 1.0) * 0.2
        
        # Clip to valid range
        risk = np.clip(base_risk, 0.0, 1.0)
        self.risk_history.append(risk)
        
        return {
            'risk_score': risk,
            'sofa_score': sofa,
            'sirs_score': sirs,
            'shock_index': shock_idx,
            'sofa_trend': sofa_trend,
            'components': {
                'sofa_contrib': min(sofa / 24, 1.0) * 0.4,
                'sirs_contrib': min(sirs / 4, 1.0) * 0.2,
                'shock_contrib': min((shock_idx - 0.7) / 0.6, 1.0) * 0.2 if not np.isnan(shock_idx) else 0,
                'trend_contrib': min(sofa_trend / 4, 1.0) * 0.2 if sofa_trend > 0 else 0
            }
        }
    
    def calculate_patient_risk_trajectory(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk trajectory for entire patient stay.
        
        Args:
            patient_df: DataFrame with patient time series
            
        Returns:
            DataFrame with added risk columns
        """
        self.reset()
        
        df = patient_df.copy()
        df = df.sort_values('timestep') if 'timestep' in df.columns else df
        
        risk_data = []
        for idx, row in df.iterrows():
            risk_result = self.calculate_risk(row)
            risk_data.append({
                'idx': idx,
                'enhanced_risk_score': risk_result['risk_score'],
                'sofa_calc': risk_result['sofa_score'],
                'sirs_calc': risk_result['sirs_score'],
                'shock_index_calc': risk_result['shock_index'],
                'sofa_trend': risk_result['sofa_trend']
            })
        
        risk_df = pd.DataFrame(risk_data).set_index('idx')
        
        for col in risk_df.columns:
            df[col] = risk_df[col]
        
        return df


# Metrics calculation (from metrics.py)
def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Based on MIMIC-sepsis/src/metrics.py
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve
    )
    
    metrics = {}
    
    if len(np.unique(y_true)) < 2:
        return {'auroc': 0.5, 'auprc': 0.5, 'sensitivity': 0, 'specificity': 0}
    
    # AUROC and AUPRC
    metrics['auroc'] = roc_auc_score(y_true, y_pred)
    metrics['auprc'] = average_precision_score(y_true, y_pred)
    
    # Find optimal threshold using PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[min(optimal_idx, len(thresholds)-1)]
    
    # Calculate metrics at optimal threshold
    y_pred_binary = (y_pred >= optimal_threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics['optimal_threshold'] = optimal_threshold
    
    return metrics


if __name__ == "__main__":
    # Demo usage
    print("MIMIC Utilities Bridge Demo")
    print("=" * 50)
    
    # Load measurement mappings
    measurements, code_to_concept, hold_times = load_measurement_mappings()
    print(f"\nLoaded {len(measurements)} measurement definitions")
    print(f"Categories: {set(m.get('category') for m in measurements.values())}")
    
    # Demo clinical score calculation
    sample_data = pd.Series({
        'heart_rate': 105,
        'map': 62,
        'respiratory_rate': 24,
        'spo2': 92,
        'temp_C': 38.5,
        'wbc': 14.5,
        'platelets': 120,
        'creatinine': 1.8,
        'bilirubin_total': 1.5,
        'gcs': 14
    })
    
    sofa = calculate_sofa_score(sample_data)
    sirs = calculate_sirs_score(sample_data)
    shock_idx = calculate_shock_index(sample_data)
    
    print(f"\nSample patient scores:")
    print(f"  SOFA: {sofa}")
    print(f"  SIRS: {sirs}")
    print(f"  Shock Index: {shock_idx:.2f}")
