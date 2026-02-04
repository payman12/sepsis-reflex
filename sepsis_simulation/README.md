# Sepsis Risk Simulation System

**Real-time Physiological Simulation for Early Sepsis Risk Tracking**  
**Optimized for Cerebras Cloud Compute**

---

## ⚠️ Research Prototype Disclaimer

This is a **simulation system for research purposes only**.
- No clinical deployment claims are made
- No promises of "earliest diagnosis"
- Speed metrics measure **computational performance**, not biological onset timing

---

## 🎯 Objective

Build a simulation-first system that:
1. Ingests patient vital-sign time series
2. Continuously updates a **latent sepsis risk state** over time
3. Demonstrates **multi-cycle reasoning** per timestep
4. Tracks **uncertainty** in risk estimates
5. Leverages **Cerebras' massive parallelism** and low-latency memory

The goal is **smarter, deeper real-time inference**: tracking physiological regime shifts, trend acceleration, and uncertainty *before* formal diagnostic criteria are met.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SEPSIS SIMULATION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  Data Sources   │───▶│ Feature Extractor │───▶│ Risk Tracker  │  │
│  │                 │    │                  │    │  (Bayesian)   │  │
│  │ • Synthetic     │    │ • Slopes         │    │               │  │
│  │ • EHR Dataset   │    │ • Volatility     │    │ • Belief      │  │
│  │ • ICU Dataset   │    │ • Correlations   │    │ • Uncertainty │  │
│  └─────────────────┘    │ • Regime shifts  │    │ • Trend       │  │
│                         └──────────────────┘    └───────┬───────┘  │
│                                                         │          │
│                         ┌──────────────────────────────┐│          │
│                         │   CEREBRAS INFERENCE ENGINE  ││          │
│                         │                              ││          │
│                         │  • 10-20 reasoning cycles    │◀          │
│                         │  • 8+ parallel hypotheses    │           │
│                         │  • Sub-ms latency            │           │
│                         │  • 1000+ patients/sec        │           │
│                         └──────────────────────────────┘           │
│                                                                     │
│  ┌─────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   Evaluation    │    │    Dashboard     │    │    Output     │  │
│  │                 │    │   (Streamlit)    │    │               │  │
│  │ • AUROC/AUPRC   │    │                  │    │ • Risk over   │  │
│  │ • F1 Score      │    │ • Risk vs Time   │    │   time        │  │
│  │ • Latency       │    │ • Uncertainty    │    │ • Uncertainty │  │
│  │ • Throughput    │    │ • Comparison     │    │   bands       │  │
│  └─────────────────┘    └──────────────────┘    └───────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

```bash
# Navigate to the simulation directory
cd sepsis_simulation

# Install dependencies
pip install -r requirements.txt

# Optional: Install Cerebras SDK for accelerated inference
pip install cerebras-cloud-sdk
```

---

## 🚀 Quick Start

### 1. Run Comparison Simulation

```bash
python main.py --mode comparison --n-septic 10 --n-stable 10
```

### 2. Run Speed Benchmark

```bash
python main.py --mode benchmark --n-patients 100 --n-timesteps 100
```

### 3. Launch Dashboard

```bash
python main.py --mode dashboard
# or
streamlit run dashboard.py
```

### 4. Generate Synthetic Data

```bash
python main.py --mode generate --n-patients 50 --output synthetic_data.csv
```

### 5. Use Cerebras Cloud

```bash
python main.py --mode comparison --cerebras-key YOUR_API_KEY --n-cycles 20
```

---

## 📊 Input Signals

Each patient stream may include:

| Signal | Unit | Description |
|--------|------|-------------|
| Heart Rate (HR) | bpm | Cardiac rhythm |
| Mean Arterial Pressure (MAP) | mmHg | Perfusion pressure |
| Respiratory Rate (RR) | breaths/min | Ventilation |
| SpO₂ | % | Oxygen saturation |
| Temperature | °C | Core body temp |
| Lactate | mmol/L | Tissue hypoxia marker |
| WBC | K/uL | Infection marker |

---

## 🧠 Core Simulation Behavior

The system does **NOT** output binary "sepsis / no sepsis" decisions.

Instead, it:
1. **Maintains a latent risk state** per patient
2. **Updates continuously** as new signals arrive
3. **Tracks uncertainty** in the estimate
4. **Detects trend acceleration** and regime shifts

### Risk Regimes

| Regime | Risk Range | Description |
|--------|------------|-------------|
| Baseline | 0.0 - 0.25 | Normal physiological state |
| Elevated | 0.25 - 0.50 | Some concerning signals |
| High | 0.50 - 0.75 | Multiple abnormalities |
| Critical | 0.75 - 1.00 | Severe derangement |

---

## ⚡ Cerebras Utilization

### Why Cerebras for This Task

| Feature | GPU Limitation | Cerebras Advantage |
|---------|----------------|-------------------|
| Reasoning cycles | 1-3 (latency-limited) | 10-20 cycles/timestep |
| Per-cycle latency | 10-50ms | <1ms |
| Parallel hypotheses | Memory-limited | 8+ simultaneously |
| Patient throughput | ~100/sec | ~1000+/sec |

### Multi-Cycle Reasoning

```
Timestep t arrives
    │
    ├── Cycle 1: Raw observation update
    ├── Cycle 2: Trend analysis
    ├── Cycle 3: Cross-correlation patterns
    ├── Cycle 4: Hypothesis evaluation
    ├── Cycle 5-N: Ensemble refinement
    │
    └── Updated belief with reduced uncertainty
```

**Why this works on Cerebras:**
- On-chip SRAM (40GB) eliminates memory transfers
- Dataflow architecture makes sequential cycles nearly free
- Massive parallelism (850K+ cores) enables real-time processing

---

## 📈 Evaluation

### Accuracy Metrics

| Metric | Description |
|--------|-------------|
| AUROC | Area under ROC curve |
| AUPRC | Area under Precision-Recall curve |
| F1 Score | Harmonic mean of precision & recall |
| Sensitivity | True positive rate |
| Specificity | True negative rate |

### Speed Metrics

**⚠️ These measure COMPUTATIONAL performance, not biological timing.**

| Metric | Description |
|--------|-------------|
| Latency | Time from observation to risk update |
| Throughput | Observations processed per second |
| Cycles/sec | Reasoning cycles per second |

---

## 📁 Module Structure

```
sepsis_simulation/
├── __init__.py              # Package exports
├── config.py                # Configuration and constants
├── synthetic_data_generator.py  # High-resolution synthetic data
├── physiological_features.py    # Deterministic feature extraction
├── risk_state_tracker.py        # Bayesian latent risk state
├── cerebras_inference.py        # Cerebras-optimized inference
├── simulation_engine.py         # Main orchestration
├── evaluation.py                # Accuracy and speed metrics
├── dashboard.py                 # Streamlit visualization
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🗃️ Datasets (Evaluation Only)

Use these for case studies, **NOT** for supervised training:

1. **EHR ICU Dataset (Kaggle)**  
   https://www.kaggle.com/datasets/vipulshahi/ehr-data

2. **ICU Sepsis Dataset (Kaggle)**  
   https://www.kaggle.com/datasets/mehmetakifciftci/icu-sepsis-dataset-xlsx/data

3. **Synthetic Data** (built-in generator)  
   Generate with: `python main.py --mode generate`

---

## 📖 API Reference

### SepsisSimulationEngine

```python
from sepsis_simulation import SepsisSimulationEngine

# Initialize
engine = SepsisSimulationEngine(
    cerebras_api_key="YOUR_KEY",  # Optional
)

# Run comparison
septic, stable = engine.run_comparison(n_septic=10, n_stable=10)

# Process single observation
belief, features = engine.process_observation(
    patient_id="P001",
    timestamp=0.0,
    vital_values={"heart_rate": 95, "map": 70}
)

# Run benchmark
benchmark = engine.run_speed_benchmark(n_patients=100)
```

### SyntheticDataGenerator

```python
from sepsis_simulation import SyntheticDataGenerator

generator = SyntheticDataGenerator(random_seed=42)

# Single patient
patient = generator.generate_patient(
    patient_id="TEST_001",
    scenario="sepsis",  # or "stable", "deteriorating", "random"
    duration_hours=24
)

# Cohort
patients = generator.generate_cohort(n_patients=50)
```

---

## 🎨 Dashboard Features

Launch with: `streamlit run dashboard.py`

- **Risk Visualization**: Risk score over time with uncertainty bands
- **Comparison Analysis**: Side-by-side septic vs stable patients
- **Accuracy Metrics**: ROC curves, PR curves, confusion matrix
- **Performance Metrics**: Latency, throughput, Cerebras utilization

---

## 📜 License

MIT License - Research use only.

---

## 🤝 Contributing

Contributions welcome! Please ensure:
1. No clinical deployment claims
2. Clear documentation
3. Proper testing

---

## 📚 References

- Sepsis-3 Criteria
- SOFA Score methodology
- Bayesian filtering and state estimation
- Cerebras CS-2 architecture
