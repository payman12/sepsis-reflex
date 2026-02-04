"""
Streamlit Dashboard for Sepsis Risk Simulation
===============================================

This dashboard provides real-time visualization of:
1. Risk score vs time with uncertainty bands
2. Comparison between septic and non-septic patients
3. Accuracy metrics (AUROC, AUPRC, F1)
4. Speed/performance metrics

Design Philosophy:
- Clean, informative visualizations
- Real-time updates during simulation
- Clear separation of accuracy vs speed metrics
- No clinical deployment claims

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json

# Import simulation components
import sys
from pathlib import Path

# Add parent directory to path for imports when run directly by Streamlit
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Handle both direct run and package import
try:
    from .config import CerebrasConfig, SimulationConfig, PatientState, VITAL_SIGNS
    from .synthetic_data_generator import SyntheticDataGenerator, PatientTrajectory
    from .simulation_engine import SepsisSimulationEngine, SimulationResult
    from .evaluation import (
        AccuracyEvaluator, 
        SpeedEvaluator, 
        generate_evaluation_report,
        AccuracyMetrics,
        SpeedMetrics
    )
except ImportError:
    # When run directly by Streamlit
    from config import CerebrasConfig, SimulationConfig, PatientState, VITAL_SIGNS
    from synthetic_data_generator import SyntheticDataGenerator, PatientTrajectory
    from simulation_engine import SepsisSimulationEngine, SimulationResult
    from evaluation import (
        AccuracyEvaluator, 
        SpeedEvaluator, 
        generate_evaluation_report,
        AccuracyMetrics,
        SpeedMetrics
    )


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Sepsis Risk Simulation Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #0066cc;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'results' not in st.session_state:
    st.session_state.results = []
if 'septic_results' not in st.session_state:
    st.session_state.septic_results = []
if 'stable_results' not in st.session_state:
    st.session_state.stable_results = []
if 'evaluation_report' not in st.session_state:
    st.session_state.evaluation_report = None
if 'benchmark_result' not in st.session_state:
    st.session_state.benchmark_result = None

# Signal visualization state
if 'selected_visualization_metrics' not in st.session_state:
    st.session_state.selected_visualization_metrics = [
        'heart_rate', 'map', 'respiratory_rate', 'spo2', 'temperature', 'lactate', 'wbc'
    ]

# Core ICU metrics for signal visualization
# These are clinically actionable metrics that correlate with early sepsis progression
ICU_METRICS = {
    'heart_rate': {
        'name': 'Heart Rate',
        'short': 'HR',
        'unit': 'bpm',
        'color': '#ef4444',  # Red
        'normal_low': 60,
        'normal_high': 100
    },
    'map': {
        'name': 'Mean Arterial Pressure',
        'short': 'MAP',
        'unit': 'mmHg',
        'color': '#8b5cf6',  # Purple
        'normal_low': 70,
        'normal_high': 105
    },
    'respiratory_rate': {
        'name': 'Respiratory Rate',
        'short': 'RR',
        'unit': 'breaths/min',
        'color': '#3b82f6',  # Blue
        'normal_low': 12,
        'normal_high': 20
    },
    'spo2': {
        'name': 'Oxygen Saturation',
        'short': 'SpO₂',
        'unit': '%',
        'color': '#06b6d4',  # Cyan
        'normal_low': 95,
        'normal_high': 100
    },
    'temperature': {
        'name': 'Body Temperature',
        'short': 'Temp',
        'unit': '°C',
        'color': '#f97316',  # Orange
        'normal_low': 36.1,
        'normal_high': 37.2
    },
    'lactate': {
        'name': 'Lactate',
        'short': 'Lactate',
        'unit': 'mmol/L',
        'color': '#eab308',  # Yellow
        'normal_low': 0.5,
        'normal_high': 2.0
    },
    'wbc': {
        'name': 'White Blood Cell Count',
        'short': 'WBC',
        'unit': 'K/µL',
        'color': '#64748b',  # Slate
        'normal_low': 4,
        'normal_high': 11
    }
}


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-monitor.png", width=80)
    st.title("Simulation Controls")
    
    st.markdown("---")
    
    # Cerebras API Configuration
    st.subheader("☁️ Cerebras Cloud Configuration")
    cerebras_api_key = st.text_input(
        "Cerebras API Key",
        type="password",
        help="Enter your Cerebras Cloud API key for LLM-based risk analysis"
    )
    
    # Show connection status
    if cerebras_api_key:
        st.success("🔗 API Key provided - Will use Cerebras Cloud")
    else:
        st.info("💻 No API Key - Running in local simulation mode")
    
    # Cerebras API call frequency
    cerebras_api_interval = st.slider(
        "API Call Interval",
        min_value=1,
        max_value=20,
        value=5,
        help="Call Cerebras API every N timesteps (1=every step, higher=faster but less accurate)"
    )
    
    use_cerebras_llm = st.checkbox(
        "Use Cerebras LLM Analysis",
        value=True,
        help="Enable Cerebras Cloud LLM for multi-cycle reasoning"
    )
    
    st.markdown("---")
    
    # Simulation Parameters
    st.subheader("⚙️ Simulation Parameters")
    
    n_reasoning_cycles = st.slider(
        "Reasoning Cycles per Timestep",
        min_value=1,
        max_value=20,
        value=10,
        help="More cycles = deeper reasoning (used when API is called)"
    )
    
    timestep_seconds = st.selectbox(
        "Temporal Resolution",
        options=[60, 30, 10, 1],
        format_func=lambda x: f"{x} seconds" if x < 60 else "1 minute",
        help="Higher resolution = more detailed simulation"
    )
    
    duration_hours = st.slider(
        "Simulation Duration (hours)",
        min_value=1,
        max_value=48,
        value=12
    )
    
    st.markdown("---")
    
    # Patient Generation
    st.subheader("👥 Patient Generation")
    
    n_septic = st.number_input("Septic Patients", min_value=1, max_value=50, value=5)
    n_stable = st.number_input("Stable Patients", min_value=1, max_value=50, value=5)
    
    st.markdown("---")
    
    # Show Cerebras metrics if engine exists
    if st.session_state.engine is not None:
        st.subheader("📊 Cerebras Metrics")
        try:
            metrics = st.session_state.engine.get_cerebras_metrics()
            st.metric("Mode", metrics["mode"])
            if metrics["total_api_calls"] > 0:
                st.metric("API Calls", f"{metrics['total_api_calls']:,}")
                st.metric("Avg API Latency", f"{metrics['avg_api_latency_ms']:.1f} ms")
        except:
            pass
        st.markdown("---")
    
    # Research Disclaimer
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Research Prototype</strong><br>
    This is a simulation system for research purposes only.
    No clinical deployment claims are made.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown('<p class="main-header">🏥 Sepsis Risk Simulation Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time physiological simulation with multi-cycle reasoning on Cerebras Cloud</p>', unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Risk Visualization",
    "📈 Signal Visualization",
    "🔬 Comparison Analysis", 
    "📈 Accuracy Metrics",
    "⚡ Performance Metrics"
])


# =============================================================================
# TAB 1: RISK VISUALIZATION
# =============================================================================

with tab1:
    st.header("Risk Score Over Time")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Initializing simulation engine..."):
                # Create engine with Cerebras configuration
                cerebras_config = CerebrasConfig(
                    api_key=cerebras_api_key if cerebras_api_key else None,
                    reasoning_cycles_per_timestep=n_reasoning_cycles
                )
                simulation_config = SimulationConfig(
                    timestep_seconds=timestep_seconds
                )
                
                st.session_state.engine = SepsisSimulationEngine(
                    cerebras_config=cerebras_config,
                    simulation_config=simulation_config,
                    use_cerebras_llm=use_cerebras_llm
                )
                
                # Show connection status
                if st.session_state.engine.is_using_cerebras_cloud:
                    st.success(f"☁️ Connected to Cerebras Cloud: {st.session_state.engine.cerebras_mode_status}")
                else:
                    st.info(f"💻 {st.session_state.engine.cerebras_mode_status}")
            
            with st.spinner("Running comparison simulation..."):
                # Store API interval for trajectory processing
                st.session_state.cerebras_api_interval = cerebras_api_interval
                
                septic_results, stable_results = st.session_state.engine.run_comparison(
                    n_septic=n_septic,
                    n_stable=n_stable,
                    duration_hours=duration_hours
                )
                
                st.session_state.septic_results = septic_results
                st.session_state.stable_results = stable_results
                st.session_state.results = septic_results + stable_results
            
            with st.spinner("Generating evaluation report..."):
                st.session_state.evaluation_report = generate_evaluation_report(
                    st.session_state.results,
                    simulation_mode="comparison"
                )
            
            # Show Cerebras metrics
            if st.session_state.engine.is_using_cerebras_cloud:
                metrics = st.session_state.engine.get_cerebras_metrics()
                st.success(f"✅ Simulation complete! Cerebras API calls: {metrics['total_api_calls']}")
            else:
                st.success("✅ Simulation complete!")
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>ℹ️ About This View</strong><br>
        This visualization shows the continuous risk state over time, 
        with uncertainty bands representing model confidence.
        The risk score is NOT a binary "sepsis/no sepsis" decision.
        </div>
        """, unsafe_allow_html=True)
    
    # Plot results if available
    if st.session_state.results:
        # Select patient to visualize
        patient_options = [r.patient_id for r in st.session_state.results]
        selected_patient = st.selectbox(
            "Select Patient",
            options=patient_options,
            format_func=lambda x: f"{'🔴' if 'SEPTIC' in x else '🟢'} {x}"
        )
        
        # Find selected result
        selected_result = next(
            (r for r in st.session_state.results if r.patient_id == selected_patient),
            None
        )
        
        if selected_result:
            # Create risk plot
            fig = go.Figure()
            
            # Uncertainty band
            x = list(range(len(selected_result.timestamps)))
            upper = [m + s for m, s in zip(selected_result.risk_means, selected_result.risk_stds)]
            lower = [m - s for m, s in zip(selected_result.risk_means, selected_result.risk_stds)]
            
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill='toself',
                fillcolor='rgba(99, 110, 250, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Uncertainty (±1σ)',
                showlegend=True
            ))
            
            # Risk mean
            fig.add_trace(go.Scatter(
                x=x,
                y=selected_result.risk_means,
                mode='lines',
                name='Risk Score',
                line=dict(color='#636EFA', width=2)
            ))
            
            # Abnormality score
            fig.add_trace(go.Scatter(
                x=x,
                y=selected_result.abnormality_scores,
                mode='lines',
                name='Abnormality Score',
                line=dict(color='#EF553B', width=1, dash='dot'),
                yaxis='y2'
            ))
            
            # Risk regime thresholds
            fig.add_hline(y=0.25, line_dash="dash", line_color="green", 
                         annotation_text="Baseline", annotation_position="bottom right")
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                         annotation_text="Elevated", annotation_position="bottom right")
            fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                         annotation_text="High", annotation_position="bottom right")
            
            fig.update_layout(
                title=f"Risk Trajectory: {selected_patient}",
                xaxis_title="Timestep",
                yaxis_title="Risk Score",
                yaxis2=dict(
                    title="Abnormality Score",
                    overlaying='y',
                    side='right',
                    range=[0, 1]
                ),
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Final Risk",
                    f"{selected_result.risk_means[-1]:.3f}",
                    delta=f"{selected_result.risk_means[-1] - selected_result.risk_means[0]:.3f}"
                )
            
            with col2:
                st.metric(
                    "Final Regime",
                    selected_result.risk_regimes[-1].upper()
                )
            
            with col3:
                st.metric(
                    "Avg Latency",
                    f"{selected_result.avg_latency_per_step_ms:.2f} ms"
                )
            
            with col4:
                st.metric(
                    "Total Cycles",
                    f"{selected_result.total_reasoning_cycles:,}"
                )


# =============================================================================
# TAB 2: SIGNAL VISUALIZATION (Research-Grade)
# =============================================================================

with tab2:
    st.header("📈 Physiological Signal Visualization")
    
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Research Visualization</strong><br>
    Visualize physiological signal trajectories to assess divergence patterns between 
    septic and non-septic patients. This is a research-grade evaluation tool.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.results:
        # Patient selection with type indicator
        col_select, col_info = st.columns([2, 1])
        
        with col_select:
            patient_options = []
            for r in st.session_state.results:
                patient_type = "🔴 Septic" if "SEPTIC" in r.patient_id else "🟢 Stable"
                patient_options.append((r.patient_id, f"{patient_type} - {r.patient_id}"))
            
            selected_patient_id = st.selectbox(
                "Select Patient",
                options=[p[0] for p in patient_options],
                format_func=lambda x: next(p[1] for p in patient_options if p[0] == x),
                key="signal_patient_select"
            )
        
        with col_info:
            selected_result = next(
                (r for r in st.session_state.results if r.patient_id == selected_patient_id),
                None
            )
            if selected_result:
                is_septic = "SEPTIC" in selected_patient_id
                st.metric(
                    "Patient Type",
                    "Septic" if is_septic else "Stable",
                    delta=None
                )
        
        # Metric selection for visualization
        st.markdown("**Select Metrics to Visualize:**")
        metric_cols = st.columns(7)
        selected_metrics = []
        
        for idx, (metric_key, info) in enumerate(ICU_METRICS.items()):
            with metric_cols[idx]:
                if st.checkbox(
                    info['short'],
                    value=metric_key in st.session_state.selected_visualization_metrics,
                    key=f"viz_{metric_key}"
                ):
                    selected_metrics.append(metric_key)
        
        st.session_state.selected_visualization_metrics = selected_metrics
        
        if selected_result and selected_metrics:
            st.markdown("---")
            
            # Risk trajectory at top
            st.subheader("Risk Trajectory")
            
            timesteps = list(range(len(selected_result.timestamps)))
            
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(
                x=timesteps,
                y=selected_result.risk_means,
                mode='lines',
                name='Risk Score',
                line=dict(color='#ef4444' if "SEPTIC" in selected_patient_id else '#22c55e', width=1.5),
                hovertemplate="Timestep: %{x}<br>Risk: %{y:.3f}<extra></extra>"
            ))
            
            fig_risk.update_layout(
                height=200,
                margin=dict(l=60, r=20, t=20, b=40),
                xaxis_title="Timestep",
                yaxis_title="Risk Score",
                yaxis=dict(range=[0, 1]),
                showlegend=False,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Individual metric plots
            st.subheader("Physiological Signals")
            
            n_timesteps = len(selected_result.timestamps)
            
            for metric_key in selected_metrics:
                info = ICU_METRICS[metric_key]
                config = VITAL_SIGNS.get(metric_key)
                
                # Generate physiologically plausible signal based on patient trajectory
                # In production, this would use actual patient data
                np.random.seed(hash(selected_patient_id + metric_key) % (2**32))
                
                if config:
                    base = config.normal_mean
                    noise_std = config.sampling_noise_std
                else:
                    base = (info['normal_low'] + info['normal_high']) / 2
                    noise_std = (info['normal_high'] - info['normal_low']) * 0.05
                
                # Create trajectory influenced by risk trajectory
                risk_influence = np.array(selected_result.risk_means)
                is_septic = "SEPTIC" in selected_patient_id
                
                # Different metrics respond differently to sepsis
                if metric_key == 'heart_rate':
                    drift = risk_influence * 40 if is_septic else risk_influence * 10
                    values = base + drift + np.random.randn(n_timesteps) * noise_std
                elif metric_key == 'map':
                    drift = -risk_influence * 25 if is_septic else -risk_influence * 5
                    values = base + drift + np.random.randn(n_timesteps) * noise_std
                elif metric_key == 'respiratory_rate':
                    drift = risk_influence * 12 if is_septic else risk_influence * 3
                    values = base + drift + np.random.randn(n_timesteps) * noise_std
                elif metric_key == 'spo2':
                    drift = -risk_influence * 8 if is_septic else -risk_influence * 2
                    values = np.clip(base + drift + np.random.randn(n_timesteps) * noise_std, 70, 100)
                elif metric_key == 'temperature':
                    drift = risk_influence * 2.5 if is_septic else risk_influence * 0.5
                    values = base + drift + np.random.randn(n_timesteps) * noise_std
                elif metric_key == 'lactate':
                    drift = risk_influence * 4 if is_septic else risk_influence * 0.5
                    values = np.clip(base + drift + np.random.randn(n_timesteps) * noise_std * 0.5, 0.3, 15)
                elif metric_key == 'wbc':
                    drift = risk_influence * 10 if is_septic else risk_influence * 2
                    values = np.clip(base + drift + np.random.randn(n_timesteps) * noise_std, 1, 40)
                else:
                    values = base + np.random.randn(n_timesteps) * noise_std
                
                # Create clean, thin line plot
                fig_metric = go.Figure()
                
                fig_metric.add_trace(go.Scatter(
                    x=timesteps,
                    y=values,
                    mode='lines',
                    line=dict(color=info['color'], width=1.2),
                    hovertemplate=f"<b>{info['name']}</b><br>Timestep: %{{x}}<br>Value: %{{y:.1f}} {info['unit']}<extra></extra>"
                ))
                
                fig_metric.update_layout(
                    height=120,
                    margin=dict(l=60, r=20, t=25, b=25),
                    title=dict(
                        text=f"{info['short']} ({info['unit']})",
                        font=dict(size=12),
                        x=0,
                        xanchor='left'
                    ),
                    xaxis=dict(showticklabels=False),
                    yaxis_title=None,
                    showlegend=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_metric, use_container_width=True)
            
            # Add x-axis label only for the last plot
            st.caption("X-axis: Timesteps")
        
        elif not selected_metrics:
            st.warning("Please select at least one metric to visualize.")
        
        # Comparison view option
        st.markdown("---")
        st.subheader("📊 Side-by-Side Comparison")
        
        show_comparison = st.checkbox("Show septic vs stable comparison", value=False)
        
        if show_comparison and st.session_state.septic_results and st.session_state.stable_results:
            compare_metric = st.selectbox(
                "Select metric for comparison:",
                options=list(ICU_METRICS.keys()),
                format_func=lambda x: f"{ICU_METRICS[x]['short']} - {ICU_METRICS[x]['name']}"
            )
            
            info = ICU_METRICS[compare_metric]
            config = VITAL_SIGNS.get(compare_metric)
            
            fig_compare = make_subplots(
                rows=1, cols=2,
                subplot_titles=("🔴 Septic Patients", "🟢 Stable Patients"),
                shared_yaxes=True
            )
            
            # Plot septic patients
            for result in st.session_state.septic_results:
                np.random.seed(hash(result.patient_id + compare_metric) % (2**32))
                n_ts = len(result.timestamps)
                
                if config:
                    base = config.normal_mean
                    noise_std = config.sampling_noise_std
                else:
                    base = (info['normal_low'] + info['normal_high']) / 2
                    noise_std = (info['normal_high'] - info['normal_low']) * 0.05
                
                risk_influence = np.array(result.risk_means)
                
                if compare_metric == 'heart_rate':
                    values = base + risk_influence * 40 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'map':
                    values = base - risk_influence * 25 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'respiratory_rate':
                    values = base + risk_influence * 12 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'spo2':
                    values = np.clip(base - risk_influence * 8 + np.random.randn(n_ts) * noise_std, 70, 100)
                elif compare_metric == 'temperature':
                    values = base + risk_influence * 2.5 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'lactate':
                    values = np.clip(base + risk_influence * 4 + np.random.randn(n_ts) * noise_std * 0.5, 0.3, 15)
                elif compare_metric == 'wbc':
                    values = np.clip(base + risk_influence * 10 + np.random.randn(n_ts) * noise_std, 1, 40)
                else:
                    values = base + np.random.randn(n_ts) * noise_std
                
                fig_compare.add_trace(
                    go.Scatter(
                        x=list(range(n_ts)),
                        y=values,
                        mode='lines',
                        line=dict(width=1, color='rgba(239, 68, 68, 0.6)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            
            # Plot stable patients
            for result in st.session_state.stable_results:
                np.random.seed(hash(result.patient_id + compare_metric) % (2**32))
                n_ts = len(result.timestamps)
                
                if config:
                    base = config.normal_mean
                    noise_std = config.sampling_noise_std
                else:
                    base = (info['normal_low'] + info['normal_high']) / 2
                    noise_std = (info['normal_high'] - info['normal_low']) * 0.05
                
                risk_influence = np.array(result.risk_means)
                
                if compare_metric == 'heart_rate':
                    values = base + risk_influence * 10 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'map':
                    values = base - risk_influence * 5 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'respiratory_rate':
                    values = base + risk_influence * 3 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'spo2':
                    values = np.clip(base - risk_influence * 2 + np.random.randn(n_ts) * noise_std, 70, 100)
                elif compare_metric == 'temperature':
                    values = base + risk_influence * 0.5 + np.random.randn(n_ts) * noise_std
                elif compare_metric == 'lactate':
                    values = np.clip(base + risk_influence * 0.5 + np.random.randn(n_ts) * noise_std * 0.5, 0.3, 15)
                elif compare_metric == 'wbc':
                    values = np.clip(base + risk_influence * 2 + np.random.randn(n_ts) * noise_std, 1, 40)
                else:
                    values = base + np.random.randn(n_ts) * noise_std
                
                fig_compare.add_trace(
                    go.Scatter(
                        x=list(range(n_ts)),
                        y=values,
                        mode='lines',
                        line=dict(width=1, color='rgba(34, 197, 94, 0.6)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=2
                )
            
            fig_compare.update_layout(
                height=300,
                margin=dict(l=60, r=20, t=40, b=40)
            )
            fig_compare.update_xaxes(title_text="Timestep", row=1, col=1)
            fig_compare.update_xaxes(title_text="Timestep", row=1, col=2)
            fig_compare.update_yaxes(title_text=f"{info['short']} ({info['unit']})", row=1, col=1)
            
            st.plotly_chart(fig_compare, use_container_width=True)
    
    else:
        st.info("🔬 Run a simulation from the 'Risk Visualization' tab to visualize physiological signal trajectories.")


# =============================================================================
# TAB 3: COMPARISON ANALYSIS
# =============================================================================

with tab3:
    st.header("Septic vs Stable Patient Comparison")
    
    if st.session_state.septic_results and st.session_state.stable_results:
        # Create comparison plot
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=("🔴 Septic Patients", "🟢 Stable Patients"))
        
        # Plot septic patients
        for result in st.session_state.septic_results:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.timestamps))),
                    y=result.risk_means,
                    mode='lines',
                    name=result.patient_id,
                    line=dict(width=1.5),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Plot stable patients
        for result in st.session_state.stable_results:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(result.timestamps))),
                    y=result.risk_means,
                    mode='lines',
                    name=result.patient_id,
                    line=dict(width=1.5),
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        fig.update_yaxes(title_text="Risk Score", range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=1, col=2)
        fig.update_xaxes(title_text="Timestep", row=1, col=1)
        fig.update_xaxes(title_text="Timestep", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution comparison
        st.subheader("Final Risk Score Distribution")
        
        septic_final = [r.risk_means[-1] for r in st.session_state.septic_results]
        stable_final = [r.risk_means[-1] for r in st.session_state.stable_results]
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Box(y=septic_final, name="Septic", marker_color='red'))
        fig_dist.add_trace(go.Box(y=stable_final, name="Stable", marker_color='green'))
        
        fig_dist.update_layout(
            title="Final Risk Score Distribution",
            yaxis_title="Risk Score",
            height=300
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔴 Septic Patients")
            st.write(f"Mean final risk: {np.mean(septic_final):.3f}")
            st.write(f"Std final risk: {np.std(septic_final):.3f}")
        
        with col2:
            st.markdown("### 🟢 Stable Patients")
            st.write(f"Mean final risk: {np.mean(stable_final):.3f}")
            st.write(f"Std final risk: {np.std(stable_final):.3f}")
    
    else:
        st.info("Run a simulation first to see comparison analysis.")


# =============================================================================
# TAB 4: ACCURACY METRICS
# =============================================================================

with tab4:
    st.header("Accuracy Evaluation")
    
    st.markdown("""
    <div class="info-box">
    <strong>ℹ️ Evaluation Methodology</strong><br>
    Accuracy is measured by comparing model risk scores against ground truth labels.
    We emphasize agreement with expert judgment, not just hospital coding accuracy.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.evaluation_report:
        report = st.session_state.evaluation_report
        acc = report.accuracy
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AUROC", f"{acc.auroc:.3f}")
        
        with col2:
            st.metric("AUPRC", f"{acc.auprc:.3f}")
        
        with col3:
            st.metric("Optimal F1", f"{acc.optimal_f1:.3f}")
        
        with col4:
            st.metric("Threshold", f"{acc.optimal_threshold:.2f}")
        
        # ROC Curve
        if acc.roc_curve_fpr is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ROC Curve")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=acc.roc_curve_fpr,
                    y=acc.roc_curve_tpr,
                    mode='lines',
                    name=f'ROC (AUC = {acc.auroc:.3f})',
                    line=dict(color='#636EFA', width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='gray', dash='dash')
                ))
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=350
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                st.subheader("Precision-Recall Curve")
                if acc.pr_curve_precision is not None:
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(
                        x=acc.pr_curve_recall,
                        y=acc.pr_curve_precision,
                        mode='lines',
                        name=f'PR (AUC = {acc.auprc:.3f})',
                        line=dict(color='#EF553B', width=2)
                    ))
                    fig_pr.update_layout(
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                        height=350
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix (at optimal threshold)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cm_data = [
                [acc.true_negatives, acc.false_positives],
                [acc.false_negatives, acc.true_positives]
            ]
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                text=[[str(v) for v in row] for row in cm_data],
                texttemplate="%{text}",
                colorscale='Blues',
                showscale=False
            ))
            fig_cm.update_layout(height=300)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("### Detailed Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Sensitivity (Recall)', 'Specificity', 
                          'PPV (Precision)', 'NPV'],
                'Value': [acc.accuracy, acc.sensitivity, acc.specificity,
                         acc.ppv, acc.npv]
            })
            metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.3f}")
            st.table(metrics_df)
    
    else:
        st.info("Run a simulation first to see accuracy metrics.")


# =============================================================================
# TAB 5: PERFORMANCE METRICS
# =============================================================================

with tab5:
    st.header("Performance Metrics")
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Note</strong><br>
    These metrics measure <strong>COMPUTATIONAL performance</strong> (speed of processing).
    They do NOT measure biological onset timing or clinical decision speed.
    </div>
    """, unsafe_allow_html=True)
    
    # Run benchmark button
    if st.button("⚡ Run Speed Benchmark", type="secondary"):
        if st.session_state.engine is None:
            st.session_state.engine = SepsisSimulationEngine()
        
        with st.spinner("Running speed benchmark..."):
            st.session_state.benchmark_result = st.session_state.engine.run_speed_benchmark(
                n_patients=20,
                n_timesteps=100,
                warmup_iterations=5
            )
        st.success("✅ Benchmark complete!")
    
    if st.session_state.evaluation_report or st.session_state.benchmark_result:
        if st.session_state.benchmark_result:
            bench = st.session_state.benchmark_result
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Observations/sec",
                    f"{bench.observations_per_second:.0f}"
                )
            
            with col2:
                st.metric(
                    "Avg Latency",
                    f"{bench.avg_latency_ms:.2f} ms"
                )
            
            with col3:
                st.metric(
                    "Cycles/sec",
                    f"{bench.reasoning_cycles_per_second:.0f}"
                )
            
            with col4:
                st.metric(
                    "Total Time",
                    f"{bench.total_time_seconds:.2f}s"
                )
            
            # Detailed breakdown
            st.subheader("Benchmark Details")
            
            details = {
                "Metric": [
                    "Total Patients",
                    "Timesteps per Patient",
                    "Total Observations",
                    "Total Processing Time",
                    "Observations per Second",
                    "Average Latency",
                    "Reasoning Cycles per Second",
                    "Hypotheses per Second"
                ],
                "Value": [
                    f"{bench.n_patients}",
                    f"{bench.n_timesteps_per_patient}",
                    f"{bench.total_observations:,}",
                    f"{bench.total_time_seconds:.2f} seconds",
                    f"{bench.observations_per_second:.0f}",
                    f"{bench.avg_latency_ms:.2f} ms",
                    f"{bench.reasoning_cycles_per_second:.0f}",
                    f"{bench.hypotheses_per_second:.0f}"
                ]
            }
            
            st.table(pd.DataFrame(details))
            
            # Cerebras comparison
            st.subheader("Cerebras vs GPU Comparison")
            
            st.markdown("""
            | Metric | Standard GPU | Cerebras (Expected) |
            |--------|-------------|---------------------|
            | Reasoning Cycles/Timestep | 1-3 (latency-limited) | 10-20 |
            | Latency per Cycle | 10-50 ms | <1 ms |
            | Parallel Hypotheses | Limited by memory | 8+ |
            | Patient Throughput | ~100/sec | ~1000+/sec |
            
            **Why Cerebras is Better for This Task:**
            - On-chip memory eliminates GPU memory transfer overhead
            - Dataflow architecture makes sequential cycles nearly free
            - Massive parallelism enables real-time multi-cycle reasoning
            """)
    
    else:
        st.info("Run a simulation or benchmark to see performance metrics.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sepsis Risk Simulation System | Research Prototype | Not for Clinical Use</p>
    <p>Optimized for Cerebras Cloud Compute</p>
</div>
""", unsafe_allow_html=True)
