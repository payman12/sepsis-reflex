"""
Sepsis Risk Simulation System - Main Entry Point
=================================================

This is the main entry point for the sepsis risk simulation system.
It provides a command-line interface for running simulations, benchmarks,
and evaluations.

Usage:
    # Run comparison simulation
    python main.py --mode comparison --n-septic 10 --n-stable 10

    # Run speed benchmark
    python main.py --mode benchmark --n-patients 100
    
    # Run with Cerebras API
    python main.py --mode comparison --cerebras-key YOUR_API_KEY
    
    # Launch dashboard
    python main.py --mode dashboard
    
    # Generate synthetic dataset
    python main.py --mode generate --output synthetic_data.csv

Research Prototype Disclaimer:
-----------------------------
This is a simulation system for research purposes only.
No clinical deployment claims are made.
Speed metrics measure COMPUTATIONAL performance, not biological timing.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .config import CerebrasConfig, SimulationConfig, EvaluationConfig
    from .synthetic_data_generator import SyntheticDataGenerator, generate_benchmark_dataset
    from .simulation_engine import SepsisSimulationEngine
    from .evaluation import generate_evaluation_report, AccuracyEvaluator, SpeedEvaluator
except ImportError:
    from config import CerebrasConfig, SimulationConfig, EvaluationConfig
    from synthetic_data_generator import SyntheticDataGenerator, generate_benchmark_dataset
    from simulation_engine import SepsisSimulationEngine
    from evaluation import generate_evaluation_report, AccuracyEvaluator, SpeedEvaluator


def run_comparison(args):
    """Run comparison simulation between septic and stable patients."""
    print("=" * 70)
    print("SEPSIS RISK SIMULATION - COMPARISON MODE")
    print("=" * 70)
    print()
    
    # Configure
    cerebras_config = CerebrasConfig(
        api_key=args.cerebras_key,
        reasoning_cycles_per_timestep=args.n_cycles
    )
    simulation_config = SimulationConfig(
        timestep_seconds=args.timestep_seconds
    )
    
    # Initialize engine
    print("Initializing simulation engine...")
    engine = SepsisSimulationEngine(
        cerebras_config=cerebras_config,
        simulation_config=simulation_config
    )
    
    # Run comparison
    print(f"\nRunning comparison simulation:")
    print(f"  Septic patients: {args.n_septic}")
    print(f"  Stable patients: {args.n_stable}")
    print(f"  Duration: {args.duration_hours} hours")
    print(f"  Reasoning cycles: {args.n_cycles}")
    print()
    
    septic_results, stable_results = engine.run_comparison(
        n_septic=args.n_septic,
        n_stable=args.n_stable,
        duration_hours=args.duration_hours
    )
    
    # Generate evaluation report
    print("\n" + "-" * 50)
    print("EVALUATION RESULTS")
    print("-" * 50)
    
    all_results = septic_results + stable_results
    report = generate_evaluation_report(all_results, simulation_mode="comparison")
    
    print(f"\nAccuracy Metrics:")
    print(f"  AUROC: {report.accuracy.auroc:.3f}")
    print(f"  AUPRC: {report.accuracy.auprc:.3f}")
    print(f"  Optimal F1: {report.accuracy.optimal_f1:.3f} @ threshold {report.accuracy.optimal_threshold:.2f}")
    print(f"  Sensitivity: {report.accuracy.sensitivity:.3f}")
    print(f"  Specificity: {report.accuracy.specificity:.3f}")
    
    print(f"\nSpeed Metrics:")
    print(f"  Observations processed: {report.n_observations}")
    print(f"  Total patients: {report.n_patients}")
    
    # Summary by group
    print("\n" + "-" * 50)
    print("GROUP SUMMARY")
    print("-" * 50)
    
    septic_final = [r.risk_means[-1] for r in septic_results]
    stable_final = [r.risk_means[-1] for r in stable_results]
    
    import numpy as np
    print(f"\nSeptic patients (n={len(septic_results)}):")
    print(f"  Mean final risk: {np.mean(septic_final):.3f} ± {np.std(septic_final):.3f}")
    print(f"  Range: [{min(septic_final):.3f}, {max(septic_final):.3f}]")
    
    print(f"\nStable patients (n={len(stable_results)}):")
    print(f"  Mean final risk: {np.mean(stable_final):.3f} ± {np.std(stable_final):.3f}")
    print(f"  Range: [{min(stable_final):.3f}, {max(stable_final):.3f}]")
    
    # Save results if requested
    if args.output:
        results_df = engine.get_results_dataframe()
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    return report


def run_benchmark(args):
    """Run speed benchmark."""
    print("=" * 70)
    print("SEPSIS RISK SIMULATION - SPEED BENCHMARK")
    print("=" * 70)
    print()
    print("NOTE: These metrics measure COMPUTATIONAL performance,")
    print("      NOT biological onset timing or clinical decision speed.")
    print()
    
    # Configure
    cerebras_config = CerebrasConfig(
        api_key=args.cerebras_key,
        reasoning_cycles_per_timestep=args.n_cycles
    )
    
    # Initialize engine
    print("Initializing simulation engine...")
    engine = SepsisSimulationEngine(cerebras_config=cerebras_config)
    
    # Run benchmark
    print(f"\nRunning speed benchmark:")
    print(f"  Patients: {args.n_patients}")
    print(f"  Timesteps per patient: {args.n_timesteps}")
    print(f"  Reasoning cycles: {args.n_cycles}")
    print()
    
    benchmark = engine.run_speed_benchmark(
        n_patients=args.n_patients,
        n_timesteps=args.n_timesteps,
        warmup_iterations=args.warmup
    )
    
    print("\n" + "-" * 50)
    print("BENCHMARK RESULTS")
    print("-" * 50)
    print(f"\nTotal observations: {benchmark.total_observations:,}")
    print(f"Total time: {benchmark.total_time_seconds:.2f} seconds")
    print(f"\nThroughput:")
    print(f"  Observations/second: {benchmark.observations_per_second:.0f}")
    print(f"  Reasoning cycles/second: {benchmark.reasoning_cycles_per_second:.0f}")
    print(f"  Hypotheses/second: {benchmark.hypotheses_per_second:.0f}")
    print(f"\nLatency:")
    print(f"  Average: {benchmark.avg_latency_ms:.2f} ms")
    
    # Save benchmark results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(benchmark.to_dict(), f, indent=2)
        print(f"\nBenchmark results saved to: {args.output}")
    
    return benchmark


def generate_data(args):
    """Generate synthetic dataset."""
    print("=" * 70)
    print("SEPSIS RISK SIMULATION - DATA GENERATION")
    print("=" * 70)
    print()
    
    output_path = args.output or "synthetic_sepsis_data.csv"
    
    print(f"Generating synthetic dataset:")
    print(f"  Patients: {args.n_patients}")
    print(f"  Duration: {args.duration_hours} hours")
    print(f"  Timestep: {args.timestep_seconds} seconds")
    print(f"  Output: {output_path}")
    print()
    
    df = generate_benchmark_dataset(
        n_patients=args.n_patients,
        duration_hours=args.duration_hours,
        timestep_seconds=args.timestep_seconds,
        output_path=output_path,
        random_seed=args.seed
    )
    
    print(f"\nGenerated {len(df)} observations")
    print(f"Columns: {list(df.columns)}")
    
    # Show summary statistics
    print("\nSample statistics:")
    print(df.describe())
    
    return df


def run_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess
    
    print("=" * 70)
    print("SEPSIS RISK SIMULATION - LAUNCHING DASHBOARD")
    print("=" * 70)
    print()
    print("Starting Streamlit server...")
    print("Press Ctrl+C to stop")
    print()
    
    dashboard_path = Path(__file__).parent / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def main():
    parser = argparse.ArgumentParser(
        description="Sepsis Risk Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comparison simulation
  python main.py --mode comparison --n-septic 10 --n-stable 10
  
  # Run speed benchmark
  python main.py --mode benchmark --n-patients 100
  
  # Generate synthetic data
  python main.py --mode generate --n-patients 50 --output data.csv
  
  # Launch dashboard
  python main.py --mode dashboard

Research Prototype - Not for Clinical Use
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode", "-m",
        choices=["comparison", "benchmark", "generate", "dashboard"],
        default="comparison",
        help="Simulation mode"
    )
    
    # Cerebras configuration
    parser.add_argument(
        "--cerebras-key",
        type=str,
        default=None,
        help="Cerebras Cloud API key (optional)"
    )
    
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=10,
        help="Reasoning cycles per timestep (default: 10)"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--n-septic",
        type=int,
        default=5,
        help="Number of septic patients (comparison mode)"
    )
    
    parser.add_argument(
        "--n-stable",
        type=int,
        default=5,
        help="Number of stable patients (comparison mode)"
    )
    
    parser.add_argument(
        "--n-patients",
        type=int,
        default=50,
        help="Number of patients (benchmark/generate mode)"
    )
    
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=100,
        help="Timesteps per patient (benchmark mode)"
    )
    
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=12,
        help="Simulation duration in hours"
    )
    
    parser.add_argument(
        "--timestep-seconds",
        type=int,
        default=60,
        help="Timestep resolution in seconds"
    )
    
    # Benchmark parameters
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations for benchmark"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Print header
    print()
    print("=" * 70)
    print("           SEPSIS RISK SIMULATION SYSTEM")
    print("           Optimized for Cerebras Cloud Compute")
    print("           Research Prototype - Not for Clinical Use")
    print("=" * 70)
    print()
    
    # Run selected mode
    if args.mode == "comparison":
        run_comparison(args)
    elif args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "generate":
        generate_data(args)
    elif args.mode == "dashboard":
        run_dashboard(args)
    
    print()
    print("=" * 70)
    print("Complete.")


if __name__ == "__main__":
    main()
