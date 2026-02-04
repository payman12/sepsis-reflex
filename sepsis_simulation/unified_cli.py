"""
Unified CLI for Sepsis Simulation System
==========================================

This is the main command-line interface that provides access to all
system components:

1. generate    - Generate synthetic patient data
2. train       - Train transformer model on synthetic data
3. compare     - Run Expert vs AI comparison
4. simulate    - Run real-time simulation
5. dashboard   - Launch Streamlit dashboard
6. benchmark   - Run comprehensive benchmarks
7. cerebras    - Run analysis using Cerebras Cloud API

Usage:
    python unified_cli.py generate --n_patients 20
    python unified_cli.py train --n_patients 500 --epochs 15
    python unified_cli.py compare --n_patients 20
    python unified_cli.py simulate
    python unified_cli.py dashboard
    python unified_cli.py cerebras --api_key YOUR_KEY --n_patients 10

Cerebras Cloud Integration:
---------------------------
The 'cerebras' command runs the full analysis pipeline on Cerebras Cloud:
- Multi-cycle reasoning (5-20 cycles per patient per timestep)
- LLM-powered risk assessment with clinical reasoning
- Parallel patient processing for high throughput
- Real-time inference with sub-100ms latency
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def cmd_generate(args):
    """Generate synthetic patient data"""
    from .enhanced_data_generator import EnhancedMIMICDataGenerator
    
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION")
    print("=" * 60)
    
    # Use None seed for different data each run
    seed = args.seed if args.seed != -1 else None
    
    generator = EnhancedMIMICDataGenerator(seed=seed)
    
    patient_data, expert_labels = generator.generate_patient_cohort(
        n_patients=args.n_patients,
        duration_hours=args.duration,
        timestep_minutes=args.timestep,
        sepsis_ratio=args.sepsis_ratio,
        include_expert_labels=True
    )
    
    # Save to files
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    patient_path = output_dir / f"patient_data_{timestamp}.csv"
    expert_path = output_dir / f"expert_labels_{timestamp}.csv"
    
    patient_data.to_csv(patient_path, index=False)
    expert_labels.to_csv(expert_path, index=False)
    
    print(f"\nGenerated {args.n_patients} patients:")
    print(f"  - {int(expert_labels['develops_sepsis'].sum())} sepsis patients")
    print(f"  - {args.n_patients - int(expert_labels['develops_sepsis'].sum())} non-sepsis patients")
    print(f"\nData saved to:")
    print(f"  - {patient_path}")
    print(f"  - {expert_path}")
    
    return patient_data, expert_labels


def cmd_train(args):
    """Train transformer model on synthetic data"""
    from .transformer_training_bridge import TransformerTrainingBridge, TrainingConfig
    
    print("\n" + "=" * 60)
    print("TRANSFORMER TRAINING")
    print("=" * 60)
    
    # Use None seed for different data each run
    seed = args.seed if args.seed != -1 else None
    
    config = TrainingConfig(
        n_patients=args.n_patients,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        seed=seed
    )
    
    bridge = TransformerTrainingBridge(config)
    results = bridge.run_full_training_pipeline()
    
    print("\nTraining complete!")
    print(f"Final validation AUROC: {results['metrics']['val_auroc']:.4f}")
    print(f"Model saved to: {results['paths']['model']}")
    
    return results


def cmd_compare(args):
    """Run Expert vs AI comparison"""
    from .expert_ai_comparison import ExpertAIComparison, ComparisonConfig
    
    print("\n" + "=" * 60)
    print("EXPERT vs AI COMPARISON")
    print("=" * 60)
    
    # Use None seed for different data each run
    seed = args.seed if args.seed != -1 else None
    
    config = ComparisonConfig(
        n_patients=args.n_patients,
        duration_hours=args.duration,
        timestep_minutes=args.timestep,
        sepsis_ratio=args.sepsis_ratio,
        seed=seed
    )
    
    comparison = ExpertAIComparison(config)
    result = comparison.run_comparison()
    
    if args.save:
        comparison.save_results()
    
    return result


def cmd_simulate(args):
    """Run real-time simulation"""
    from .main import run_comparison_demo, run_benchmark_demo
    
    print("\n" + "=" * 60)
    print("SIMULATION")
    print("=" * 60)
    
    if args.mode == "comparison":
        run_comparison_demo()
    elif args.mode == "benchmark":
        run_benchmark_demo()
    else:
        print(f"Unknown simulation mode: {args.mode}")
        print("Available modes: comparison, benchmark")


def cmd_dashboard(args):
    """Launch Streamlit dashboard"""
    import subprocess
    import os
    
    print("\n" + "=" * 60)
    print("LAUNCHING DASHBOARD")
    print("=" * 60)
    
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        return
    
    print(f"Starting Streamlit dashboard...")
    print(f"Dashboard file: {dashboard_path}")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(dashboard_path),
        "--server.headless", "true" if args.headless else "false"
    ])


def cmd_benchmark(args):
    """Run comprehensive benchmarks"""
    from .expert_ai_comparison import run_expert_ai_comparison
    from .transformer_training_bridge import train_on_synthetic_data
    import time
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK")
    print("=" * 60)
    
    results = {
        "comparison_results": [],
        "training_results": None,
        "timing": {}
    }
    
    # Run multiple comparison rounds with different random data
    print(f"\n--- Running {args.rounds} comparison rounds ---")
    
    for i in range(args.rounds):
        print(f"\nRound {i+1}/{args.rounds}")
        start = time.time()
        
        result = run_expert_ai_comparison(
            n_patients=args.n_patients,
            seed=None  # Different each round
        )
        
        elapsed = time.time() - start
        
        results["comparison_results"].append({
            "round": i+1,
            "auroc": result.auroc,
            "auprc": result.auprc,
            "accuracy": result.accuracy,
            "f1": result.f1,
            "time_seconds": elapsed
        })
    
    # Calculate aggregate metrics
    import numpy as np
    aurocs = [r["auroc"] for r in results["comparison_results"]]
    auprcs = [r["auprc"] for r in results["comparison_results"]]
    
    print(f"\n--- Aggregate Results ({args.rounds} rounds) ---")
    print(f"AUROC: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
    print(f"AUPRC: {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}")
    
    # Optionally run training
    if args.include_training:
        print(f"\n--- Running Training Benchmark ---")
        start = time.time()
        
        training_result = train_on_synthetic_data(
            n_patients=args.n_patients * 5,  # More patients for training
            epochs=args.epochs,
            seed=None
        )
        
        results["training_results"] = {
            "val_auroc": training_result["metrics"]["val_auroc"],
            "val_auprc": training_result["metrics"]["val_auprc"],
            "time_seconds": time.time() - start
        }
        
        print(f"Training AUROC: {training_result['metrics']['val_auroc']:.4f}")
    
    return results


def cmd_cerebras(args):
    """Run analysis using Cerebras Cloud API"""
    import os
    import time
    import numpy as np
    from .cerebras_inference import CerebrasRiskEngine, CerebrasTransformerInference
    from .enhanced_data_generator import EnhancedMIMICDataGenerator
    
    print("\n" + "=" * 60)
    print("CEREBRAS CLOUD ANALYSIS")
    print("=" * 60)
    
    # Get API key
    api_key = args.api_key or os.environ.get("CEREBRAS_API_KEY")
    
    if not api_key:
        print("\n[WARN] No API key provided!")
        print("Please provide your Cerebras API key via:")
        print("  --api_key YOUR_KEY")
        print("  or set CEREBRAS_API_KEY environment variable")
        return
    
    # Initialize Cerebras engine
    print(f"\nInitializing Cerebras Risk Engine...")
    print(f"  Model tier: {args.model_tier}")
    print(f"  Reasoning cycles: {args.reasoning_cycles}")
    print(f"  Parallel patients: {args.parallel}")
    
    engine = CerebrasRiskEngine(
        api_key=api_key,
        model_tier=args.model_tier,
        reasoning_cycles=args.reasoning_cycles,
        parallel_patients=args.parallel
    )
    
    # Generate synthetic data
    print(f"\nGenerating {args.n_patients} synthetic patients...")
    generator = EnhancedMIMICDataGenerator(seed=None)
    
    patient_data, expert_labels = generator.generate_patient_cohort(
        n_patients=args.n_patients,
        duration_hours=args.duration,
        timestep_minutes=args.timestep,
        sepsis_ratio=0.5,
        include_expert_labels=True
    )
    
    # Prepare patients for batch analysis
    patients = []
    patient_ids = patient_data['patient_id'].unique()
    
    for pid in patient_ids:
        patient_df = patient_data[patient_data['patient_id'] == pid].copy()
        
        # Get latest vitals
        latest = patient_df.iloc[-1]
        vitals = {}
        for col in patient_df.columns:
            if col not in ['patient_id', 'timestep', 'timestamp']:
                val = latest[col]
                if not np.isnan(val):
                    vitals[col] = val
        
        # Get history
        history = []
        for _, row in patient_df.tail(6).iterrows():
            h = {}
            for col in ['heart_rate', 'map', 'respiratory_rate', 'spo2', 'temperature']:
                if col in row and not np.isnan(row[col]):
                    h[col] = row[col]
            if h:
                history.append(h)
        
        patients.append({
            "vitals": vitals,
            "history": history,
            "info": {"patient_id": pid}
        })
    
    # Run batch analysis on Cerebras
    print(f"\nRunning Cerebras analysis on {len(patients)} patients...")
    print(f"   (Using {args.reasoning_cycles} reasoning cycles per patient)")
    
    start_time = time.time()
    results = engine.analyze_batch(patients)
    total_time = time.time() - start_time
    
    # Display results
    print("\n" + "-" * 60)
    print("ANALYSIS RESULTS")
    print("-" * 60)
    
    # Collect predictions vs ground truth
    predictions = []
    ground_truth = []
    
    for i, (pid, result) in enumerate(zip(patient_ids, results)):
        expert_row = expert_labels[expert_labels['patient_id'] == pid].iloc[0]
        actual_sepsis = expert_row['develops_sepsis']
        
        predictions.append(result.risk_score)
        ground_truth.append(actual_sepsis)
        
        status = "[SEPSIS]" if actual_sepsis else "[STABLE]"
        risk_level = "HIGH" if result.risk_score > 0.7 else "MEDIUM" if result.risk_score > 0.4 else "LOW"
        
        if args.verbose or i < 5:  # Show first 5 or all if verbose
            print(f"\nPatient {pid} ({status}):")
            print(f"  Risk Score: {result.risk_score:.3f} ({risk_level})")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Uncertainty: +/- {result.uncertainty:.3f}")
            
            # Trend Analysis (Cerebras Cycle 2)
            accel = "+" if result.trend_acceleration > 0 else ""
            print(f"  Trend: {result.trend_direction} (acceleration: {accel}{result.trend_acceleration:.2f})")
            
            # Regime Shift Detection (Cerebras Cycle 3)
            if hasattr(result, 'regime_shift_detected') and result.regime_shift_detected:
                print(f"  [!] Regime Shift: {result.regime_description}")
            elif hasattr(result, 'regime_description'):
                print(f"  Regime: {result.regime_description}")
            
            # Correlation Analysis (Cerebras Cycle 4)
            if hasattr(result, 'correlation_abnormality') and result.correlation_abnormality > 0.2:
                print(f"  Correlation Abnormality: {result.correlation_abnormality:.2f}")
            
            # Confidence Interval (Cerebras Cycle 5)
            if hasattr(result, 'confidence_interval'):
                ci_low, ci_high = result.confidence_interval
                print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
            
            if result.key_factors:
                print(f"  Key Factors: {', '.join(result.key_factors[:5])}")
    
    if not args.verbose and len(patients) > 5:
        print(f"\n  ... ({len(patients) - 5} more patients)")
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    binary_preds = (predictions > 0.5).astype(int)
    
    print("\n" + "-" * 60)
    print("ACCURACY METRICS")
    print("-" * 60)
    
    try:
        auroc = roc_auc_score(ground_truth, predictions)
        print(f"  AUROC: {auroc:.4f}")
    except:
        print(f"  AUROC: N/A (insufficient class variety)")
    
    accuracy = accuracy_score(ground_truth, binary_preds)
    f1 = f1_score(ground_truth, binary_preds, zero_division=0)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Performance metrics
    print("\n" + "-" * 60)
    print("PERFORMANCE METRICS")
    print("-" * 60)
    
    avg_latency = sum(r.metrics.latency_ms for r in results) / len(results)
    total_cycles = sum(r.metrics.reasoning_cycles for r in results)
    
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg Latency/Patient: {avg_latency:.2f}ms")
    print(f"  Patients/Second: {len(patients) / total_time:.2f}")
    print(f"  Total Reasoning Cycles: {total_cycles}")
    print(f"  Model Used: {results[0].metrics.model_used if results else 'N/A'}")
    
    # Engine metrics
    metrics = engine.get_metrics()
    print(f"\n  Cerebras Connection: {'[OK] Connected' if metrics['connected'] else '[SIMULATION MODE]'}")
    
    # Summary of what ran on Cerebras
    print("\n" + "-" * 60)
    print("WHAT RAN ON CEREBRAS")
    print("-" * 60)
    print(f"  [1] Multi-Cycle Reasoning: {args.reasoning_cycles} cycles/patient")
    print(f"      - Each patient processed with {args.reasoning_cycles} LLM reasoning cycles")
    print(f"      - Total cycles executed: {total_cycles}")
    print(f"  [2] Risk Score Calculation: Clinical knowledge-based LLM analysis")
    print(f"      - Used clinical context (SOFA, SIRS, vital thresholds)")
    print(f"  [3] Trend Analysis: Acceleration & direction detection")
    # Count patients with acceleration
    n_accel = sum(1 for r in results if abs(r.trend_acceleration) > 0.1)
    print(f"      - Patients with significant acceleration: {n_accel}/{len(results)}")
    print(f"  [4] Regime Shift Detection: Physiological state transitions")
    # Count regime shifts
    n_shifts = sum(1 for r in results if hasattr(r, 'regime_shift_detected') and r.regime_shift_detected)
    print(f"      - Regime shifts detected: {n_shifts}/{len(results)}")
    print(f"  [5] Uncertainty Quantification: Confidence intervals per patient")
    print(f"      - Average confidence: {sum(r.confidence for r in results)/len(results):.2f}")
    print(f"  [6] Parallel Processing: {args.parallel} patients simultaneously")
    print(f"      - Throughput: {len(patients) / total_time:.1f} patients/sec")
    
    return {
        "predictions": predictions.tolist(),
        "ground_truth": ground_truth.tolist(),
        "results": results,
        "metrics": {
            "accuracy": accuracy,
            "f1": f1,
            "total_time": total_time,
            "avg_latency_ms": avg_latency
        }
    }


def cmd_cloud_train(args):
    """
    Train AND run inference entirely on Cerebras Cloud.
    
    This moves BOTH training AND inference to Cerebras Cloud:
    - No local GPU/CPU needed for model training
    - No PyTorch model files to manage
    - All computation on Cerebras's 850K+ cores
    """
    import os
    import numpy as np
    import time
    
    print("\n" + "=" * 60)
    print("CEREBRAS CLOUD TRAINING & INFERENCE")
    print("=" * 60)
    print("\nThis runs EVERYTHING on Cerebras Cloud:")
    print("  [x] Training: Pattern learning via LLM")
    print("  [x] Inference: Risk prediction via LLM")
    print("  [ ] Local GPU: NOT NEEDED")
    print("=" * 60)
    
    # Check API key
    api_key = args.api_key or os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("\n[WARN] No API key provided!")
        print("Please provide your Cerebras API key via:")
        print("  --api_key YOUR_KEY")
        print("  or set CEREBRAS_API_KEY environment variable")
        return
    
    # Import Cerebras components
    from .cerebras_inference import CerebrasTransformerTraining, CerebrasTransformerInferenceCloud
    from .enhanced_data_generator import EnhancedMIMICDataGenerator
    
    # Generate training data (still local - just data generation)
    print("\n" + "-" * 60)
    print("STEP 1: Generate Training Data (Local)")
    print("-" * 60)
    
    generator = EnhancedMIMICDataGenerator(seed=None)
    train_data, train_labels = generator.generate_patient_cohort(
        n_patients=args.n_patients,
        duration_hours=48,
        timestep_minutes=60,
        sepsis_ratio=0.5,
        include_expert_labels=True
    )
    
    print(f"  Generated {args.n_patients} training patients")
    
    # Prepare data for training
    feature_cols = [col for col in train_data.columns 
                   if col not in ['patient_id', 'timestamp', 'timestep']]
    
    # Group by patient and create sequences
    X_train_list = []
    y_train_list = []
    
    for pid in train_data['patient_id'].unique():
        patient_data = train_data[train_data['patient_id'] == pid][feature_cols].values
        label = train_labels[train_labels['patient_id'] == pid]['develops_sepsis'].values[0]
        
        if len(patient_data) > 0:
            X_train_list.append(patient_data[-6:] if len(patient_data) >= 6 else patient_data)
            y_train_list.append(label)
    
    # Pad sequences to same length
    max_len = max(len(x) for x in X_train_list)
    X_train = np.array([
        np.pad(x, ((max_len - len(x), 0), (0, 0)), mode='edge') 
        for x in X_train_list
    ])
    y_train = np.array(y_train_list)
    
    print(f"  Training shape: {X_train.shape}")
    print(f"  Sepsis cases: {y_train.sum()} / {len(y_train)}")
    
    # ================================================
    # STEP 2: TRAIN ON CEREBRAS CLOUD
    # ================================================
    print("\n" + "-" * 60)
    print("STEP 2: Train on Cerebras Cloud")
    print("-" * 60)
    print("  All training computation on Cerebras (no local GPU!)")
    
    trainer = CerebrasTransformerTraining(api_key=api_key)
    
    start_train = time.perf_counter()
    training_metrics = trainer.train(
        X_train=X_train,
        y_train=y_train,
        feature_names=feature_cols,
        epochs=args.epochs,
        batch_size=20
    )
    train_time = time.perf_counter() - start_train
    
    # Save model if requested
    if args.save_model:
        from pathlib import Path
        import datetime
        
        output_dir = Path("trained_models")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"cerebras_model_{timestamp}.json"
        trainer.save_model(str(model_path))
    
    # ================================================
    # STEP 3: GENERATE TEST DATA
    # ================================================
    print("\n" + "-" * 60)
    print("STEP 3: Generate Test Data (Local)")
    print("-" * 60)
    
    test_data, test_labels = generator.generate_patient_cohort(
        n_patients=args.n_test,
        duration_hours=48,
        timestep_minutes=60,
        sepsis_ratio=0.5,
        include_expert_labels=True
    )
    
    print(f"  Generated {args.n_test} test patients")
    
    # Prepare test data
    X_test_list = []
    y_test_list = []
    
    for pid in test_data['patient_id'].unique():
        patient_data = test_data[test_data['patient_id'] == pid][feature_cols].values
        label = test_labels[test_labels['patient_id'] == pid]['develops_sepsis'].values[0]
        
        if len(patient_data) > 0:
            X_test_list.append(patient_data[-6:] if len(patient_data) >= 6 else patient_data)
            y_test_list.append(label)
    
    X_test = np.array([
        np.pad(x, ((max_len - len(x), 0), (0, 0)), mode='edge') 
        for x in X_test_list
    ])
    y_test = np.array(y_test_list)
    
    # ================================================
    # STEP 4: INFERENCE ON CEREBRAS CLOUD
    # ================================================
    print("\n" + "-" * 60)
    print("STEP 4: Run Inference on Cerebras Cloud")
    print("-" * 60)
    print("  All inference computation on Cerebras (no local GPU!)")
    
    inferencer = CerebrasTransformerInferenceCloud(
        api_key=api_key,
        learned_model=trainer.get_model()
    )
    
    start_inference = time.perf_counter()
    predictions = inferencer.predict(X_test, feature_cols)
    inference_time = time.perf_counter() - start_inference
    
    # ================================================
    # STEP 5: EVALUATE RESULTS
    # ================================================
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    
    binary_preds = (predictions > 0.5).astype(int)
    
    print("\nAccuracy Metrics:")
    try:
        auroc = roc_auc_score(y_test, predictions)
        print(f"  AUROC: {auroc:.4f}")
    except:
        print(f"  AUROC: N/A")
    
    accuracy = accuracy_score(y_test, binary_preds)
    f1 = f1_score(y_test, binary_preds, zero_division=0)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    print("\nPerformance Metrics:")
    print(f"  Training time: {train_time:.2f}s (on Cerebras)")
    print(f"  Inference time: {inference_time:.2f}s (on Cerebras)")
    print(f"  Training throughput: {args.n_patients / train_time:.1f} patients/sec")
    print(f"  Inference throughput: {args.n_test / inference_time:.1f} patients/sec")
    
    print("\n" + "-" * 60)
    print("WHAT RAN WHERE")
    print("-" * 60)
    print("  LOCAL (Your Computer):")
    print("    - Data generation (synthetic patient creation)")
    print("    - Data formatting (prepare for API)")
    print("    - Result parsing (extract from API response)")
    print("    - Metrics calculation (AUROC, F1)")
    print()
    print("  CEREBRAS CLOUD:")
    print("    - [x] Model Training (pattern learning)")
    print("    - [x] Feature Extraction (in training)")
    print("    - [x] Inference (risk prediction)")
    print("    - [x] All LLM computation")
    print()
    print("  NO LOCAL GPU USED!")
    
    return {
        "training_metrics": training_metrics,
        "inference_metrics": inferencer.get_metrics(),
        "accuracy": accuracy,
        "f1": f1,
        "auroc": auroc if 'auroc' in dir() else None
    }


def main():
    """Main entry point for unified CLI"""
    parser = argparse.ArgumentParser(
        description="Sepsis Simulation System - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 20 patients with different data each run
    python -m sepsis_simulation.unified_cli generate --n_patients 20
    
    # Train transformer with 500 patients
    python -m sepsis_simulation.unified_cli train --n_patients 500 --epochs 15
    
    # Compare expert vs AI with 30 patients
    python -m sepsis_simulation.unified_cli compare --n_patients 30
    
    # Run comprehensive benchmark
    python -m sepsis_simulation.unified_cli benchmark --rounds 5 --n_patients 20
    
    # Launch dashboard
    python -m sepsis_simulation.unified_cli dashboard
    
    # Run analysis on Cerebras Cloud (requires API key)
    python -m sepsis_simulation.unified_cli cerebras --api_key YOUR_KEY --n_patients 20
    
    # Cerebras with more reasoning cycles for higher accuracy
    python -m sepsis_simulation.unified_cli cerebras --api_key YOUR_KEY --reasoning_cycles 10 --model_tier accurate
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic patient data")
    gen_parser.add_argument("--n_patients", type=int, default=20, 
                           help="Number of patients to generate")
    gen_parser.add_argument("--duration", type=int, default=48,
                           help="Duration in hours")
    gen_parser.add_argument("--timestep", type=int, default=60,
                           help="Timestep in minutes")
    gen_parser.add_argument("--sepsis_ratio", type=float, default=0.5,
                           help="Ratio of sepsis patients")
    gen_parser.add_argument("--output_dir", type=str, default="generated_data",
                           help="Output directory")
    gen_parser.add_argument("--seed", type=int, default=-1,
                           help="Random seed (-1 for different data each run)")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train transformer model")
    train_parser.add_argument("--n_patients", type=int, default=500,
                             help="Number of patients for training")
    train_parser.add_argument("--epochs", type=int, default=15,
                             help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=32,
                             help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=0.001,
                             help="Learning rate")
    train_parser.add_argument("--window_size", type=int, default=6,
                             help="Input window size")
    train_parser.add_argument("--prediction_horizon", type=int, default=6,
                             help="Prediction horizon")
    train_parser.add_argument("--hidden_dim", type=int, default=64,
                             help="Hidden dimension")
    train_parser.add_argument("--num_layers", type=int, default=2,
                             help="Number of transformer layers")
    train_parser.add_argument("--seed", type=int, default=-1,
                             help="Random seed (-1 for different data each run)")
    
    # Compare command
    comp_parser = subparsers.add_parser("compare", help="Run Expert vs AI comparison")
    comp_parser.add_argument("--n_patients", type=int, default=20,
                            help="Number of patients to compare")
    comp_parser.add_argument("--duration", type=int, default=48,
                            help="Duration in hours")
    comp_parser.add_argument("--timestep", type=int, default=60,
                            help="Timestep in minutes")
    comp_parser.add_argument("--sepsis_ratio", type=float, default=0.5,
                            help="Ratio of sepsis patients")
    comp_parser.add_argument("--save", action="store_true",
                            help="Save results to file")
    comp_parser.add_argument("--seed", type=int, default=-1,
                            help="Random seed (-1 for different data each run)")
    
    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run real-time simulation")
    sim_parser.add_argument("--mode", type=str, default="comparison",
                           choices=["comparison", "benchmark"],
                           help="Simulation mode")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--headless", action="store_true",
                            help="Run in headless mode")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run comprehensive benchmarks")
    bench_parser.add_argument("--n_patients", type=int, default=20,
                             help="Number of patients per round")
    bench_parser.add_argument("--rounds", type=int, default=5,
                             help="Number of comparison rounds")
    bench_parser.add_argument("--include_training", action="store_true",
                             help="Include training benchmark")
    bench_parser.add_argument("--epochs", type=int, default=10,
                             help="Training epochs (if included)")
    
    # Cerebras command
    cerebras_parser = subparsers.add_parser("cerebras", 
                                            help="Run analysis using Cerebras Cloud API")
    cerebras_parser.add_argument("--api_key", type=str, default=None,
                                help="Cerebras API key (or set CEREBRAS_API_KEY env var)")
    cerebras_parser.add_argument("--n_patients", type=int, default=10,
                                help="Number of patients to analyze")
    cerebras_parser.add_argument("--model_tier", type=str, default="fast",
                                choices=["fast", "balanced", "accurate"],
                                help="Model tier (fast=low latency, accurate=best quality)")
    cerebras_parser.add_argument("--reasoning_cycles", type=int, default=5,
                                help="Number of reasoning cycles per patient (1-20)")
    cerebras_parser.add_argument("--parallel", type=int, default=10,
                                help="Max parallel patient processing")
    cerebras_parser.add_argument("--duration", type=int, default=48,
                                help="Patient data duration in hours")
    cerebras_parser.add_argument("--timestep", type=int, default=60,
                                help="Timestep in minutes")
    cerebras_parser.add_argument("--verbose", action="store_true",
                                help="Show detailed output for all patients")
    
    # Cerebras Cloud Training command
    cloud_train_parser = subparsers.add_parser("cloud_train",
                                               help="Train AND run inference on Cerebras Cloud (no local GPU)")
    cloud_train_parser.add_argument("--api_key", type=str, default=None,
                                   help="Cerebras API key (or set CEREBRAS_API_KEY env var)")
    cloud_train_parser.add_argument("--n_patients", type=int, default=100,
                                   help="Number of patients for training")
    cloud_train_parser.add_argument("--n_test", type=int, default=20,
                                   help="Number of patients for testing")
    cloud_train_parser.add_argument("--epochs", type=int, default=5,
                                   help="Number of learning epochs on Cerebras")
    cloud_train_parser.add_argument("--model_tier", type=str, default="fast",
                                   choices=["fast", "balanced", "accurate"],
                                   help="Model tier")
    cloud_train_parser.add_argument("--save_model", action="store_true",
                                   help="Save learned model to file")
    cloud_train_parser.add_argument("--verbose", action="store_true",
                                   help="Show detailed output")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Dispatch to appropriate command
    commands = {
        "generate": cmd_generate,
        "train": cmd_train,
        "compare": cmd_compare,
        "simulate": cmd_simulate,
        "dashboard": cmd_dashboard,
        "benchmark": cmd_benchmark,
        "cerebras": cmd_cerebras,
        "cloud_train": cmd_cloud_train
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
