#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# =============================================================================
# AUTO-VALIDATION SCRIPT
# =============================================================================

BASE_DIR = Path("d:/ASIC_RAG/REAL_DATA_LV06/chimera_handoff")
NEW_BASE = BASE_DIR / "NEW"
BASELINE_FILE = NEW_BASE / "definitive_results_1766347344.json"
RUNS_DIR = BASE_DIR / "runs"

def run_experiment():
    """Runs the integrated bridge experiment"""
    print("🚀 STARTING INTEGRATED MEDICAL-HARDWARE VALIDATION...")
    
    # We run the bridge module
    # Note: In a real scenario, this would block until duration completes.
    # For automation, we assume the user starts the miner.
    try:
        cmd = [sys.executable, "-m", "chimera_handoff.experiments.chimera_medical_handoff"]
        # Set PYTHONPATH to include the src directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR / "chimera_handoff" / "python" / "src")
        
        proc = subprocess.run(cmd, env=env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        return False

def compare_results(new_metrics_path: Path):
    """Compares new metrics with baseline"""
    if not BASELINE_FILE.exists():
        print(f"⚠️ Baseline file not found: {BASELINE_FILE}")
        return
    
    with open(BASELINE_FILE, "r") as f:
        baseline = json.load(f)
        
    if not new_metrics_path.exists():
        print(f"❌ New metrics not found at {new_metrics_path}")
        return
        
    with open(new_metrics_path, "r") as f:
        new = json.load(f)
        
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON: BASELINE VS IMPROVED")
    print("="*80)
    print(f"{'Metric':<30} | {'Baseline (1766)':<20} | {'New (Integrated)':<20}")
    print("-" * 80)
    
    # Map definitive_results structure to application_metrics structure
    m_map = {
        "Hashrate (MH/s)": (baseline["performance"]["avg_hashrate_mhs"], new["effective_hashrate_mhs"]),
        "GPU Multiplier": (baseline["comparative"]["vs_rtx3090_efficiency_multiplier"], new["vs_rtx3090_efficiency"]),
        "Shares Captured": (baseline["performance"]["total_shares"], new["shares_captured"]),
    }
    
    for label, (b_val, n_val) in m_map.items():
        diff = ((n_val / b_val) - 1) * 100 if b_val > 0 else 0
        status = "🟢 IMPROVED" if diff > 0 else "🔴 DEGRADED"
        print(f"{label:<30} | {b_val:<20.2f} | {n_val:<20.2f} ({status} {diff:+.1f}%)")
        
    print("="*80)

def main():
    # 1. Start the Experiment
    # In this automated script, we assume the environment is ready.
    # We won't actually run it if the user is present to avoid port conflicts,
    # but we provide the logic for the final "one-click" validation.
    
    run_id = f"lv06_medical_validation_FINAL"
    run_dir = RUNS_DIR / run_id
    metrics_path = run_dir / "application_metrics.json"
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        if run_experiment():
            compare_results(metrics_path)
    else:
        # Just compare if data already exists
        if metrics_path.exists():
            compare_results(metrics_path)
        else:
            print(f"No existing data at {metrics_path}. Use --run to initiate hardware capture.")

if __name__ == "__main__":
    main()
