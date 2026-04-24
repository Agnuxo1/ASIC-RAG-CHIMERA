#!/usr/bin/env python3
"""
================================================================================
TPF DEEP READOUT TRAINING
================================================================================
Trains a Neural Network (MLP) to distinguish between "Target-Bound" hashing
and "Dead-End" entropy based on real LV06 silicon jitter.

Input: tpf_balanced_dataset.csv
Output: Trained model weights for real-time inference.
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_tpf_model():
    dataset_path = "tpf_continuous_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"[ERR] Dataset not found at {dataset_path}")
        return

    # 1. Load Data
    print(f"[DISK] Loading dataset: {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Features: window_shares, jitter_mean, jitter_std
    X = df[["window_shares", "jitter_mean", "jitter_std"]]
    y = df["label"]

    # 2. Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Model Architecture (Deep MLP)
    # 3 hidden layers to capture nonlinear silicon dynamics
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8, 4), 
        activation='relu', 
        solver='adam', 
        max_iter=2000, 
        random_state=42
    )

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"[TRAIN] Training Deep Readout on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test)
    print("\n" + "="*40)
    print("   TPF DEEP READOUT METRICS")
    print("="*40)
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    # CM: [[TN, FP], [FN, TP]]
    # TN: Loser identified as Loser (Correct Abort)
    # TP: Winner identified as Winner (Correct Continue)
    # FN: Winner aborted (FALSE ABORT - Very dangerous!)
    # FP: Loser continued (Energy waste - Suboptimal)
    
    print(f"Confusion Matrix:\n{cm}")
    
    if len(cm) > 1:
        false_aborts = cm[1][0]
        reliability = (1.0 - (false_aborts / (sum(y_test)+1e-9))) * 100
    else:
        reliability = 100.0

    print(f"\nRELIABILITY (Winning Block Preservation): {reliability:.2f}%")
    
    if reliability < 100:
        print("[WARNING] The model aborted a valid block signature. Increase dataset size.")
    else:
        print("[SUCCESS] Model is 100% reliable for captured samples.")

    # 6. Save Artifacts
    joblib.dump(model, "tpf_deep_model.pkl")
    joblib.dump(scaler, "tpf_scaler.pkl")
    print("\n[DISK] Model and Scaler saved for Stratum integration.")

if __name__ == "__main__":
    train_tpf_model()
