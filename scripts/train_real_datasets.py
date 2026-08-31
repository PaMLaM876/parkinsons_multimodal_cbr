"""
Train independent models on the three real datasets for Review 2.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def train_uci_speech():
    print("[*] Training UCI Parkinson's (Speech) Classifier...")
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "real", "uci_parkinsons", "raw_parkinsons.csv"))
    
    X = df.drop(columns=["status", "name"], errors="ignore")
    y = df["status"]
    
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 3),
        "f1_score": round(float(f1_score(y_test, preds)), 3),
        "dataset_size": len(df)
    }

def train_uci_telemonitoring():
    print("[*] Training UCI Telemonitoring (Progression) Regressor...")
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "real", "uci_telemonitoring", "raw_telemonitoring.csv"))
    
    X = df.drop(columns=["motor_UPDRS", "total_UPDRS", "subject#", "test_time"], errors="ignore")
    y = df["motor_UPDRS"]
    
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    
    return {
        "mse": round(float(mean_squared_error(y_test, preds)), 3),
        "r2_score": round(float(r2_score(y_test, preds)), 3),
        "dataset_size": len(df)
    }

def train_ppmi_clinical():
    print("[*] Training PPMI Clinical Tabular Classifier...")
    df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "real", "ppmi", "ppmi_merged_bl.csv"))
    
    if "diagnosis" not in df.columns:
        print("[!] Could not find diagnosis column in PPMI data.")
        return None
        
    df = df.dropna(subset=["diagnosis"])
    unique_vals = df["diagnosis"].unique()
    if len(unique_vals) < 2:
        print("[!] Not enough classes in PPMI data.")
        return None
        
    # PD vs HC classification (assume first class is PD or whatever)
    pd_label = unique_vals[0] if unique_vals[0] != "Healthy Control" else unique_vals[1]
    df['target'] = (df["diagnosis"] == pd_label).astype(int)
    
    X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore')
    
    # Handle NaNs with SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 3),
        "f1_score": round(float(f1_score(y_test, preds)), 3),
        "dataset_size": len(df)
    }

if __name__ == "__main__":
    results = {}
    
    try:
        results["uci_speech"] = train_uci_speech()
    except Exception as e:
        print(f"[!] Error training UCI Speech: {e}")
        
    try:
        results["uci_telemonitoring"] = train_uci_telemonitoring()
    except Exception as e:
        print(f"[!] Error training UCI Telemonitoring: {e}")
        
    try:
        results["ppmi_clinical"] = train_ppmi_clinical()
    except Exception as e:
        print(f"[!] Error training PPMI Clinical: {e}")
        
    out_dir = os.path.join(PROJECT_ROOT, "data", "real", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "review2_results.json")
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"[+] Review 2 results saved to {out_file}")
