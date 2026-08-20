"""
Download UCI Parkinson's Datasets and Map to Internal Schema.

Fetches:
  1. UCI Parkinson's Speech (ID=174): 195 samples, 31 subjects, PD vs HC classification
  2. UCI Parkinson's Telemonitoring (ID=189): 5875 samples, 42 PD subjects, UPDRS regression

Saves mapped CSVs into data/real/uci_parkinsons/ and data/real/uci_telemonitoring/
"""

import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def download_uci_parkinsons_speech(output_dir: str) -> pd.DataFrame:
    """
    Download UCI Parkinson's Speech dataset (ID=174).
    195 voice recordings from 31 people (23 PD, 8 HC).
    Features: MDVP Jitter, Shimmer, HNR, RPDE, DFA, D2, PPE, F0 stats.
    """
    from ucimlrepo import fetch_ucirepo

    print("[*] Downloading UCI Parkinson's Speech Dataset (ID=174)...")
    dataset = fetch_ucirepo(id=174)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # Save raw as-is
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_parkinsons.csv")
    df.to_csv(raw_path, index=False)
    print(f"[+] Saved raw dataset: {raw_path} ({len(df)} samples)")

    # Map to our internal schema
    mapped = pd.DataFrame()
    mapped["subject_id"] = df["name"] if "name" in df.columns else [f"UCI_S{i:03d}" for i in range(len(df))]
    mapped["diagnosis"] = df["status"].map({1: "PD", 0: "HC"})

    # Acoustic features mapping
    # Note: ucimlrepo strips parenthetical units from column names, e.g.
    # "MDVP:Fo(Hz)" becomes "MDVP:Fo", "MDVP:Shimmer(dB)" becomes "MDVP:Shimmer.1"
    col_map = {
        "MDVP:Fo": "f0_mean",
        "MDVP:Fhi": "f0_max",
        "MDVP:Flo": "f0_min",
        "MDVP:Jitter": "jitter_local",
        "MDVP:RAP": "jitter_rap",
        "Jitter:DDP": "jitter_ddp",
        "MDVP:Shimmer": "shimmer_local",
        "Shimmer:APQ3": "shimmer_apq3",
        "Shimmer:APQ5": "shimmer_apq5",
        "MDVP:APQ": "shimmer_apq11",
        "Shimmer:DDA": "shimmer_dda",
        "NHR": "nhr",
        "HNR": "hnr",
        "RPDE": "rpde",
        "DFA": "dfa",
        "spread1": "spread1",
        "spread2": "spread2",
        "D2": "d2",
        "PPE": "ppe",
    }

    for uci_col, our_col in col_map.items():
        if uci_col in df.columns:
            vals = df[uci_col]
            # Handle case where pandas returns a DataFrame (duplicate cols) instead of Series
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[:, 0]
            mapped[our_col] = vals.values

    mapped_path = os.path.join(output_dir, "mapped_parkinsons.csv")
    mapped.to_csv(mapped_path, index=False)
    print(f"[+] Saved mapped dataset: {mapped_path}")

    # Generate summary stats
    n_pd = int((mapped["diagnosis"] == "PD").sum())
    n_hc = int((mapped["diagnosis"] == "HC").sum())

    summary = {
        "dataset_name": "UCI Parkinson's Speech Dataset",
        "uci_id": 174,
        "total_samples": len(mapped),
        "n_pd": n_pd,
        "n_hc": n_hc,
        "pd_ratio": round(n_pd / len(mapped), 3),
        "n_features": len(col_map),
        "features": list(col_map.values()),
        "source_url": "https://archive.ics.uci.edu/dataset/174/parkinsons",
        "citation": "Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M. (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection. BioMedical Engineering OnLine, 6:23.",
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] PD: {n_pd}, HC: {n_hc}, Features: {len(col_map)}")
    return mapped


def download_uci_telemonitoring(output_dir: str) -> pd.DataFrame:
    """
    Download UCI Parkinson's Telemonitoring dataset (ID=189).
    5875 voice recordings from 42 early-stage PD patients over ~6 months.
    Features: Jitter, Shimmer, HNR, NHR + motor_UPDRS, total_UPDRS as targets.
    """
    from ucimlrepo import fetch_ucirepo

    print("\n[*] Downloading UCI Parkinson's Telemonitoring Dataset (ID=189)...")
    dataset = fetch_ucirepo(id=189)

    X = dataset.data.features
    y = dataset.data.targets
    ids = dataset.data.ids

    df = pd.concat([ids, X, y], axis=1)

    # Save raw
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_telemonitoring.csv")
    df.to_csv(raw_path, index=False)
    print(f"[+] Saved raw dataset: {raw_path} ({len(df)} samples)")

    # Map to our schema
    mapped = pd.DataFrame()
    if "subject#" in df.columns:
        mapped["subject_id"] = df["subject#"].apply(lambda x: f"TELE_{int(x):03d}")
    else:
        mapped["subject_id"] = [f"TELE_{i:04d}" for i in range(len(df))]
    mapped["diagnosis"] = "PD"  # All subjects in this dataset are PD patients

    # Demographics
    if "age" in df.columns:
        mapped["age"] = df["age"].values
    if "sex" in df.columns:
        mapped["sex"] = df["sex"].map({0: "M", 1: "F"}).values
    if "test_time" in df.columns:
        mapped["test_time_days"] = df["test_time"].values

    # Acoustic features mapping (ucimlrepo keeps parenthetical names for this dataset)
    col_map = {
        "Jitter(%)": "jitter_local",
        "Jitter(Abs)": "jitter_abs",
        "Jitter:RAP": "jitter_rap",
        "Jitter:PPQ5": "jitter_ppq5",
        "Jitter:DDP": "jitter_ddp",
        "Shimmer": "shimmer_local",
        "Shimmer(dB)": "shimmer_db",
        "Shimmer:APQ3": "shimmer_apq3",
        "Shimmer:APQ5": "shimmer_apq5",
        "Shimmer:APQ11": "shimmer_apq11",
        "Shimmer:DDA": "shimmer_dda",
        "NHR": "nhr",
        "HNR": "hnr",
        "RPDE": "rpde",
        "DFA": "dfa",
        "PPE": "ppe",
    }

    for uci_col, our_col in col_map.items():
        if uci_col in df.columns:
            vals = df[uci_col]
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[:, 0]
            mapped[our_col] = vals.values

    # UPDRS targets
    if "motor_UPDRS" in df.columns:
        mapped["updrs_part_3"] = df["motor_UPDRS"].values
    if "total_UPDRS" in df.columns:
        mapped["total_updrs"] = df["total_UPDRS"].values

    mapped_path = os.path.join(output_dir, "mapped_telemonitoring.csv")
    mapped.to_csv(mapped_path, index=False)
    print(f"[+] Saved mapped dataset: {mapped_path}")

    n_subjects = df["subject#"].nunique() if "subject#" in df.columns else 42

    summary = {
        "dataset_name": "UCI Parkinson's Telemonitoring Dataset",
        "uci_id": 189,
        "total_samples": len(mapped),
        "n_unique_subjects": int(n_subjects),
        "n_features": len(col_map),
        "features": list(col_map.values()),
        "targets": ["motor_UPDRS (updrs_part_3)", "total_UPDRS (total_updrs)"],
        "source_url": "https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring",
        "citation": "Tsanas, A., Little, M.A., McSharry, P.E., Ramig, L.O. (2010). Accurate Telemonitoring of Parkinson's Disease Progression by Noninvasive Speech Tests. IEEE Trans. Biomedical Engineering, 57(4):884-893.",
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Subjects: {n_subjects}, Samples: {len(mapped)}, Features: {len(col_map)}")
    return mapped


if __name__ == "__main__":
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "real"))

    df_speech = download_uci_parkinsons_speech(
        os.path.join(data_root, "uci_parkinsons")
    )
    df_tele = download_uci_telemonitoring(
        os.path.join(data_root, "uci_telemonitoring")
    )

    print("\n" + "=" * 60)
    print("  UCI DATASET DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Speech Dataset  : {len(df_speech)} samples")
    print(f"  Telemonitoring  : {len(df_tele)} samples")
    print(f"  Output Dir      : {data_root}")
    print("=" * 60)
