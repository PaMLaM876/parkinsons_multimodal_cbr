"""
PPMI (Parkinson's Progression Markers Initiative) Data Loader & Preprocessor.

Merges PPMI clinical CSVs by PATNO + EVENT_ID, maps to internal preprocessing
schema, optionally processes available T1-weighted MRI NIfTI scans, and generates
analysis output for the showcase dashboard.

Usage:
    python scripts/load_ppmi_data.py --ppmi_dir data/real/ppmi
    python scripts/load_ppmi_data.py --ppmi_dir data/real/ppmi --visit BL --process_mri
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import ClinicalTabularPreprocessor, MRI3DPreprocessor


# ═══════════════════════════════════════════════════════════════════
# PPMI CSV File Resolution — Handles naming variations across releases
# ═══════════════════════════════════════════════════════════════════

# Map of logical names → possible file name patterns (case-insensitive)
PPMI_FILE_PATTERNS = {
    "demographics": [
        "Demographics.csv",
        "Participant_Demographics.csv",
        "SCREEN_Demographics.csv",
    ],
    "participant_status": [
        "Participant_Status.csv",
        "Patient_Status.csv",
        "Screening___Demographics.csv",
    ],
    "updrs_part_1": [
        "MDS_UPDRS_Part_I.csv",
        "MDS-UPDRS_Part_I.csv",
        "NUPDRS1.csv",
        "MDS_UPDRS_Part_I_Patient_Questionnaire.csv",
    ],
    "updrs_part_2": [
        "MDS_UPDRS_Part_II.csv",
        "MDS-UPDRS_Part_II.csv",
        "NUPDRS2.csv",
        "MDS_UPDRS_Part_II__Patient_Questionnaire.csv",
    ],
    "updrs_part_3": [
        "MDS_UPDRS_Part_III.csv",
        "MDS-UPDRS_Part_III.csv",
        "NUPDRS3.csv",
        "MDS_UPDRS_Part_III__Post_Dose_.csv",
    ],
    "updrs_part_4": [
        "MDS_UPDRS_Part_IV.csv",
        "MDS-UPDRS_Part_IV.csv",
        "NUPDRS4.csv",
    ],
    "moca": [
        "Montreal_Cognitive_Assessment__MoCA_.csv",
        "Montreal_Cognitive_Assessment.csv",
        "MOCA.csv",
    ],
    "schwab_england": [
        "Modified_Schwab___England_ADL.csv",
        "Modified_Schwab_and_England_ADL.csv",
        "Schwab_England_ADL.csv",
        "MODSEADL.csv",
    ],
    "hoehn_yahr": [
        "Hoehn_and_Yahr.csv",
        "Hoehn___Yahr.csv",
        "NHY.csv",
    ],
    "datscan": [
        "DATScan_Striatal_Binding_Ratio_Results.csv",
        "DATScan_Analysis.csv",
        "DATScan_Imaging.csv",
        "DATScan_SBR.csv",
        "DaTSCAN_Striatal_Binding_Ratio_Results.csv",
    ],
    "biospecimen": [
        "Biospecimen_Analysis_Results.csv",
        "Current_Biospecimen_Analysis_Results.csv",
        "Biospecimen.csv",
    ],
    "family_history": [
        "Family_History__PD_.csv",
        "Family_History_PD.csv",
        "Family_History.csv",
    ],
    "genetics": [
        "Genetic_Testing_Results.csv",
        "Genetic_Results.csv",
        "Genetics.csv",
    ],
}


def find_ppmi_file(clinical_dir: str, logical_name: str) -> Optional[str]:
    """Find a PPMI CSV file using fuzzy name matching across release versions."""
    patterns = PPMI_FILE_PATTERNS.get(logical_name, [])
    all_csvs = {f.lower(): f for f in os.listdir(clinical_dir) if f.endswith(".csv")}

    # Try exact match first
    for pattern in patterns:
        if pattern.lower() in all_csvs:
            return os.path.join(clinical_dir, all_csvs[pattern.lower()])

    # Try substring match
    key_words = logical_name.replace("_", " ").split()
    for csv_lower, csv_actual in all_csvs.items():
        if all(kw in csv_lower for kw in key_words):
            return os.path.join(clinical_dir, csv_actual)

    return None


def safe_read_csv(filepath: Optional[str], usecols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Safely read a CSV, returning None if file doesn't exist or columns are missing."""
    if filepath is None or not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if usecols:
            available_cols = [c for c in usecols if c in df.columns]
            if not available_cols:
                # Try case-insensitive match
                col_map = {c.upper(): c for c in df.columns}
                available_cols = [col_map.get(c.upper(), None) for c in usecols]
                available_cols = [c for c in available_cols if c is not None]
            if available_cols:
                df = df[available_cols]
            else:
                print(f"    [!] Warning: None of {usecols} found in {os.path.basename(filepath)}")
                print(f"        Available columns: {df.columns.tolist()[:15]}...")
                return df  # Return full df so user can inspect
        return df
    except Exception as e:
        print(f"    [!] Error reading {filepath}: {e}")
        return None


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find the first matching column name from a list of candidates (case-insensitive)."""
    col_map = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in col_map:
            return col_map[cand.upper()]
    return None


# ═══════════════════════════════════════════════════════════════════
# Core PPMI Loader
# ═══════════════════════════════════════════════════════════════════

def load_and_merge_ppmi_clinical(clinical_dir: str, visit: str = "BL") -> pd.DataFrame:
    """
    Load all available PPMI clinical CSVs and merge into a unified DataFrame
    aligned on PATNO and the specified visit (default: Baseline 'BL').

    Returns a DataFrame with columns mapped to the internal preprocessing schema.
    """
    print(f"\n{'='*60}")
    print(f"  PPMI Clinical Data Loader")
    print(f"  Source: {clinical_dir}")
    print(f"  Visit:  {visit}")
    print(f"{'='*60}")

    # ─── 1. Participant Status (Cohort Labels) ───
    print("\n[1/10] Loading Participant Status (PD/HC/SWEDD cohort)...")
    status_path = find_ppmi_file(clinical_dir, "participant_status")
    df_status = safe_read_csv(status_path)

    if df_status is None:
        print("    [!] CRITICAL: Participant_Status.csv not found!")
        print("    Please download it from PPMI → Study Data → Participant Status")
        return pd.DataFrame()

    patno_col = find_column(df_status, ["PATNO", "Subject", "SubjectID"])
    cohort_col = find_column(df_status, ["COHORT", "APPRDX", "DIAGNOSIS", "COHORT_DEFINITION"])
    enroll_col = find_column(df_status, ["ENROLL_STATUS", "STATUS", "ENROLLMENT_STATUS"])

    if patno_col is None:
        print(f"    [!] Cannot find PATNO column. Available: {df_status.columns.tolist()[:10]}")
        return pd.DataFrame()

    # Filter to enrolled subjects
    if enroll_col and enroll_col in df_status.columns:
        df_status = df_status[df_status[enroll_col].astype(str).str.lower().isin(["enrolled", "complete", "1"])]

    # Build base patient roster
    merged = pd.DataFrame()
    merged["PATNO"] = df_status[patno_col].astype(int)

    if cohort_col:
        merged["cohort_raw"] = df_status[cohort_col].values
        # Map cohort to PD / HC
        cohort_map = {
            "PD": "PD", "Parkinson Disease": "PD", "Parkinson's Disease": "PD",
            "Healthy Control": "HC", "Control": "HC", "HC": "HC",
            "SWEDD": "SWEDD", "Scans Without Evidence of Dopaminergic Deficit": "SWEDD",
            "Prodromal": "Prodromal", "PRODROMAL": "Prodromal",
            "GenCohort PD": "PD", "GenCohort Unaffected": "HC",
            "GenReg PD": "PD", "GenReg Unaffected": "HC",
        }
        merged["diagnosis"] = merged["cohort_raw"].map(
            lambda x: cohort_map.get(str(x).strip(), "Other")
        )
    else:
        merged["diagnosis"] = "Unknown"

    merged = merged.drop_duplicates(subset=["PATNO"]).reset_index(drop=True)
    print(f"    ✓ Found {len(merged)} subjects")
    if cohort_col:
        print(f"    Cohorts: {merged['diagnosis'].value_counts().to_dict()}")

    # ─── 2. Demographics ───
    print("\n[2/10] Loading Demographics...")
    demo_path = find_ppmi_file(clinical_dir, "demographics")
    df_demo = safe_read_csv(demo_path)
    if df_demo is not None:
        dp = find_column(df_demo, ["PATNO"])
        sex_col = find_column(df_demo, ["SEX", "GENDER", "GENSEX"])
        age_col = find_column(df_demo, ["AGE_AT_VISIT", "ENROLL_AGE", "BIRTHDT", "AGE"])
        hand_col = find_column(df_demo, ["HANDED", "HANDEDNESS", "DOMHAND"])

        if dp:
            demo_data = df_demo.drop_duplicates(subset=[dp])[[c for c in [dp, sex_col, age_col, hand_col] if c]].copy()
            demo_data = demo_data.rename(columns={dp: "PATNO"})
            demo_data["PATNO"] = demo_data["PATNO"].astype(int)

            if sex_col:
                sex_map = {0: "F", 1: "M", 2: "Other", "Male": "M", "Female": "F", "M": "M", "F": "F"}
                demo_data["sex"] = demo_data[sex_col].map(lambda x: sex_map.get(x, sex_map.get(str(x), "Unknown")))
            if age_col:
                demo_data["age"] = pd.to_numeric(demo_data[age_col], errors="coerce")
            if hand_col:
                hand_map = {1: "Right", 2: "Left", 3: "Symmetric",
                            "Right": "Right", "Left": "Left", "Mixed": "Symmetric"}
                demo_data["dominant_side"] = demo_data[hand_col].map(
                    lambda x: hand_map.get(x, hand_map.get(str(x), "Unknown"))
                )

            keep_cols = ["PATNO"] + [c for c in ["sex", "age", "dominant_side"] if c in demo_data.columns]
            merged = merged.merge(demo_data[keep_cols], on="PATNO", how="left")
            print(f"    ✓ Merged demographics for {demo_data[dp].nunique()} subjects")
    else:
        print("    ⚠ Demographics file not found — skipping")

    # ─── Helper: Merge a visit-level PPMI file ───
    def merge_visit_file(logical_name: str, step: str, target_cols: Dict[str, List[str]],
                         label: str) -> None:
        nonlocal merged
        print(f"\n[{step}] Loading {label}...")
        fpath = find_ppmi_file(clinical_dir, logical_name)
        df = safe_read_csv(fpath)
        if df is None:
            print(f"    ⚠ {label} file not found — skipping")
            return

        p_col = find_column(df, ["PATNO"])
        ev_col = find_column(df, ["EVENT_ID", "VISIT", "EVENT"])
        if not p_col:
            print(f"    ⚠ No PATNO column found — skipping")
            return

        # Filter to target visit
        if ev_col:
            df_visit = df[df[ev_col].astype(str).str.strip().str.upper() == visit.upper()].copy()
        else:
            df_visit = df.copy()

        if len(df_visit) == 0:
            print(f"    ⚠ No records for visit '{visit}' — trying to use latest available")
            df_visit = df.drop_duplicates(subset=[p_col], keep="last")

        df_visit[p_col] = df_visit[p_col].astype(int)
        df_visit = df_visit.drop_duplicates(subset=[p_col], keep="last")

        # Map columns
        rename_map = {}
        for our_name, candidates in target_cols.items():
            found_col = find_column(df_visit, candidates)
            if found_col:
                rename_map[found_col] = our_name

        if rename_map:
            df_mapped = df_visit[[p_col] + list(rename_map.keys())].copy()
            df_mapped = df_mapped.rename(columns={**rename_map, p_col: "PATNO"})
            merged = merged.merge(df_mapped, on="PATNO", how="left")
            print(f"    ✓ Mapped {len(rename_map)} columns: {list(rename_map.values())}")
        else:
            print(f"    ⚠ No target columns found in {label}")
            print(f"      Available: {df.columns.tolist()[:15]}...")

    # ─── 3. MDS-UPDRS Part I ───
    merge_visit_file("updrs_part_1", "3/10",
                     {"updrs_part_1": ["NP1TOTAL", "NP1TOT", "NUPDRS1_TOTAL", "TOTAL"]},
                     "MDS-UPDRS Part I")

    # ─── 4. MDS-UPDRS Part II ───
    merge_visit_file("updrs_part_2", "4/10",
                     {"updrs_part_2": ["NP2TOTAL", "NP2TOT", "NUPDRS2_TOTAL", "NP2PTOT"]},
                     "MDS-UPDRS Part II")

    # ─── 5. MDS-UPDRS Part III (Motor) ───
    merge_visit_file("updrs_part_3", "5/10",
                     {"updrs_part_3": ["NP3TOT", "NP3TOTAL", "NUPDRS3_TOTAL", "NP3_TOTAL"]},
                     "MDS-UPDRS Part III (Motor)")

    # ─── 6. MDS-UPDRS Part IV ───
    merge_visit_file("updrs_part_4", "6/10",
                     {"updrs_part_4": ["NP4TOT", "NP4TOTAL", "NUPDRS4_TOTAL"]},
                     "MDS-UPDRS Part IV")

    # ─── 7. MoCA ───
    merge_visit_file("moca", "7/10",
                     {"moca_score": ["MCATOT", "MCATOTAL", "MOCA_TOTAL"]},
                     "Montreal Cognitive Assessment (MoCA)")

    # ─── 8. Schwab & England ADL ───
    merge_visit_file("schwab_england", "8/10",
                     {"schwab_england_adl": ["MSEADLG", "SCHWAB_ENGLAND", "SE_ADL"]},
                     "Schwab & England ADL")

    # ─── 9. Hoehn & Yahr ───
    merge_visit_file("hoehn_yahr", "9/10",
                     {"hoehn_yahr_stage": ["NHY", "H_AND_Y", "HOEHN_YAHR", "HY_STAGE"]},
                     "Hoehn & Yahr Staging")

    # ─── 10. DaTscan SBR ───
    print("\n[10/10] Loading DaTscan Striatal Binding Ratios...")
    dat_path = find_ppmi_file(clinical_dir, "datscan")
    df_dat = safe_read_csv(dat_path)
    if df_dat is not None:
        dp = find_column(df_dat, ["PATNO"])
        ev = find_column(df_dat, ["EVENT_ID", "VISIT"])
        if dp:
            if ev:
                df_dat_visit = df_dat[df_dat[ev].astype(str).str.strip().str.upper() == visit.upper()].copy()
            else:
                df_dat_visit = df_dat.copy()

            if len(df_dat_visit) == 0:
                df_dat_visit = df_dat.drop_duplicates(subset=[dp], keep="last")

            df_dat_visit[dp] = df_dat_visit[dp].astype(int)
            df_dat_visit = df_dat_visit.drop_duplicates(subset=[dp], keep="last")

            dat_cols = {
                "datscan_caudate_left": ["CAUDATE_L", "CAUDATE_LEFT", "L_CAUDATE", "LCAUDATE"],
                "datscan_caudate_right": ["CAUDATE_R", "CAUDATE_RIGHT", "R_CAUDATE", "RCAUDATE"],
                "datscan_putamen_left": ["PUTAMEN_L", "PUTAMEN_LEFT", "L_PUTAMEN", "LPUTAMEN"],
                "datscan_putamen_right": ["PUTAMEN_R", "PUTAMEN_RIGHT", "R_PUTAMEN", "RPUTAMEN"],
            }
            rename_map = {}
            for our_name, candidates in dat_cols.items():
                found = find_column(df_dat_visit, candidates)
                if found:
                    rename_map[found] = our_name

            if rename_map:
                df_mapped = df_dat_visit[[dp] + list(rename_map.keys())].copy()
                df_mapped = df_mapped.rename(columns={**rename_map, dp: "PATNO"})

                # Convert to numeric
                for col in rename_map.values():
                    if col in df_mapped.columns:
                        df_mapped[col] = pd.to_numeric(df_mapped[col], errors="coerce")

                # Compute asymmetry index
                if "datscan_putamen_left" in df_mapped.columns and "datscan_putamen_right" in df_mapped.columns:
                    pl = df_mapped["datscan_putamen_left"]
                    pr = df_mapped["datscan_putamen_right"]
                    df_mapped["datscan_asymmetry_index"] = (
                        (pl - pr).abs() / (pl + pr + 1e-6)
                    ).round(4)

                merged = merged.merge(df_mapped, on="PATNO", how="left")
                print(f"    ✓ Mapped {len(rename_map)} DaTscan columns + asymmetry index")
            else:
                print(f"    ⚠ No DaTscan SBR columns found")
                print(f"      Available: {df_dat.columns.tolist()[:15]}...")
    else:
        print("    ⚠ DaTscan file not found — skipping")

    # ─── Compute derived fields ───
    updrs_parts = ["updrs_part_1", "updrs_part_2", "updrs_part_3", "updrs_part_4"]
    present_parts = [p for p in updrs_parts if p in merged.columns]
    if present_parts:
        for col in present_parts:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged["total_updrs"] = merged[present_parts].sum(axis=1, min_count=1)

    # Map subject_id
    merged["subject_id"] = merged["PATNO"].apply(lambda x: f"PPMI_{int(x):04d}")

    # Clean up hoehn_yahr
    if "hoehn_yahr_stage" in merged.columns:
        merged["hoehn_yahr_stage"] = merged["hoehn_yahr_stage"].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace(".", "").replace("-", "").isdigit() else str(x) if pd.notna(x) else "0"
        )

    # Disease duration placeholder (baseline = 0 for new PD, varies for others)
    merged["disease_duration_months"] = 0

    # Drop internal cols
    merged = merged.drop(columns=["PATNO", "cohort_raw"], errors="ignore")

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"  PPMI MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"  Total Subjects     : {len(merged)}")
    print(f"  Cohort Distribution: {merged['diagnosis'].value_counts().to_dict()}")
    print(f"  Columns Loaded     : {len(merged.columns)}")
    available_features = [c for c in merged.columns if c not in ["subject_id", "diagnosis", "cohort_raw"]]
    print(f"  Features Available : {available_features}")

    non_null_pcts = {}
    for col in available_features:
        pct = merged[col].notna().mean() * 100
        non_null_pcts[col] = f"{pct:.0f}%"
    print(f"  Data Completeness  : {non_null_pcts}")
    print(f"{'='*60}")

    return merged


# ═══════════════════════════════════════════════════════════════════
# MRI Scanner — Find Available NIfTI Scans
# ═══════════════════════════════════════════════════════════════════

def scan_mri_directory(mri_dir: str, patient_ids: List[str]) -> Dict[str, str]:
    """Scan MRI directory for NIfTI files matching patient IDs."""
    if not os.path.exists(mri_dir):
        return {}

    mri_map = {}
    # Check for various naming patterns
    for pid in patient_ids:
        # Extract numeric PATNO from subject_id
        patno = pid.replace("PPMI_", "")

        candidates = [
            os.path.join(mri_dir, f"{patno}.nii.gz"),
            os.path.join(mri_dir, f"{patno}.nii"),
            os.path.join(mri_dir, f"{patno}.npy"),
            os.path.join(mri_dir, f"PPMI_{patno}.nii.gz"),
            os.path.join(mri_dir, f"s{patno}.nii.gz"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                mri_map[pid] = cand
                break

    # Also check subdirectories (PPMI often downloads as PATNO/session/scan.nii.gz)
    if not mri_map:
        for entry in os.listdir(mri_dir):
            entry_path = os.path.join(mri_dir, entry)
            if os.path.isdir(entry_path):
                # Look for .nii.gz inside subdirectories
                niftis = glob.glob(os.path.join(entry_path, "**", "*.nii.gz"), recursive=True)
                niftis += glob.glob(os.path.join(entry_path, "**", "*.nii"), recursive=True)
                if niftis:
                    pid = f"PPMI_{int(entry):04d}" if entry.isdigit() else entry
                    if pid in patient_ids:
                        mri_map[pid] = niftis[0]  # Take the first available scan

    return mri_map


# ═══════════════════════════════════════════════════════════════════
# Analysis & Showcase Data Generation
# ═══════════════════════════════════════════════════════════════════

def generate_ppmi_analysis(merged: pd.DataFrame, mri_map: Dict[str, str],
                           mri_dir: str, output_dir: str, process_mri: bool = False) -> dict:
    """Generate comprehensive analysis JSON for the showcase."""
    print("\n[*] Generating PPMI analysis for showcase...")
    os.makedirs(output_dir, exist_ok=True)

    clinical_features = [
        "age", "updrs_part_1", "updrs_part_2", "updrs_part_3", "updrs_part_4",
        "total_updrs", "moca_score", "schwab_england_adl",
        "datscan_caudate_left", "datscan_caudate_right",
        "datscan_putamen_left", "datscan_putamen_right",
        "datscan_asymmetry_index",
    ]

    # Compute per-feature stats with PD vs HC comparison
    feature_stats = {}
    groups = merged["diagnosis"].unique().tolist()
    has_pd_hc = "PD" in groups and "HC" in groups

    for feat in clinical_features:
        if feat not in merged.columns:
            continue
        vals = pd.to_numeric(merged[feat], errors="coerce").dropna()
        if len(vals) == 0:
            continue

        entry = {
            "overall": {
                "mean": round(float(vals.mean()), 3),
                "std": round(float(vals.std()), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
                "median": round(float(vals.median()), 3),
                "count": int(len(vals)),
            }
        }

        if has_pd_hc:
            group_data = {}
            for g in ["PD", "HC"]:
                if g not in groups:
                    continue
                g_vals = pd.to_numeric(merged[merged["diagnosis"] == g][feat], errors="coerce").dropna()
                if len(g_vals) == 0:
                    continue
                group_data[g] = {
                    "mean": round(float(g_vals.mean()), 3),
                    "std": round(float(g_vals.std()), 3),
                    "count": int(len(g_vals)),
                }

            # Welch's t-test
            pd_vals = pd.to_numeric(merged[merged["diagnosis"] == "PD"][feat], errors="coerce").dropna()
            hc_vals = pd.to_numeric(merged[merged["diagnosis"] == "HC"][feat], errors="coerce").dropna()
            if len(pd_vals) > 1 and len(hc_vals) > 1:
                t_stat, p_val = stats.ttest_ind(pd_vals, hc_vals, equal_var=False)
                pooled_std = np.sqrt((pd_vals.std() ** 2 + hc_vals.std() ** 2) / 2)
                cohens_d = abs(pd_vals.mean() - hc_vals.mean()) / (pooled_std + 1e-8)
                group_data["test"] = {
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": float(p_val),
                    "cohens_d": round(float(cohens_d), 4),
                }
            entry["groups"] = group_data

        feature_stats[feat] = entry

    # Run through ClinicalTabularPreprocessor
    preprocessor_demo = {}
    try:
        prep = ClinicalTabularPreprocessor(target_label_col="diagnosis")
        prep.fit(merged)
        X, y = prep.transform(merged)
        preprocessor_demo = {
            "n_features_output": int(X.shape[1]),
            "feature_names": prep.feature_names_out[:20],
            "shape": list(X.shape),
            "has_nans": bool(np.isnan(X).any()),
        }
        print(f"    ✓ ClinicalTabularPreprocessor: {X.shape[0]} patients → {X.shape[1]}D feature vectors")
    except Exception as e:
        print(f"    ⚠ ClinicalTabularPreprocessor error: {e}")

    # MRI Processing
    mri_results = {}
    if process_mri and mri_map:
        print(f"\n[*] Processing {len(mri_map)} MRI scans through MRI3DPreprocessor...")
        mri_prep = MRI3DPreprocessor(target_shape=(96, 96, 96), apply_skull_strip=True)
        for pid, mri_path in list(mri_map.items())[:10]:  # Process up to 10 for demo
            try:
                tensor, slices = mri_prep.preprocess_pipeline(mri_path)
                mri_results[pid] = {
                    "shape": list(tensor.shape),
                    "min": round(float(tensor.min()), 4),
                    "max": round(float(tensor.max()), 4),
                    "mean": round(float(tensor.mean()), 4),
                    "path": mri_path,
                }
                print(f"    ✓ {pid}: {tensor.shape}, range [{tensor.min():.3f}, {tensor.max():.3f}]")
            except Exception as e:
                print(f"    ✗ {pid}: {e}")

    # Build result
    result = {
        "dataset_name": "PPMI (Parkinson's Progression Markers Initiative)",
        "n_subjects": len(merged),
        "cohort_distribution": merged["diagnosis"].value_counts().to_dict(),
        "visit": "BL (Baseline)",
        "features_available": [c for c in clinical_features if c in merged.columns],
        "feature_stats": feature_stats,
        "preprocessing": preprocessor_demo,
        "mri_available": len(mri_map),
        "mri_processed": len(mri_results),
        "mri_results": mri_results,
    }

    out_path = os.path.join(output_dir, "ppmi_analysis.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[+] PPMI analysis saved: {out_path}")

    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PPMI Data Loader & Preprocessor")
    parser.add_argument("--ppmi_dir", type=str, default="./data/real/ppmi",
                        help="Root PPMI data directory (containing clinical/ and mri/ subdirs)")
    parser.add_argument("--visit", type=str, default="BL",
                        help="Visit code to extract (default: BL = Baseline)")
    parser.add_argument("--process_mri", action="store_true",
                        help="Process available MRI NIfTI scans through 3D preprocessor")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for analysis files")
    args = parser.parse_args()

    clinical_dir = os.path.join(args.ppmi_dir, "clinical")
    mri_dir = os.path.join(args.ppmi_dir, "mri")
    output_dir = args.output_dir or os.path.join(args.ppmi_dir, "analysis")

    if not os.path.exists(clinical_dir):
        print(f"\n[!] Clinical directory not found: {clinical_dir}")
        print(f"\n📥 PPMI Download Instructions:")
        print(f"   1. Log in to https://www.ppmi-info.org/access-data-specimens/download-data")
        print(f"   2. Go to Download → Study Data")
        print(f"   3. Select all clinical CSV files and download")
        print(f"   4. Extract and place CSVs into: {os.path.abspath(clinical_dir)}")
        print(f"\n   For MRI scans:")
        print(f"   5. Go to Download → Image Collections → Advanced Search")
        print(f"   6. Filter: Modality=MRI, Protocol=T1/MPRAGE, Format=NIfTI")
        print(f"   7. Place NIfTI files into: {os.path.abspath(mri_dir)}")
        return

    # List available CSVs
    csv_files = [f for f in os.listdir(clinical_dir) if f.endswith(".csv")]
    print(f"\n📂 Found {len(csv_files)} CSV files in {clinical_dir}:")
    for f in sorted(csv_files):
        size_kb = os.path.getsize(os.path.join(clinical_dir, f)) / 1024
        print(f"   • {f} ({size_kb:.0f} KB)")

    # Load and merge
    merged = load_and_merge_ppmi_clinical(clinical_dir, visit=args.visit)

    if merged.empty:
        print("\n[!] No data loaded. Check that CSVs are in the correct directory.")
        return

    # Save merged CSV
    merged_path = os.path.join(args.ppmi_dir, f"ppmi_merged_{args.visit.lower()}.csv")
    merged.to_csv(merged_path, index=False)
    print(f"\n[+] Merged PPMI data saved: {merged_path}")

    # Scan for MRI files
    mri_map = scan_mri_directory(mri_dir, merged["subject_id"].tolist())
    if mri_map:
        print(f"\n🧠 Found {len(mri_map)} MRI scans in {mri_dir}")
    else:
        print(f"\n🧠 No MRI scans found in {mri_dir} (this is OK for clinical-only analysis)")

    # Generate analysis
    generate_ppmi_analysis(
        merged, mri_map, mri_dir, output_dir, process_mri=args.process_mri
    )

    print(f"\n{'='*60}")
    print(f"  PPMI INTEGRATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Merged CSV      : {os.path.abspath(merged_path)}")
    print(f"  Analysis Output  : {os.path.abspath(output_dir)}")
    print(f"  MRI Available    : {len(mri_map)} scans")
    print(f"\n  Next steps:")
    print(f"  1. Verify ppmi_merged_{args.visit.lower()}.csv looks correct")
    print(f"  2. If you have MRI scans, re-run with --process_mri flag")
    print(f"  3. Rebuild showcase: python scripts/build_showcase.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
