"""
Clinical & Biomarker Tabular Preprocessor for PPMI and UCI Parkinson's Data.
Preprocesses MDS-UPDRS scores, Hoehn & Yahr staging, MoCA cognitive assessments,
DaTscan SPECT Striatal Binding Ratios (SBR), and demographic variables.
Handles missing data imputation, robust scaling, and clinical phenotype encoding.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import os


class ClinicalTabularPreprocessor:
    # Core standardized feature schema based on PPMI & UCI Parkinson's studies
    NUMERICAL_FEATURES = [
        "age",
        "disease_duration_months",
        "updrs_part_1",              # Non-motor experiences of daily living (0-52)
        "updrs_part_2",              # Motor experiences of daily living (0-52)
        "updrs_part_3",              # Clinician-scored motor examination (0-132)
        "updrs_part_4",              # Motor complications / dyskinesias (0-24)
        "total_updrs",               # Composite UPDRS score
        "moca_score",                # Montreal Cognitive Assessment (0-30)
        "schwab_england_adl",        # Activities of Daily Living percentage (0-100%)
        "datscan_caudate_left",      # DaTscan SBR in left caudate
        "datscan_caudate_right",     # DaTscan SBR in right caudate
        "datscan_putamen_left",      # DaTscan SBR in left putamen
        "datscan_putamen_right",     # DaTscan SBR in right putamen
        "datscan_asymmetry_index",   # Striatal asymmetry ratio
        "csf_alpha_synuclein_pg_ml", # CSF alpha-synuclein concentration
        "csf_total_tau_pg_ml",       # CSF total tau
        "csf_abeta42_pg_ml",         # CSF amyloid beta 42
    ]

    CATEGORICAL_FEATURES = [
        "sex",                       # 'M', 'F'
        "family_history_pd",         # 'Yes', 'No'
        "dominant_side",             # 'Right', 'Left', 'Symmetric'
        "hoehn_yahr_stage",          # '0', '1', '2', '3', '4', '5'
        "genetic_variant",           # 'None', 'LRRK2', 'GBA', 'SNCA'
    ]

    def __init__(self, target_label_col: str = "diagnosis"):
        self.target_label_col = target_label_col
        self.num_stats: Dict[str, Dict[str, float]] = {}
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.feature_names_out: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "ClinicalTabularPreprocessor":
        """
        Fit numerical statistics (mean, std, median) and categorical mappings.
        """
        self.num_stats = {}
        self.cat_mappings = {}
        self.feature_names_out = []

        # 1. Fit numerical features
        for col in self.NUMERICAL_FEATURES:
            if col in df.columns:
                valid_vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(valid_vals) > 0:
                    self.num_stats[col] = {
                        "median": float(valid_vals.median()),
                        "mean": float(valid_vals.mean()),
                        "std": float(valid_vals.std()) if valid_vals.std() > 1e-6 else 1.0,
                        "p1": float(np.percentile(valid_vals, 1)),
                        "p99": float(np.percentile(valid_vals, 99)),
                    }
                else:
                    self.num_stats[col] = {"median": 0.0, "mean": 0.0, "std": 1.0, "p1": 0.0, "p99": 1.0}
            else:
                self.num_stats[col] = {"median": 0.0, "mean": 0.0, "std": 1.0, "p1": 0.0, "p99": 1.0}
            self.feature_names_out.append(col)

        # 2. Fit categorical mappings
        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().astype(str).unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                mapping["<missing>"] = len(mapping)
                self.cat_mappings[col] = mapping
                # One-hot encoded feature column names
                for val in unique_vals:
                    self.feature_names_out.append(f"{col}_{val}")
                self.feature_names_out.append(f"{col}_missing")
            else:
                self.cat_mappings[col] = {"<missing>": 0}
                self.feature_names_out.append(f"{col}_missing")

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Impute missing values, scale numerical features (Z-score + robust clipping),
        and one-hot encode categorical features.
        Returns:
            X: 2D numpy array of shape (N, total_features)
            y: 1D numpy array of labels (if target column present)
        """
        if not self.is_fitted:
            raise ValueError("ClinicalTabularPreprocessor must be fitted before calling transform()")

        N = len(df)
        feature_matrix_list = []

        # 1. Transform numerical features
        for col in self.NUMERICAL_FEATURES:
            stats = self.num_stats[col]
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").fillna(stats["median"]).to_numpy()
            else:
                series = np.full(N, stats["median"])

            # Robust percentile clipping & Z-score scaling
            clipped = np.clip(series, stats["p1"], stats["p99"])
            scaled = (clipped - stats["mean"]) / stats["std"]
            feature_matrix_list.append(scaled.reshape(-1, 1))

        # 2. Transform categorical features (One-hot encoding)
        for col in self.CATEGORICAL_FEATURES:
            mapping = self.cat_mappings[col]
            n_cats = len(mapping)
            one_hot = np.zeros((N, n_cats), dtype=np.float32)

            if col in df.columns:
                for row_idx, val in enumerate(df[col]):
                    val_str = str(val) if pd.notna(val) else "<missing>"
                    cat_idx = mapping.get(val_str, mapping.get("<missing>", 0))
                    one_hot[row_idx, cat_idx] = 1.0
            else:
                missing_idx = mapping.get("<missing>", 0)
                one_hot[:, missing_idx] = 1.0

            feature_matrix_list.append(one_hot)

        X = np.hstack(feature_matrix_list).astype(np.float32)

        # 3. Extract target labels if available
        y = None
        if self.target_label_col in df.columns:
            y_raw = df[self.target_label_col].values
            # Standardize label mapping: 1 for PD (Parkinson's Disease), 0 for HC (Healthy Control)
            label_map = {"PD": 1, "Parkinson": 1, "1": 1, 1: 1, "HC": 0, "Control": 0, "0": 0, 0: 0}
            y = np.array([label_map.get(v, 0) for v in y_raw], dtype=np.int64)

        return X, y

    def transform_single_patient(self, patient_dict: Dict[str, Union[float, str]]) -> np.ndarray:
        """Transform a single patient clinical profile dictionary into a 1D tensor."""
        df_single = pd.DataFrame([patient_dict])
        X, _ = self.transform(df_single)
        return X[0]

    def get_feature_dimension(self) -> int:
        """Return total output feature dimension for the clinical embedding MLP."""
        return len(self.feature_names_out)

    def save_state(self, filepath: str) -> None:
        """Save preprocessor state (statistics and categorical mappings) to JSON."""
        state = {
            "num_stats": self.num_stats,
            "cat_mappings": self.cat_mappings,
            "feature_names_out": self.feature_names_out,
            "is_fitted": self.is_fitted,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> "ClinicalTabularPreprocessor":
        """Load preprocessor state from JSON."""
        with open(filepath, "r") as f:
            state = json.load(f)
        self.num_stats = state["num_stats"]
        self.cat_mappings = state["cat_mappings"]
        self.feature_names_out = state["feature_names_out"]
        self.is_fitted = state["is_fitted"]
        return self
