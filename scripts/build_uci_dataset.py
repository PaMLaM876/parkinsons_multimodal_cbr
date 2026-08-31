"""
Build the Multimodal Dataset using REAL acoustic features from the UCI Parkinson's Dataset (ID 174).
This script will pair the 195 real voice samples with synthetic MRI and Clinical placeholders.
"""

import os
import sys
import numpy as np
import json
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    print("Please install ucimlrepo: pip install ucimlrepo")
    sys.exit(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[*] Fetching real dataset from UCI Machine Learning Repository (ID: 174)...")
    parkinsons = fetch_ucirepo(id=174)
    
    # Extract features and targets
    # X has 22 acoustic features (e.g., MDVP:Fo(Hz), MDVP:Jitter(%), HNR, etc.)
    X = parkinsons.data.features.to_numpy(dtype=np.float32)
    y = parkinsons.data.targets.to_numpy(dtype=np.int64).squeeze()
    
    n_subjects = len(X)
    print(f"[+] Loaded {n_subjects} real subjects with 22 acoustic features each.")
    
    # Normalize real acoustic features using standard scaling
    # so neural networks train better
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    
    # ─── Generate Synthetic Placeholders for MRI, Spectrograms, and Clinical ───
    # Since we only have the 22 real acoustic features, we will generate dummy 
    # data for the rest of the modalities so the architecture doesn't break.
    
    # Dummy Spectrograms: (N, 1, 64, 184)
    speech_specs = np.zeros((n_subjects, 1, 64, 184), dtype=np.float32)
    
    # NO SYNTHETIC NOISE ALLOWED: Fill missing modalities with absolute zeros
    mri_tensors = np.zeros((n_subjects, 1, 96, 96, 96), dtype=np.float32)
    clinical_vecs = np.zeros((n_subjects, 37), dtype=np.float32)
    
    # Modality Mask: [has_speech, has_mri, has_clinical]
    # Speech is present (1.0). MRI and Clinical are explicitly MISSING (0.0).
    modality_masks = np.zeros((n_subjects, 3), dtype=np.float32)
    modality_masks[:, 0] = 1.0  # Only Speech is present
    
    subject_ids = [f"UCI_{i:04d}" for i in range(n_subjects)]
    
    # ─── Stratified Train/Val/Test Split ───
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(X_scaled, y))

    val_relative_size = 0.15 / (1.0 - 0.15)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=42)
    sub_train_idx, sub_val_idx = next(sss_val.split(X_scaled[train_val_idx], y[train_val_idx]))

    train_idx = train_val_idx[sub_train_idx]
    val_idx = train_val_idx[sub_val_idx]
    
    print(f"[*] Splitting -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    
    for split_name, idxs in splits.items():
        split_file = os.path.join(OUTPUT_DIR, f"{split_name}_multimodal.npz")
        np.savez_compressed(
            split_file,
            speech_specs=speech_specs[idxs],
            acoustic_feats=X_scaled[idxs],  # The real UCI data!
            mri_tensors=mri_tensors[idxs],
            clinical_vecs=clinical_vecs[idxs],
            modality_masks=modality_masks[idxs],
            labels=y[idxs],
            subject_ids=np.array([subject_ids[i] for i in idxs])
        )
        print(f"[+] Saved {split_file}")
        
    # Save manifest
    manifest = {
        "n_total": n_subjects,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "shapes": {
            "speech_spec": [1, 64, 184],
            "acoustic_feats": [22],
            "mri_tensor": [1, 96, 96, 96],
            "clinical_vec": [37],
        },
        "clinical_features": [f"dummy_{i}" for i in range(37)],
        "modalities": ["speech_cnn_bilstm", "mri_3d_resnet50", "clinical_tabular"],
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n[*] UCI Dataset integration complete!")

if __name__ == "__main__":
    build_dataset()
