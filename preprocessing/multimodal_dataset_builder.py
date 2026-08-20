"""
Unified Multimodal Dataset Builder & Synchronizer.
Integrates Speech (CNN+BiLSTM spectrograms), 3D Structural MRI (3D ResNet-50 tensors),
and Clinical & Biomarker Tabular profiles into synchronized, stratified PyTorch datasets.
Handles missing modality masks and generates comprehensive metadata manifests.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from sklearn.model_selection import StratifiedShuffleSplit

from .audio_speech_preprocessor import AudioSpeechPreprocessor
from .mri_3d_preprocessor import MRI3DPreprocessor
from .clinical_tabular_preprocessor import ClinicalTabularPreprocessor


class MultimodalParkinsonsDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Parkinson's Disease Decision Support.
    Returns:
        speech_spec: (1, 64, 187) Log-Mel Spectrogram for CNN+BiLSTM
        acoustic_vec: (32,) Engineered acoustic biomarkers
        mri_tensor: (1, 96, 96, 96) Volumetric brain scan for 3D ResNet-50
        clinical_vec: (D_clin,) Imputed & normalized clinical features
        modality_mask: (3,) Binary flags [has_speech, has_mri, has_clinical]
        label: Int64 (0: HC, 1: PD)
        subject_id: str
    """

    def __init__(
        self,
        speech_specs: np.ndarray,
        acoustic_feats: np.ndarray,
        mri_tensors: np.ndarray,
        clinical_vecs: np.ndarray,
        modality_masks: np.ndarray,
        labels: np.ndarray,
        subject_ids: List[str],
        metadata_df: Optional[pd.DataFrame] = None,
    ):
        self.speech_specs = torch.tensor(speech_specs, dtype=torch.float32)
        self.acoustic_feats = torch.tensor(acoustic_feats, dtype=torch.float32)
        self.mri_tensors = torch.tensor(mri_tensors, dtype=torch.float32)
        self.clinical_vecs = torch.tensor(clinical_vecs, dtype=torch.float32)
        self.modality_masks = torch.tensor(modality_masks, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = subject_ids
        self.metadata_df = metadata_df

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return {
            "speech_spec": self.speech_specs[idx],
            "acoustic_vec": self.acoustic_feats[idx],
            "mri_tensor": self.mri_tensors[idx],
            "clinical_vec": self.clinical_vecs[idx],
            "modality_mask": self.modality_masks[idx],
            "label": self.labels[idx],
            "subject_id": self.subject_ids[idx],
        }


class MultimodalDatasetBuilder:
    def __init__(
        self,
        audio_preprocessor: Optional[AudioSpeechPreprocessor] = None,
        mri_preprocessor: Optional[MRI3DPreprocessor] = None,
        clinical_preprocessor: Optional[ClinicalTabularPreprocessor] = None,
    ):
        self.audio_prep = audio_preprocessor or AudioSpeechPreprocessor()
        self.mri_prep = mri_preprocessor or MRI3DPreprocessor()
        self.clin_prep = clinical_preprocessor or ClinicalTabularPreprocessor()

    def build_and_preprocess(
        self,
        clinical_csv_path: str,
        audio_dir: str,
        mri_dir: str,
        output_processed_dir: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, MultimodalParkinsonsDataset]:
        """
        End-to-end multimodal pipeline:
        1. Read clinical metadata
        2. Ingest and preprocess all speech audio files
        3. Ingest and preprocess all 3D structural MRI scans
        4. Fit & transform clinical tabular profiles
        5. Build synchronized multimodal tensors with missing-modality masks
        6. Stratified train/val/test split
        7. Save cached processed datasets and manifests
        """
        os.makedirs(output_processed_dir, exist_ok=True)
        df_clinical = pd.read_csv(clinical_csv_path)

        # 1. Fit clinical preprocessor
        self.clin_prep.fit(df_clinical)
        self.clin_prep.save_state(os.path.join(output_processed_dir, "clinical_preprocessor_state.json"))
        clinical_features, labels = self.clin_prep.transform(df_clinical)

        # 2. Process all subjects
        subject_ids = []
        speech_specs_list = []
        acoustic_feats_list = []
        mri_tensors_list = []
        clinical_vecs_list = []
        modality_masks_list = []
        final_labels_list = []

        print(f"[*] Preprocessing multimodal cohort of {len(df_clinical)} subjects...")

        # Determine reference shapes
        ref_dummy_audio = np.zeros(self.audio_prep.target_samples, dtype=np.float32)
        ref_spec = self.audio_prep.compute_log_mel_spectrogram(ref_dummy_audio)
        ref_mri_shape = (1, *self.mri_prep.target_shape)
        ref_clin_dim = self.clin_prep.get_feature_dimension()

        for idx, row in df_clinical.iterrows():
            sub_id = str(row["subject_id"])
            subject_ids.append(sub_id)
            final_labels_list.append(labels[idx] if labels is not None else 0)
            clinical_vecs_list.append(clinical_features[idx])

            # A. Audio Modality
            audio_path_wav = os.path.join(audio_dir, f"{sub_id}.wav")
            has_audio = os.path.exists(audio_path_wav)
            if has_audio:
                try:
                    spec, ac_dict = self.audio_prep.preprocess_pipeline(audio_path_wav)
                    ac_vec = np.array(list(ac_dict.values()), dtype=np.float32)
                except Exception as e:
                    print(f"[!] Warning: Failed to process audio for {sub_id}: {e}")
                    has_audio = False
                    spec = np.zeros_like(ref_spec)
                    ac_vec = np.zeros(26, dtype=np.float32)
            else:
                spec = np.zeros_like(ref_spec)
                ac_vec = np.zeros(26, dtype=np.float32)

            speech_specs_list.append(spec)
            acoustic_feats_list.append(ac_vec)

            # B. 3D MRI Modality
            mri_path_npy = os.path.join(mri_dir, f"{sub_id}.npy")
            mri_path_nii = os.path.join(mri_dir, f"{sub_id}.nii.gz")
            if not os.path.exists(mri_path_nii):
                mri_path_nii = os.path.join(mri_dir, f"{sub_id}.nii")

            has_mri = os.path.exists(mri_path_npy) or os.path.exists(mri_path_nii)
            if has_mri:
                try:
                    target_path = mri_path_npy if os.path.exists(mri_path_npy) else mri_path_nii
                    if target_path.endswith(".npy"):
                        raw_vol = np.load(target_path)
                    else:
                        raw_vol, _ = self.mri_prep.load_nifti(target_path)
                    mri_t, _ = self.mri_prep.preprocess_pipeline(raw_vol)
                except Exception as e:
                    print(f"[!] Warning: Failed to process MRI for {sub_id}: {e}")
                    has_mri = False
                    mri_t = np.zeros(ref_mri_shape, dtype=np.float32)
            else:
                mri_t = np.zeros(ref_mri_shape, dtype=np.float32)

            mri_tensors_list.append(mri_t)

            # Modality Mask: [has_speech, has_mri, has_clinical]
            modality_masks_list.append([1.0 if has_audio else 0.0, 1.0 if has_mri else 0.0, 1.0])

        # Convert to unified NumPy arrays
        all_speech_specs = np.stack(speech_specs_list, axis=0)
        all_acoustic_feats = np.stack(acoustic_feats_list, axis=0)
        all_mri_tensors = np.stack(mri_tensors_list, axis=0)
        all_clinical_vecs = np.stack(clinical_vecs_list, axis=0)
        all_modality_masks = np.array(modality_masks_list, dtype=np.float32)
        all_labels = np.array(final_labels_list, dtype=np.int64)

        # 3. Stratified Train / Val / Test Split
        # Combine diagnosis label and H&Y stage for fine-grained stratification
        strat_key = [
            f"{lbl}_{str(row.get('hoehn_yahr_stage', 0))}"
            for lbl, (_, row) in zip(all_labels, df_clinical.iterrows())
        ]
        # Fallback to pure labels if counts are small
        unique_counts = pd.Series(strat_key).value_counts()
        strat_labels = strat_key if (unique_counts.min() >= 2) else all_labels

        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(sss_test.split(all_clinical_vecs, strat_labels))

        val_relative_size = val_size / (1.0 - test_size)
        strat_train_val = [strat_labels[i] for i in train_val_idx]
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=random_state)
        sub_train_idx, sub_val_idx = next(sss_val.split(all_clinical_vecs[train_val_idx], strat_train_val))

        train_idx = train_val_idx[sub_train_idx]
        val_idx = train_val_idx[sub_val_idx]

        print(f"[*] Stratified Split -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

        datasets = {}
        for split_name, idxs in splits.items():
            ds = MultimodalParkinsonsDataset(
                speech_specs=all_speech_specs[idxs],
                acoustic_feats=all_acoustic_feats[idxs],
                mri_tensors=all_mri_tensors[idxs],
                clinical_vecs=all_clinical_vecs[idxs],
                modality_masks=all_modality_masks[idxs],
                labels=all_labels[idxs],
                subject_ids=[subject_ids[i] for i in idxs],
                metadata_df=df_clinical.iloc[idxs].reset_index(drop=True),
            )
            datasets[split_name] = ds

            # Cache preprocessed split
            split_file = os.path.join(output_processed_dir, f"{split_name}_multimodal.npz")
            np.savez_compressed(
                split_file,
                speech_specs=all_speech_specs[idxs],
                acoustic_feats=all_acoustic_feats[idxs],
                mri_tensors=all_mri_tensors[idxs],
                clinical_vecs=all_clinical_vecs[idxs],
                modality_masks=all_modality_masks[idxs],
                labels=all_labels[idxs],
                subject_ids=[subject_ids[i] for i in idxs],
            )
            print(f"[+] Saved cached split: {split_file}")

        # Save manifest & feature schema
        manifest = {
            "n_total": len(df_clinical),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "shapes": {
                "speech_spec": list(ref_spec.shape),
                "acoustic_feats": [all_acoustic_feats.shape[1]],
                "mri_tensor": list(ref_mri_shape),
                "clinical_vec": [ref_clin_dim],
            },
            "clinical_features": self.clin_prep.feature_names_out,
            "modalities": ["speech_cnn_bilstm", "mri_3d_resnet50", "clinical_tabular"],
        }
        with open(os.path.join(output_processed_dir, "dataset_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        return datasets
