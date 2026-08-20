"""
Unit and Integration Test Suite for Parkinson's Multimodal Preprocessing.
Tests Audio (CNN+BiLSTM spectrograms), 3D MRI (3D ResNet-50 tensors),
Clinical Tabular preprocessors, and PyTorch Multimodal DataLoader integration.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import (
    AudioSpeechPreprocessor,
    MRI3DPreprocessor,
    ClinicalTabularPreprocessor,
    MultimodalDatasetBuilder,
    MultimodalParkinsonsDataset,
    SyntheticParkinsonsDatasetGenerator,
)


class TestMultimodalPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticParkinsonsDatasetGenerator(random_seed=42)

    def test_01_audio_speech_preprocessor(self):
        """Test audio loading, VAD, Log-Mel Spectrogram, and acoustic feature extraction."""
        audio_prep = AudioSpeechPreprocessor(
            sample_rate=16000, target_duration=3.0, n_mels=64, n_fft=1024, hop_length=256
        )

        synthetic_audio = self.generator.generate_synthetic_audio(is_parkinsons=True, duration=3.2)
        self.assertGreater(len(synthetic_audio), 0)

        spectrogram, acoustic_feats = audio_prep.preprocess_pipeline(synthetic_audio)

        # Check tensor dimensions (1, n_mels, time_steps)
        self.assertEqual(spectrogram.ndim, 3)
        self.assertEqual(spectrogram.shape[0], 1)
        self.assertEqual(spectrogram.shape[1], 64)
        self.assertFalse(np.isnan(spectrogram).any(), "Spectrogram contains NaNs")

        # Check engineered acoustic features
        expected_keys = [
            "f0_mean", "f0_std", "jitter_local", "jitter_rap", "jitter_ppq5",
            "shimmer_local", "shimmer_apq3", "shimmer_apq5", "hnr", "nhr", "rpde", "dfa"
        ]
        for key in expected_keys:
            self.assertIn(key, acoustic_feats)
            self.assertFalse(np.isnan(acoustic_feats[key]), f"Feature {key} is NaN")

        print(f"[PASSED] AudioSpeechPreprocessor -> Spectrogram Shape: {spectrogram.shape}, Acoustic Features: {len(acoustic_feats)}")

    def test_02_mri_3d_preprocessor(self):
        """Test 3D MRI skull-stripping, resampling to 96x96x96, and intensity normalization."""
        mri_prep = MRI3DPreprocessor(target_shape=(96, 96, 96), apply_skull_strip=True)

        raw_vol = self.generator.generate_synthetic_3d_mri(is_parkinsons=True, shape=(100, 100, 100))
        mri_tensor, slices = mri_prep.preprocess_pipeline(raw_vol)

        # Check 3D tensor dimensions (1, D, H, W)
        self.assertEqual(mri_tensor.ndim, 4)
        self.assertEqual(mri_tensor.shape, (1, 96, 96, 96))
        self.assertFalse(np.isnan(mri_tensor).any(), "MRI tensor contains NaNs")

        # Check multi-planar orthogonal slices
        self.assertIn("axial", slices)
        self.assertIn("coronal", slices)
        self.assertIn("sagittal", slices)
        self.assertEqual(slices["axial"].shape, (96, 96))

        print(f"[PASSED] MRI3DPreprocessor -> Volumetric Tensor Shape: {mri_tensor.shape}, Slices: {list(slices.keys())}")

    def test_03_clinical_tabular_preprocessor(self):
        """Test clinical tabular imputation, robust scaling, one-hot encoding, and serialization."""
        clin_prep = ClinicalTabularPreprocessor(target_label_col="diagnosis")

        df = self.generator.generate_clinical_cohort(n_samples=50, pd_ratio=0.6)
        
        # Inject deliberate missing values to test imputation robustness
        df.loc[2:5, "updrs_part_3"] = np.nan
        df.loc[10:12, "datscan_putamen_left"] = np.nan
        df.loc[20, "sex"] = np.nan

        clin_prep.fit(df)
        X, y = clin_prep.transform(df)

        self.assertEqual(X.shape[0], len(df))
        self.assertFalse(np.isnan(X).any(), "Clinical feature matrix contains NaNs")
        self.assertEqual(len(y), len(df))

        # Test single patient transform
        sample_patient = df.iloc[0].to_dict()
        vec_single = clin_prep.transform_single_patient(sample_patient)
        self.assertEqual(vec_single.shape[0], clin_prep.get_feature_dimension())

        print(f"[PASSED] ClinicalTabularPreprocessor -> Feature Dim: {X.shape[1]}, Subjects: {X.shape[0]}")

    def test_04_multimodal_dataset_builder_and_dataloader(self):
        """Test end-to-end dataset builder and PyTorch DataLoader batching."""
        test_dir = "./scratch_test_data"
        os.makedirs(test_dir, exist_ok=True)

        # Generate small cohort
        self.generator.export_full_multimodal_dataset(output_dir=test_dir, n_samples=20, pd_ratio=0.5)

        builder = MultimodalDatasetBuilder()
        datasets = builder.build_and_preprocess(
            clinical_csv_path=os.path.join(test_dir, "raw", "clinical_data.csv"),
            audio_dir=os.path.join(test_dir, "raw", "audio"),
            mri_dir=os.path.join(test_dir, "raw", "mri"),
            output_processed_dir=os.path.join(test_dir, "processed"),
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )

        train_ds = datasets["train"]
        self.assertGreater(len(train_ds), 0)

        # Test PyTorch DataLoader batch collation
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        batch = next(iter(train_loader))

        self.assertEqual(batch["speech_spec"].shape[0], 4)
        self.assertEqual(batch["mri_tensor"].shape[0], 4)
        self.assertEqual(batch["clinical_vec"].shape[0], 4)
        self.assertEqual(batch["modality_mask"].shape[0], 4)
        self.assertEqual(batch["label"].shape[0], 4)

        print(f"[PASSED] PyTorch Multimodal DataLoader -> Batch size 4 loaded smoothly with all tensors synchronized!")


if __name__ == "__main__":
    unittest.main()
