# Multimodal Case-Based Clinical Decision Support for Parkinson's Disease

An end-to-end clinical decision support system integrating:
1. **Speech Biomarkers (UCI & PPMI)**: 2D CNN + Bidirectional LSTM with Temporal Attention.
2. **Volumetric Neuroimaging (PPMI 3D Structural MRI)**: 3D ResNet-50 with 3D Grad-CAM explainability.
3. **Clinical & Biomarker Tabular Data (PPMI MDS-UPDRS, MoCA, DaTscan SBR)**: Deep Tabular Embedding MLP.
4. **Multimodal Fusion & Case-Based Reasoning (CBR)**: Gated Multimodal Unit (GMU) and 4R CBR engine retrieving historical patient twins and projecting progression trajectories.

---

## First Review: Dataset Preprocessing Architecture

### 1. Speech Preprocessing (`CNN + BiLSTM` Ready)
- **File**: `preprocessing/audio_speech_preprocessor.py`
- Voice Activity Detection (VAD) & silence trimming
- Log-Mel Spectrogram computation (STFT, 64 Mel filterbanks) -> Shape `(1, 64, Time)`
- Comprehensive UCI Dysphonia features:
  * Jitter variants (Local, RAP, PPQ5, DDP)
  * Shimmer variants (Local, APQ3, APQ5, DDA)
  * Harmonics-to-Noise Ratio (HNR) & Noise-to-Harmonics Ratio (NHR)
  * Pitch / Fundamental Frequency ($F_0$) statistics
  * 13 MFCCs + Delta + Delta-Delta coefficients
  * Non-linear dynamics: RPDE and DFA

### 2. 3D Structural MRI Preprocessing (`3D ResNet-50` Ready)
- **File**: `preprocessing/mri_3d_preprocessor.py`
- NIfTI loading and RAS+ canonical orientation
- Adaptive 3D Otsu thresholding + 3D morphological skull-stripping / brain extraction
- Isometric 3D spatial resampling to standardized `(96, 96, 96)` grid
- Outlier-resistant intensity normalization (1st-99th percentile clipping + Z-score standardization) -> Shape `(1, 96, 96, 96)`
- Multi-planar orthogonal slice extraction (Axial, Coronal, Sagittal)

### 3. Clinical Tabular Preprocessing (PPMI Schema)
- **File**: `preprocessing/clinical_tabular_preprocessor.py`
- Preprocesses MDS-UPDRS (Parts I, II, III Motor Exam, IV), Hoehn & Yahr staging (0-5), MoCA cognitive assessment, Schwab & England ADL %, and DaTscan SPECT SBR ratios (caudate and putamen)
- Automated missing-value imputation (median / mode)
- Robust quantile scaling & categorical one-hot encoding
- Serializable transformer state (`clinical_preprocessor_state.json`)

### 4. Multimodal Synchronizer & PyTorch DataLoader
- **File**: `preprocessing/multimodal_dataset_builder.py`
- Synchronizes multimodal subjects across modalities
- Handles missing modalities via binary presence masks `[has_speech, has_mri, has_clinical]`
- Stratified train (70%), validation (15%), and test (15%) splits
- High-performance caching to compressed `.npz` files and PyTorch `DataLoader` factory

---

## Quickstart Commands

```bash
# 1. Run automated unit & integration tests
python scripts/test_preprocessing.py

# 2. Run full multimodal preprocessing pipeline on 120 subjects
python scripts/run_preprocessing.py --n_synthetic_samples 120

# 3. Generate clinical visualization figures (MRI, Spectrogram, Clinical plots)
python scripts/visualize_preprocessing.py
```

---

## Output Artifacts & Reports
- Processed splits: `data/processed/train_multimodal.npz`, `val_multimodal.npz`, `test_multimodal.npz`
- Dataset metadata manifest: `data/processed/dataset_manifest.json`
- Clinical figures:
  * `reports/mri_preprocessing_pipeline.png` (3D multi-slice raw vs preprocessed)
  * `reports/speech_preprocessing_pipeline.png` (Waveform, Spectrograms, Acoustic radar)
  * `reports/clinical_biomarkers_distribution.png` (UPDRS vs DaTscan SBR, H&Y staging)
