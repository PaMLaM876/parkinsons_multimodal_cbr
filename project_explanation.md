# Parkinson's Disease Prediction — How It Works, What You've Done, & What's Next

---

## 1. How Are We Predicting Parkinson's?

### Prediction Method: **Multimodal Classification (PD vs HC)**

Your system predicts **binary classification**: **PD (Parkinson's Disease) = 1** vs **HC (Healthy Control) = 0**.

It does NOT rely on a single statistic. Instead, it fuses **three complementary modalities** to make the prediction:

| Modality | What It Captures | Model Architecture | Key Biomarkers |
|---|---|---|---|
| **Speech / Voice** | Vocal tremors, dysarthria, glottal noise | 2D CNN + Bidirectional LSTM | Jitter, Shimmer, HNR, MFCCs, RPDE, DFA |
| **3D Brain MRI** | Neurodegeneration in substantia nigra & basal ganglia | 3D ResNet-50 | Volumetric atrophy patterns, ventricular enlargement |
| **Clinical Tabular** | Motor & cognitive severity scores | Deep Tabular MLP | MDS-UPDRS (I-IV), DaTscan SBR, MoCA, H&Y stage |

### The Core Statistics That Differentiate PD from HC

> [!IMPORTANT]
> These are the key discriminative biomarkers your preprocessing extracts:

**Speech biomarkers** (higher in PD → voice deterioration):
- **Jitter (Local)**: PD patients have higher pitch instability (~1.8% vs ~0.3%)
- **Shimmer (Local)**: Higher amplitude flutter in PD
- **HNR (Harmonics-to-Noise Ratio)**: Lower in PD (more breath noise / hoarseness)
- **NHR**: Higher in PD (inverse of HNR)
- **RPDE / DFA**: Non-linear vocal dynamics → more irregular in PD

**DaTscan SBR** (lower in PD → dopamine loss):
- **Putamen SBR**: PD ≈ 0.7–0.9 vs HC ≈ 2.4–2.5 (dramatic reduction)
- **Caudate SBR**: PD ≈ 1.5–1.7 vs HC ≈ 3.1–3.2

**Motor scores** (higher in PD → motor impairment):
- **UPDRS Part III (Motor Exam)**: PD ≈ 30–50 vs HC ≈ 2–5
- **H&Y Stage**: PD = 1–4, HC = 0

### How the Final Prediction Will Work (After Model Is Built)

```
Speech Audio → CNN+BiLSTM → e_speech (latent vector)
3D MRI Scan  → 3D ResNet-50 → e_mri (latent vector)
Clinical Data → Tabular MLP → e_clinical (latent vector)
                    ↓
            Gated Multimodal Unit (GMU)
                    ↓
            Joint Embedding z
                    ↓
        ┌───────────────┬────────────────┐
        │  Classification│  CBR Engine    │
        │  PD vs HC      │  Patient Twins │
        │  (Softmax)     │  + Trajectory  │
        └───────────────┴────────────────┘
```

---

## 2. What You've Done So Far (Review 1 — Preprocessing)

### Files to Show Your Teacher

#### 🔊 Speech Preprocessing
**Primary file**: [`audio_speech_preprocessor.py`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py)

| Step | What It Does | Code Lines |
|---|---|---|
| Audio Loading & Resampling | Loads WAV, converts stereo→mono, resamples to 16kHz | [L71-96](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py#L71-L96) |
| Voice Activity Detection (VAD) | Removes leading/trailing silence using RMS energy thresholding | [L98-118](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py#L98-L118) |
| Pad / Truncate | Standardizes all audio to exactly 3 seconds (48,000 samples) | [L120-129](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py#L120-L129) |
| Log-Mel Spectrogram | Pre-emphasis → STFT → 64 Mel filterbanks → log compression → Z-score → shape `(1, 64, T)` | [L131-165](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py#L131-L165) |
| Mel Filterbank Construction | Hand-built triangular Mel filter matrix (Hz → Mel → Hz) | [L38-69](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py#L38-L69) |
| Acoustic Feature Extraction | F0 pitch tracking, Jitter (local/RAP/PPQ5/DDP), Shimmer (local/APQ3/APQ5/DDA), HNR, NHR, 13 MFCCs, RPDE, DFA | [L167-300](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py#L167-L300) |
| Complete Pipeline | End-to-end: load → VAD → pad → spectrogram + features | [L302-318](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/audio_speech_preprocessor.py#L302-L318) |

#### 🧠 MRI Preprocessing
**Primary file**: [`mri_3d_preprocessor.py`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/mri_3d_preprocessor.py)

| Step | What It Does | Code Lines |
|---|---|---|
| NIfTI Loading + RAS+ Reorientation | Loads `.nii.gz` files, reorients to canonical RAS+ coordinates | [L25-50](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/mri_3d_preprocessor.py#L25-L50) |
| 3D Otsu Skull-Stripping | Adaptive thresholding → morphological opening → largest connected component → closing + hole-filling | [L52-100](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/mri_3d_preprocessor.py#L52-L100) |
| 3D Isometric Resampling | Resamples any scanner resolution to uniform `(96, 96, 96)` voxel grid using spline interpolation | [L102-117](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/mri_3d_preprocessor.py#L102-L117) |
| Robust Intensity Normalization | 1st–99th percentile clipping → Z-score → scale to `[-1, 1]` | [L119-145](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/mri_3d_preprocessor.py#L119-L145) |
| Multi-Planar Slice Extraction | Axial, Coronal, Sagittal central slices for visual inspection | [L147-169](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/mri_3d_preprocessor.py#L147-L169) |
| Complete Pipeline | End-to-end: load → skull-strip → resample → normalize → `(1, 96, 96, 96)` tensor | [L171-200](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/mri_3d_preprocessor.py#L171-L200) |

#### 📋 Clinical Tabular Preprocessing
**Primary file**: [`clinical_tabular_preprocessor.py`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/clinical_tabular_preprocessor.py)

| Step | What It Does | Code Lines |
|---|---|---|
| Feature Schema | 17 numerical features (UPDRS, MoCA, DaTscan SBR, CSF markers) + 5 categorical (sex, H&Y, genetics) | [L17-43](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/clinical_tabular_preprocessor.py#L17-L43) |
| Fit (Stats Calculation) | Computes median, mean, std, 1st/99th percentiles for each numerical feature; builds categorical mappings | [L52-94](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/clinical_tabular_preprocessor.py#L52-L94) |
| Transform (Impute + Scale + Encode) | Median imputation → percentile clipping → Z-score scaling → one-hot encoding → label mapping (PD=1, HC=0) | [L96-150](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/clinical_tabular_preprocessor.py#L96-L150) |
| Serializable State | Save/load preprocessor state as JSON for reproducibility | [L162-182](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/clinical_tabular_preprocessor.py#L162-L182) |

#### 🔗 Multimodal Synchronization & Dataset
**Primary file**: [`multimodal_dataset_builder.py`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/multimodal_dataset_builder.py)

| Step | What It Does | Code Lines |
|---|---|---|
| PyTorch Dataset | Returns speech spec, acoustic vec, MRI tensor, clinical vec, modality mask, label, subject ID | [L22-67](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/multimodal_dataset_builder.py#L22-L67) |
| Missing Modality Masks | Binary `[has_speech, has_mri, has_clinical]` flags for gated fusion | [L175-176](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/multimodal_dataset_builder.py#L175-L176) |
| Stratified Splitting | 70% train / 15% val / 15% test with diagnosis+H&Y stratification | [L186-207](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/multimodal_dataset_builder.py#L186-L207) |
| Cached .npz Output | Saves compressed splits + dataset manifest JSON | [L229-259](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/preprocessing/multimodal_dataset_builder.py#L229-L259) |

### Output Artifacts Already Generated

| Artifact | Path |
|---|---|
| Processed train split | [`data/processed/train_multimodal.npz`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/data/processed/train_multimodal.npz) (62 MB) |
| Processed val split | [`data/processed/val_multimodal.npz`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/data/processed/val_multimodal.npz) (13 MB) |
| Processed test split | [`data/processed/test_multimodal.npz`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/data/processed/test_multimodal.npz) (13 MB) |
| Preprocessor state | [`data/processed/clinical_preprocessor_state.json`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/data/processed/clinical_preprocessor_state.json) |
| Dataset manifest | [`data/processed/dataset_manifest.json`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/data/processed/dataset_manifest.json) |
| MRI pipeline figure | [`reports/mri_preprocessing_pipeline.png`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/reports/mri_preprocessing_pipeline.png) |
| Speech pipeline figure | [`reports/speech_preprocessing_pipeline.png`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/reports/speech_preprocessing_pipeline.png) |
| Clinical distributions | [`reports/clinical_biomarkers_distribution.png`](file:///c:/Users/Admin/Desktop/parkinsons_multimodal_cbr/reports/clinical_biomarkers_distribution.png) |

---

## 3. Preprocessing Explained Simply

### 🔊 Speech Preprocessing (for the teacher)

> **Raw input**: A WAV audio recording of a patient saying "aaaaah" (sustained vowel phonation)
> **Final output**: A 2D spectrogram tensor `(1, 64, T)` + 42 acoustic biomarker numbers

**Pipeline step-by-step:**
1. **Load audio** → Read the WAV file, convert stereo to mono, resample to 16,000 Hz
2. **VAD (Voice Activity Detection)** → Detect where actual voice starts/ends, remove silence
3. **Pad/Truncate** → Make every recording exactly 3 seconds long (center-pad shorter ones, center-crop longer ones)
4. **Pre-emphasis** → Boost high frequencies (helps detect vocal cord issues)
5. **STFT** → Slide a 64ms window across the audio with 16ms hops, compute frequency spectrum at each window
6. **Mel Filterbank** → Convert the frequency axis from linear Hz to perceptual Mel scale using 64 triangular filters (mimics human hearing)
7. **Log compression** → Take log₁₀ to compress dynamic range (quieter sounds become more visible)
8. **Z-score normalization** → Zero mean, unit variance for neural network training
9. **Acoustic features** → Separately compute Jitter (pitch instability), Shimmer (amplitude instability), HNR (voice quality), MFCCs (spectral shape), F0 (fundamental frequency), RPDE, DFA

### 🧠 MRI Preprocessing (for the teacher)

> **Raw input**: A 3D NIfTI brain scan (.nii.gz) from an MRI scanner with arbitrary resolution
> **Final output**: A 3D tensor `(1, 96, 96, 96)` with values in `[-1, 1]`

**Pipeline step-by-step:**
1. **Load NIfTI** → Read the 3D volume from `.nii.gz` file using nibabel
2. **RAS+ Reorientation** → Rotate the volume so axes are always Right-to-left, Anterior-to-posterior, Superior-to-inferior (standardized across different scanners)
3. **Skull-Stripping** using 3D Otsu:
   - Compute intensity histogram of non-background voxels
   - Find optimal threshold using Otsu's method (maximizes between-class variance)
   - Create binary mask → morphological opening (remove thin skull bridges)
   - Keep only the largest 3D connected component (the brain)
   - Morphological closing + hole-filling (include ventricles & deep nuclei)
4. **3D Resampling** → Resize from scanner's native resolution (e.g., 256×256×180) to uniform `96×96×96` using 2nd-order spline interpolation
5. **Intensity Normalization**:
   - Clip at 1st and 99th percentiles (remove extreme outlier voxels from scanner artifacts)
   - Z-score on brain voxels only (zero mean, unit variance)
   - Rescale to `[-1, 1]` range for 3D ResNet input
6. **Multi-planar slices** → Extract central axial, coronal, sagittal slices for visual QA

---

## 4. What to Implement for the Next Review

> [!IMPORTANT]
> The next review should show progress on **model architecture** and **training**. Here's what to implement:

### Priority 1: Individual Modality Encoders
- [ ] **Speech Encoder**: 2D CNN (3–4 conv layers) → Bidirectional LSTM (2 layers) → Temporal Attention → `e_speech` (128-dim or 256-dim latent vector)
- [ ] **MRI Encoder**: 3D ResNet-50 (using 3D Conv layers) → Global Average Pooling → FC → `e_mri` (256-dim)
- [ ] **Clinical Encoder**: Tabular Embedding MLP (37 features → 128 → 64 → `e_clinical`)

### Priority 2: Multimodal Fusion
- [ ] **Gated Multimodal Unit (GMU)**: Learns which modalities to weight dynamically. Uses sigmoid gating: `z = σ(W_g · [e_speech; e_mri; e_clinical]) ⊙ e_fused`
- [ ] **Modality Dropout**: During training, randomly zero out modality embeddings so the model handles missing data at inference

### Priority 3: Training Loop
- [ ] Binary cross-entropy loss for PD vs HC classification
- [ ] DataLoader integration with the cached `.npz` files from Review 1
- [ ] Training metrics: Accuracy, Sensitivity, Specificity, AUC-ROC, F1
- [ ] Train/val loss curves visualization

### Priority 4 (Stretch): CBR Engine
- [ ] Build a case library from the fused embeddings
- [ ] k-NN retrieval for "patient twins" (most similar historical cases)
- [ ] Trajectory projection using retrieved cases' longitudinal data

### What This Shows Your Teacher
The teacher will see you've gone from **raw heterogeneous data → standardized tensors** (Review 1) to **tensors → trained neural network predictions** (Review 2). This demonstrates a clear progression through the entire ML pipeline.

---

## 5. Quick Reference: File Map

```
parkinsons_multimodal_cbr/
├── preprocessing/                          ← ✅ DONE (Review 1)
│   ├── audio_speech_preprocessor.py        ← Speech VAD, Mel spectrogram, Jitter/Shimmer
│   ├── mri_3d_preprocessor.py              ← NIfTI loading, skull-strip, resample, normalize
│   ├── clinical_tabular_preprocessor.py    ← UPDRS/DaTscan imputation, Z-score, one-hot
│   ├── multimodal_dataset_builder.py       ← Sync + split + PyTorch Dataset
│   └── synthetic_data_generator.py         ← Realistic synthetic cohort generation
├── scripts/
│   ├── run_preprocessing.py                ← Run full pipeline
│   ├── test_preprocessing.py               ← Unit tests
│   ├── visualize_preprocessing.py          ← Generate report figures
│   ├── preprocess_real_data.py             ← UCI real dataset analysis
│   └── download_uci_datasets.py            ← Download UCI Speech + Telemonitoring
├── data/
│   ├── processed/                          ← ✅ train/val/test .npz + manifest
│   └── real/                               ← UCI Parkinson's datasets
├── reports/                                ← ✅ Pipeline visualization PNGs
├── showcase/                               ← Interactive HTML demo
├── README.md                               ← Project overview
└── PRESENTATION_GUIDE.md                   ← Review 1 defense script
```
