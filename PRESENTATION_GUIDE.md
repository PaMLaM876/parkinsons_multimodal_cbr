# Review 1 Presentation Guide & Defense Script
## Multimodal Case-Based Clinical Decision Support for Parkinson's Disease

Use this guide to present your dataset preprocessing implementation to your professor.

---

## 🚀 1. How to Launch the Live Interactive Showcase

Run this command in your terminal before or during the presentation:
```bash
cd C:\Users\Admin\.gemini\antigravity-ide\scratch\parkinsons_multimodal_cbr
python showcase/serve_showcase.py
```
This automatically opens the **Interactive Clinical Preprocessing Studio** (`http://localhost:8080/index.html`) in your browser.

---

## ⏱️ 2. 5-Minute Presentation Outline

### Slide 1: Introduction & Multimodal Motivation (1 min)
* **Goal**: Build an interpretable, multimodal clinical decision support system combining 3 complementary diagnostic windows:
  1. **Speech Biomarkers (UCI & PPMI)**: Non-invasive early detection of vocal tremors and dysarthria.
  2. **3D Structural MRI (PPMI T1w)**: Deep neuroimaging features of midbrain substantia nigra and basal ganglia atrophy.
  3. **Clinical & DaTscan Data (PPMI)**: MDS-UPDRS motor scores, MoCA cognitive scores, and SPECT striatal binding ratios.
* **Review 1 Focus**: End-to-end dataset preprocessing, standardizing raw heterogeneous data into synchronized deep learning tensors.

---

### Slide 2: Speech Preprocessing (`CNN + BiLSTM` Ready) (1.5 min)
* **What we did**:
  1. **Voice Activity Detection (VAD)**: Removed silence at audio start/end; standardized to 3.0-second 16 kHz mono.
  2. **Log-Mel Spectrogram Extraction**: Applied Short-Time Fourier Transform (STFT) with 64 Mel-filterbanks and dynamic range compression $\to$ Tensor shape `(1, 64, 184)`.
  3. **Acoustic Biomarker Calculation**: Computed 42 clinical dysphonia measures matching the UCI dataset:
     - **Jitter (Local, RAP, PPQ5)**: Pitch period perturbation.
     - **Shimmer (Local, APQ3, APQ5)**: Vocal amplitude flutter.
     - **Harmonics-to-Noise Ratio (HNR)**: Glottal breathiness.
     - **13 MFCCs + Delta + Delta-Delta**.
* **Why CNN + BiLSTM**:
  - The **2D CNN** captures spatial spectral energy distributions and formant shifts in the Log-Mel spectrogram.
  - The **BiLSTM** models temporal vocal tremor dynamics (characteristic 4–6 Hz frequency modulation) across time frames.

---

### Slide 3: 3D Structural MRI Preprocessing (`3D ResNet-50` Ready) (1.5 min)
* **What we did**:
  1. **Canonical Orientation**: Reoriented NIfTI scans to standardized RAS+ coordinates.
  2. **3D Otsu & Morphological Skull-Stripping**: Automatically segmented and zeroed out non-brain tissue (skull, scalp, orbital fat) using 3D connected components and morphological closing.
  3. **Isometric 3D Spatial Resampling**: Resampled varying scanner acquisition resolutions into a uniform `(96, 96, 96)` isotropic voxel grid.
  4. **Robust Intensity Normalization**: Applied 1st–99th percentile clipping to eliminate MR scanner bias artifacts, followed by brain-voxel Z-score normalization into `[-1, 1]`.
* **Why 3D ResNet-50**:
  - 3D convolutions preserve volumetric spatial relationships between the midbrain, ventricles, and striatum across axial, coronal, and sagittal planes simultaneously.

---

### Slide 4: Clinical Tabular Preprocessing & Multimodal Synchronization (1 min)
* **What we did**:
  1. Standardized MDS-UPDRS (Parts I–IV), Hoehn & Yahr staging (0–5), MoCA, and DaTscan Putamen/Caudate SBR ratios.
  2. Handled missing data via median/mode imputation and robust quantile scaling $\to$ `37`-dimensional feature vector.
  3. Created **`MultimodalParkinsonsDataset`** with synchronized patient IDs and binary missing-modality masks `[has_speech, has_mri, has_clinical]` for gated fusion.
  4. Performed stratified train (70%), validation (15%), and test (15%) splitting.

---

## ❓ 3. Expected Professor Questions & Strong Answers

### Q1: "Why do you need both spectrograms (CNN+BiLSTM) and engineered features (Jitter/Shimmer)?"
> **Your Answer**: 
> "Engineered acoustic features like Jitter and Shimmer give us clinical interpretability that doctors understand and trust. However, deep representations from the Log-Mel spectrogram via CNN+BiLSTM capture latent non-linear dynamics and harmonic decay patterns that handcrafted equations miss. Combining both gives us maximum accuracy with clinical explainability."

### Q2: "Why 3D ResNet-50 instead of extracting 2D slices with a standard ResNet?"
> **Your Answer**: 
> "Parkinson's pathology involves subcortical structures like the substantia nigra and basal ganglia whose shape and volume changes span all three orthogonal planes (axial, sagittal, coronal). 2D slice approaches lose inter-slice continuity and spatial context. 3D ResNet-50 uses 3D convolutions (`Conv3d`) to analyze the full 3D volumetric morphology of the brain."

### Q3: "In clinical practice, what if a patient doesn't have an MRI scan or voice recording?"
> **Your Answer**: 
> "Our preprocessing pipeline explicitly generates a binary modality presence mask `[has_speech, has_mri, has_clinical]` for each patient. In the upcoming multimodal fusion module, we use a Gated Multimodal Unit (GMU) with modality dropout training so the system operates reliably even when one or two modalities are missing."

### Q4: "How does Case-Based Reasoning (CBR) fit into this after preprocessing?"
> **Your Answer**: 
> "Once the encoders produce latent vectors for each modality ($\mathbf{e}_{\text{speech}}, \mathbf{e}_{\text{mri}}, \mathbf{e}_{\text{clinical}}$), the multimodal fusion creates a joint patient representation vector $\mathbf{z}$. The CBR engine then indexes verified historical PPMI patient cases. When a new patient arrives, the CBR engine retrieves the top-$k$ nearest historical 'patient twins', explains the diagnosis based on real past cases, and projects the patient's multi-year motor progression trajectory."

---

## 🖥️ 4. Interactive Live Demo Steps (During Meeting)

1. **Open the Showcase Dashboard** (`http://localhost:8080/index.html`).
2. **Switch Patients** using the dropdown at the top right:
   - Select **`PPMI_1001` (Moderate PD)** $\to$ Show elevated Jitter (1.84%), high UPDRS (58), and DaTscan putaminal deficit (0.74).
   - Select **`PPMI_1080` (Healthy Control)** $\to$ Show normal Jitter (0.32%), low UPDRS (4), and healthy DaTscan uptake (2.55).
3. **Show 3D MRI Slicing**:
   - Move the slice slider from 0 to 95 to show volumetric depth.
   - Click the **Axial**, **Coronal**, and **Sagittal** buttons to show multi-planar consistency and the skull-stripped brain parenchymal boundary.
4. **Play Speech Phonation**:
   - Click the **▶ Play** button on the speech card to demonstrate the live spectrogram tremor modulation.
5. **Show the Tensor Verification Box**:
   - Point out the clean PyTorch tensor shapes: `[1, 96, 96, 96]` for 3D ResNet-50, `[1, 64, 184]` for CNN+BiLSTM, and `[37]` for the tabular MLP.
