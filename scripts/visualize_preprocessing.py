"""
Comprehensive Preprocessing Visualization Script.
Generates publication-quality clinical figures:
1. 3D MRI Multi-Planar Slices (Axial, Coronal, Sagittal) comparing raw vs preprocessed.
2. Speech Processing: Waveform -> Voice Activity Trimming -> Log-Mel Spectrogram -> MFCC Heatmap -> Radar Chart of Vocal Tremor.
3. Clinical Biomarker Distributions: UPDRS vs DaTscan SBR & MoCA cognitive correlation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import (
    AudioSpeechPreprocessor,
    MRI3DPreprocessor,
    SyntheticParkinsonsDatasetGenerator,
)


def generate_visual_reports(output_dir: str = "./reports"):
    os.makedirs(output_dir, exist_ok=True)
    generator = SyntheticParkinsonsDatasetGenerator(random_seed=42)

    # ----------------------------------------------------
    # 1. 3D MRI Structural Neuroimaging Preprocessing Plot
    # ----------------------------------------------------
    print("[*] Generating 3D Structural MRI preprocessing visualization...")
    mri_prep = MRI3DPreprocessor(target_shape=(96, 96, 96), apply_skull_strip=True)

    # Generate synthetic PD and HC brain volumes
    mri_raw_pd = generator.generate_synthetic_3d_mri(is_parkinsons=True, shape=(110, 110, 110))
    mri_proc_pd_tensor, slices_pd = mri_prep.preprocess_pipeline(mri_raw_pd)
    mri_proc_pd = mri_proc_pd_tensor[0]

    fig = plt.figure(figsize=(15, 9), facecolor="#0f172a")
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], wspace=0.15, hspace=0.25)

    views = ["axial", "coronal", "sagittal"]
    titles_raw = ["Raw MRI (Axial Slice)", "Raw MRI (Coronal Slice)", "Raw MRI (Sagittal Slice)"]
    titles_proc = ["Preprocessed & Skull-Stripped (Axial)", "Preprocessed (Coronal)", "Preprocessed (Sagittal)"]

    raw_slices = {
        "axial": mri_raw_pd[mri_raw_pd.shape[0] // 2, :, :],
        "coronal": mri_raw_pd[:, mri_raw_pd.shape[1] // 2, :],
        "sagittal": mri_raw_pd[:, :, mri_raw_pd.shape[2] // 2],
    }

    for idx, view in enumerate(views):
        # Raw Top Row
        ax_top = fig.add_subplot(gs[0, idx])
        ax_top.imshow(raw_slices[view], cmap="bone", origin="lower")
        ax_top.set_title(titles_raw[idx], color="#94a3b8", fontsize=12, fontweight="bold", pad=8)
        ax_top.axis("off")

        # Preprocessed Bottom Row
        ax_bot = fig.add_subplot(gs[1, idx])
        im = ax_bot.imshow(slices_pd[view], cmap="magma", origin="lower")
        ax_bot.set_title(titles_proc[idx], color="#38bdf8", fontsize=12, fontweight="bold", pad=8)
        ax_bot.axis("off")

    fig.suptitle("PPMI 3D Structural MRI (sMRI) Preprocessing Pipeline (3D ResNet-50 Ready)", 
                 color="#f8fafc", fontsize=16, fontweight="bold", y=0.98)
    
    mri_fig_path = os.path.join(output_dir, "mri_preprocessing_pipeline.png")
    plt.savefig(mri_fig_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved MRI visualization: {mri_fig_path}")

    # ----------------------------------------------------
    # 2. Audio & Speech Biomarker Preprocessing Plot
    # ----------------------------------------------------
    print("[*] Generating Audio & Speech biomarker preprocessing visualization...")
    audio_prep = AudioSpeechPreprocessor(sample_rate=16000, target_duration=3.0, n_mels=64)

    audio_pd = generator.generate_synthetic_audio(is_parkinsons=True, duration=3.2)
    audio_hc = generator.generate_synthetic_audio(is_parkinsons=False, duration=3.2)

    spec_pd, feats_pd = audio_prep.preprocess_pipeline(audio_pd)
    spec_hc, feats_hc = audio_prep.preprocess_pipeline(audio_hc)

    fig = plt.figure(figsize=(16, 10), facecolor="#0f172a")
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.2, hspace=0.3)

    # A. Raw Waveform comparison
    ax_wave = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 3.0, len(audio_pd[:48000]))
    ax_wave.plot(t[:1600], audio_pd[:1600], color="#f43f5e", label="Parkinson's (Vocal Tremor & Jitter)", alpha=0.9)
    ax_wave.plot(t[:1600], audio_hc[:1600], color="#10b981", label="Healthy Control (Stable Phonation)", alpha=0.7, linestyle="--")
    ax_wave.set_title("Speech Waveform Micro-Structure (100 ms Zoom)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax_wave.set_xlabel("Time (seconds)", color="#94a3b8")
    ax_wave.set_ylabel("Amplitude", color="#94a3b8")
    ax_wave.tick_params(colors="#94a3b8")
    ax_wave.set_facecolor("#1e293b")
    ax_wave.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # B. Log-Mel Spectrogram (PD)
    ax_spec_pd = fig.add_subplot(gs[0, 1])
    im1 = ax_spec_pd.imshow(spec_pd[0], aspect="auto", origin="lower", cmap="viridis")
    ax_spec_pd.set_title("Parkinson's Log-Mel Spectrogram (CNN+BiLSTM Input: 64 Mels)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax_spec_pd.set_xlabel("Time Frames", color="#94a3b8")
    ax_spec_pd.set_ylabel("Mel Frequency Bins", color="#94a3b8")
    ax_spec_pd.tick_params(colors="#94a3b8")
    cbar1 = plt.colorbar(im1, ax=ax_spec_pd)
    cbar1.ax.tick_params(colors="#94a3b8")

    # C. MFCCs Feature Matrix
    ax_spec_hc = fig.add_subplot(gs[1, 0])
    im2 = ax_spec_hc.imshow(spec_hc[0], aspect="auto", origin="lower", cmap="viridis")
    ax_spec_hc.set_title("Healthy Control Log-Mel Spectrogram (Smooth Harmonics)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax_spec_hc.set_xlabel("Time Frames", color="#94a3b8")
    ax_spec_hc.set_ylabel("Mel Frequency Bins", color="#94a3b8")
    ax_spec_hc.tick_params(colors="#94a3b8")
    cbar2 = plt.colorbar(im2, ax=ax_spec_hc)
    cbar2.ax.tick_params(colors="#94a3b8")

    # D. Acoustic Biomarker Radar Comparison
    ax_radar = fig.add_subplot(gs[1, 1], polar=True)
    radar_keys = ["jitter_local", "jitter_rap", "shimmer_local", "shimmer_apq3", "hnr", "rpde"]
    radar_labels = ["Jitter (Loc)", "Jitter (RAP)", "Shimmer (Loc)", "Shimmer (APQ3)", "HNR (dB)", "RPDE"]
    
    vals_pd = [feats_pd[k] for k in radar_keys]
    vals_hc = [feats_hc[k] for k in radar_keys]

    # Normalize to 0-1 scale for radar
    max_vals = [np.maximum(1e-3, max(p, h) * 1.2) for p, h in zip(vals_pd, vals_hc)]
    norm_pd = [p / m for p, m in zip(vals_pd, max_vals)]
    norm_hc = [h / m for h, m in zip(vals_hc, max_vals)]

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    norm_pd += norm_pd[:1]
    norm_hc += norm_hc[:1]
    angles += angles[:1]

    ax_radar.set_facecolor("#1e293b")
    ax_radar.plot(angles, norm_pd, color="#f43f5e", linewidth=2, label="Parkinson's")
    ax_radar.fill(angles, norm_pd, color="#f43f5e", alpha=0.3)
    ax_radar.plot(angles, norm_hc, color="#10b981", linewidth=2, label="Control")
    ax_radar.fill(angles, norm_hc, color="#10b981", alpha=0.3)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, color="#f8fafc", fontsize=10)
    ax_radar.tick_params(colors="#94a3b8")
    ax_radar.set_title("UCI Acoustic Dysphonia Biomarkers", color="#f8fafc", fontsize=12, fontweight="bold", pad=15)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    fig.suptitle("Speech Preprocessing & Acoustic Feature Extraction Pipeline", 
                 color="#f8fafc", fontsize=16, fontweight="bold", y=0.98)

    speech_fig_path = os.path.join(output_dir, "speech_preprocessing_pipeline.png")
    plt.savefig(speech_fig_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved Speech visualization: {speech_fig_path}")

    # ----------------------------------------------------
    # 3. Clinical & Biomarker Correlation & Distribution Plot
    # ----------------------------------------------------
    print("[*] Generating Clinical & Biomarker correlation visualization...")
    df_clin = generator.generate_clinical_cohort(n_samples=120, pd_ratio=0.6)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#0f172a")

    # A. UPDRS-III vs DaTscan SBR Putamen
    ax_scatter = axes[0]
    ax_scatter.set_facecolor("#1e293b")
    pd_mask = df_clin["diagnosis"] == "PD"
    ax_scatter.scatter(df_clin[pd_mask]["datscan_putamen_left"], df_clin[pd_mask]["updrs_part_3"], 
                       color="#f43f5e", label="Parkinson's (PD)", alpha=0.8, s=50)
    ax_scatter.scatter(df_clin[~pd_mask]["datscan_putamen_left"], df_clin[~pd_mask]["updrs_part_3"], 
                       color="#10b981", label="Control (HC)", alpha=0.8, s=50)
    ax_scatter.set_title("DaTscan Putamen SBR vs MDS-UPDRS III Motor Score", color="#f8fafc", fontweight="bold", fontsize=11)
    ax_scatter.set_xlabel("DaTscan Left Putamen SBR", color="#94a3b8")
    ax_scatter.set_ylabel("MDS-UPDRS Part III (Motor)", color="#94a3b8")
    ax_scatter.tick_params(colors="#94a3b8")
    ax_scatter.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # B. Hoehn & Yahr Staging Distribution
    ax_hy = axes[1]
    ax_hy.set_facecolor("#1e293b")
    hy_counts = df_clin["hoehn_yahr_stage"].value_counts().sort_index()
    colors = ["#10b981", "#38bdf8", "#fbbf24", "#f97316", "#ef4444"]
    ax_hy.bar(hy_counts.index.astype(str), hy_counts.values, color=colors[:len(hy_counts)], width=0.6)
    ax_hy.set_title("Hoehn & Yahr Disease Staging Cohort Distribution", color="#f8fafc", fontweight="bold", fontsize=11)
    ax_hy.set_xlabel("Hoehn & Yahr Stage (0: HC, 1-4: PD Stages)", color="#94a3b8")
    ax_hy.set_ylabel("Patient Count", color="#94a3b8")
    ax_hy.tick_params(colors="#94a3b8")

    # C. MoCA Cognitive Score vs Disease Duration
    ax_moca = axes[2]
    ax_moca.set_facecolor("#1e293b")
    ax_moca.scatter(df_clin[pd_mask]["disease_duration_months"], df_clin[pd_mask]["moca_score"], 
                    color="#a855f7", label="PD Cognitive Trajectory", alpha=0.8, s=50)
    ax_moca.axhline(y=26, color="#e2e8f0", linestyle=":", label="Normal Cognition Cutoff (>=26)")
    ax_moca.set_title("MoCA Cognitive Score vs Disease Duration", color="#f8fafc", fontweight="bold", fontsize=11)
    ax_moca.set_xlabel("Disease Duration (Months)", color="#94a3b8")
    ax_moca.set_ylabel("MoCA Score (0-30)", color="#94a3b8")
    ax_moca.tick_params(colors="#94a3b8")
    ax_moca.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    fig.suptitle("PPMI Clinical & Biomarker Feature Space & Cohort Distributions", 
                 color="#f8fafc", fontsize=15, fontweight="bold", y=1.02)

    clin_fig_path = os.path.join(output_dir, "clinical_biomarkers_distribution.png")
    plt.savefig(clin_fig_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[+] Saved Clinical visualization: {clin_fig_path}")

    return {
        "mri_fig": mri_fig_path,
        "speech_fig": speech_fig_path,
        "clinical_fig": clin_fig_path,
    }


if __name__ == "__main__":
    generate_visual_reports()
