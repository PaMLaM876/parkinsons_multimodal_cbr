"""
Synthetic PPMI & UCI Parkinson's Multimodal Cohort Generator.
Generates physiologically and acoustically realistic multimodal data:
1. Clinical tabular records (UPDRS, MoCA, DaTscan SBR, H&Y staging)
2. Speech audio signals (.wav) with phonatory tremors, jitter, shimmer & dysarthria
3. 3D Structural MRI brain volumes with substantia nigra/basal ganglia atrophy
"""

import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import os
from typing import Dict, List, Tuple


class SyntheticParkinsonsDatasetGenerator:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def generate_clinical_cohort(self, n_samples: int = 120, pd_ratio: float = 0.6) -> pd.DataFrame:
        """
        Generate synthetic clinical tabular records matching PPMI & UCI clinical cohorts.
        Includes physiological correlations (e.g. lower DaTscan SBR correlates with higher UPDRS-III).
        """
        records = []
        n_pd = int(n_samples * pd_ratio)
        n_hc = n_samples - n_pd

        for i in range(n_samples):
            is_pd = 1 if i < n_pd else 0
            subject_id = f"PPMI_{1000 + i:04d}"

            # Demographics
            age = int(self.rng.normal(64.5 if is_pd else 62.0, 7.5))
            age = np.clip(age, 45, 85)
            sex = self.rng.choice(["M", "F"], p=[0.62, 0.38])
            family_history = self.rng.choice(["Yes", "No"], p=[0.25 if is_pd else 0.08, 0.75 if is_pd else 0.92])
            dominant_side = self.rng.choice(["Right", "Left", "Symmetric"], p=[0.55, 0.35, 0.10] if is_pd else [0.45, 0.45, 0.10])
            genetic = self.rng.choice(["None", "LRRK2", "GBA", "SNCA"], p=[0.85, 0.08, 0.05, 0.02] if is_pd else [0.98, 0.01, 0.01, 0.00])

            if is_pd:
                disease_dur = int(self.rng.gamma(shape=2.5, scale=12.0)) # 6 to 60 months
                hy_stage = int(self.rng.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.22, 0.08]))

                # UPDRS scores scale with disease severity (H&Y)
                updrs_1 = int(self.rng.normal(8.0 + hy_stage * 2.2, 3.5))
                updrs_2 = int(self.rng.normal(12.0 + hy_stage * 3.8, 4.5))
                updrs_3 = int(self.rng.normal(24.0 + hy_stage * 7.5, 6.5)) # Motor examination
                updrs_4 = int(self.rng.normal(2.0 + hy_stage * 1.5, 1.8))
                moca = int(self.rng.normal(26.5 - hy_stage * 0.8, 2.2))
                schwab_england = int(100 - hy_stage * 10 - self.rng.uniform(0, 10))

                # DaTscan SBR: marked putaminal and caudate reduction
                putamen_left = float(np.clip(self.rng.normal(0.85 - hy_stage * 0.12, 0.18), 0.25, 1.6))
                putamen_right = float(np.clip(self.rng.normal(0.95 - hy_stage * 0.10, 0.20), 0.30, 1.8))
                caudate_left = float(np.clip(self.rng.normal(1.75 - hy_stage * 0.15, 0.25), 0.80, 2.5))
                caudate_right = float(np.clip(self.rng.normal(1.90 - hy_stage * 0.14, 0.28), 0.90, 2.7))
                asymmetry = float(np.abs(putamen_left - putamen_right) / (putamen_left + putamen_right + 1e-6))

                csf_asyn = float(self.rng.normal(1420.0 - hy_stage * 80.0, 250.0))
                csf_tau = float(self.rng.normal(160.0 + hy_stage * 20.0, 35.0))
                csf_abeta = float(self.rng.normal(720.0 - hy_stage * 30.0, 95.0))
            else:
                disease_dur = 0
                hy_stage = 0
                updrs_1 = int(self.rng.normal(2.0, 1.5))
                updrs_2 = int(self.rng.normal(1.5, 1.2))
                updrs_3 = int(self.rng.normal(3.5, 2.0))
                updrs_4 = 0
                moca = int(self.rng.normal(28.8, 1.1))
                schwab_england = 100

                # DaTscan SBR: normal healthy uptake (> 2.0 in putamen, > 2.8 in caudate)
                putamen_left = float(self.rng.normal(2.45, 0.25))
                putamen_right = float(self.rng.normal(2.50, 0.24))
                caudate_left = float(self.rng.normal(3.15, 0.30))
                caudate_right = float(self.rng.normal(3.20, 0.29))
                asymmetry = float(np.abs(putamen_left - putamen_right) / (putamen_left + putamen_right + 1e-6))

                csf_asyn = float(self.rng.normal(1850.0, 220.0))
                csf_tau = float(self.rng.normal(135.0, 25.0))
                csf_abeta = float(self.rng.normal(880.0, 80.0))

            updrs_1 = int(np.clip(updrs_1, 0, 52))
            updrs_2 = int(np.clip(updrs_2, 0, 52))
            updrs_3 = int(np.clip(updrs_3, 0, 132))
            updrs_4 = int(np.clip(updrs_4, 0, 24))
            total_updrs = updrs_1 + updrs_2 + updrs_3 + updrs_4
            moca = int(np.clip(moca, 10, 30))
            schwab_england = int(np.clip(schwab_england, 20, 100))

            records.append({
                "subject_id": subject_id,
                "diagnosis": "PD" if is_pd else "HC",
                "is_parkinsons": is_pd,
                "age": age,
                "sex": sex,
                "disease_duration_months": disease_dur,
                "family_history_pd": family_history,
                "dominant_side": dominant_side,
                "genetic_variant": genetic,
                "hoehn_yahr_stage": str(hy_stage),
                "updrs_part_1": updrs_1,
                "updrs_part_2": updrs_2,
                "updrs_part_3": updrs_3,
                "updrs_part_4": updrs_4,
                "total_updrs": total_updrs,
                "moca_score": moca,
                "schwab_england_adl": schwab_england,
                "datscan_caudate_left": round(caudate_left, 3),
                "datscan_caudate_right": round(caudate_right, 3),
                "datscan_putamen_left": round(putamen_left, 3),
                "datscan_putamen_right": round(putamen_right, 3),
                "datscan_asymmetry_index": round(asymmetry, 3),
                "csf_alpha_synuclein_pg_ml": round(csf_asyn, 1),
                "csf_total_tau_pg_ml": round(csf_tau, 1),
                "csf_abeta42_pg_ml": round(csf_abeta, 1),
            })

        return pd.DataFrame(records)

    def generate_synthetic_audio(
        self,
        is_parkinsons: bool,
        duration: float = 3.0,
        sample_rate: int = 16000,
        base_f0: float = 180.0,
    ) -> np.ndarray:
        """
        Synthesize acoustic waveform of sustained vowel /a/ phonation.
        For Parkinson's cases, injects realistic phonatory micro-tremor (4-7 Hz),
        increased jitter (phase perturbations), shimmer (amplitude flutter), and turbulence noise.
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Baseline pitch with small natural drift
        if is_parkinsons:
            # Vocal tremor frequency modulation (5 Hz physiological tremor)
            tremor_freq = 5.2
            tremor_depth = 0.045  # 4.5% pitch modulation
            pitch_track = base_f0 * (1.0 + tremor_depth * np.sin(2 * np.pi * tremor_freq * t))

            # Additive high-frequency jitter (random pitch period perturbations)
            jitter_noise = self.rng.normal(0, 0.015, n_samples)
            phase = np.cumsum(2 * np.pi * (pitch_track * (1.0 + jitter_noise)) / sample_rate)

            # Shimmer amplitude modulation (tremor + random noise)
            shimmer_mod = 1.0 + 0.12 * np.sin(2 * np.pi * 5.0 * t) + self.rng.normal(0, 0.05, n_samples)
            shimmer_mod = np.clip(shimmer_mod, 0.3, 1.8)

            # Harmonics synthesis (vowel /a/ formants around 700Hz, 1200Hz, 2600Hz)
            h1 = np.sin(phase)
            h2 = 0.65 * np.sin(2 * phase)
            h3 = 0.45 * np.sin(3 * phase)
            h4 = 0.30 * np.sin(4 * phase)
            h5 = 0.18 * np.sin(5 * phase)

            harmonic_signal = (h1 + h2 + h3 + h4 + h5) * shimmer_mod

            # Additive turbulent breath noise (reduced HNR)
            breath_noise = self.rng.normal(0, 0.08, n_samples)
            audio = harmonic_signal + breath_noise
        else:
            # Healthy steady phonation
            jitter_noise = self.rng.normal(0, 0.002, n_samples)
            phase = np.cumsum(2 * np.pi * (base_f0 * (1.0 + jitter_noise)) / sample_rate)
            shimmer_mod = 1.0 + self.rng.normal(0, 0.01, n_samples)

            h1 = np.sin(phase)
            h2 = 0.70 * np.sin(2 * phase)
            h3 = 0.40 * np.sin(3 * phase)
            h4 = 0.25 * np.sin(4 * phase)
            h5 = 0.12 * np.sin(5 * phase)

            harmonic_signal = (h1 + h2 + h3 + h4 + h5) * shimmer_mod
            breath_noise = self.rng.normal(0, 0.015, n_samples)
            audio = harmonic_signal + breath_noise

        # Envelope shaping (smooth onset and offset fade)
        fade_len = int(0.05 * sample_rate)
        envelope = np.ones(n_samples)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        audio = audio * envelope

        # Normalize peak amplitude
        audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.90
        return audio.astype(np.float32)

    def generate_synthetic_3d_mri(
        self,
        is_parkinsons: bool,
        shape: Tuple[int, int, int] = (96, 96, 96),
        noise_level: float = 0.04,
    ) -> np.ndarray:
        """
        Generate anatomically structured 3D brain volume tensor.
        Simulates:
        - Skull & Scalp boundary
        - Cerebrospinal Fluid (CSF) in ventricles and sulci
        - Grey Matter (Cerebral Cortex & Subcortical Nuclei)
        - White Matter (Corpus Callosum, Internal Capsule)
        - In PD cases: Midbrain substantia nigra & putaminal signal alterations / atrophy
        """
        D, H, W = shape
        cz, cy, cx = D / 2.0, H / 2.0, W / 2.0
        z, y, x = np.ogrid[:D, :H, :W]

        volume = np.zeros(shape, dtype=np.float32)

        # 1. Skull / Head Ellipsoid (r_z=42, r_y=40, r_x=36)
        r_head = ((z - cz) / 42.0) ** 2 + ((y - cy) / 40.0) ** 2 + ((x - cx) / 36.0) ** 2
        skull_mask = (r_head <= 1.0) & (r_head > 0.88)
        volume[skull_mask] = self.rng.uniform(0.75, 0.95, size=np.sum(skull_mask))

        # 2. Brain Parenchyma (Cerebrum)
        brain_mask = r_head <= 0.86
        # White Matter core
        r_wm = ((z - cz) / 28.0) ** 2 + ((y - cy) / 26.0) ** 2 + ((x - cx) / 22.0) ** 2
        wm_mask = r_wm <= 0.70
        volume[brain_mask] = 0.55  # Grey matter baseline
        volume[wm_mask] = 0.82     # White matter baseline (high intensity in T1w)

        # 3. Lateral Ventricles (CSF - low intensity in T1w)
        # Symmetrical butterfly-like central CSF spaces
        vent_left = ((z - cz) / 12.0) ** 2 + ((y - (cy - 3)) / 16.0) ** 2 + ((x - (cx - 6)) / 4.0) ** 2 <= 0.8
        vent_right = ((z - cz) / 12.0) ** 2 + ((y - (cy - 3)) / 16.0) ** 2 + ((x - (cx + 6)) / 4.0) ** 2 <= 0.8
        volume[vent_left | vent_right] = 0.15

        # 4. Basal Ganglia & Substantia Nigra (Midbrain Region)
        # Located around central lower midbrain z in [cz-8, cz-2], y in [cy-4, cy+4], x in [cx-10, cx+10]
        sn_left = ((z - (cz - 5)) / 4.0) ** 2 + ((y - cy) / 5.0) ** 2 + ((x - (cx - 7)) / 4.0) ** 2 <= 0.9
        sn_right = ((z - (cz - 5)) / 4.0) ** 2 + ((y - cy) / 5.0) ** 2 + ((x - (cx + 7)) / 4.0) ** 2 <= 0.9

        if is_parkinsons:
            # Neurodegeneration: iron accumulation & volume loss causing reduced T1w contrast/atrophy
            volume[sn_left] = 0.38
            volume[sn_right] = 0.40
            # Ventricles slightly enlarged (compensatory hydrocephalus/atrophy)
            vent_enlarge_left = ((z - cz) / 14.0) ** 2 + ((y - (cy - 3)) / 18.0) ** 2 + ((x - (cx - 7)) / 5.0) ** 2 <= 0.9
            vent_enlarge_right = ((z - cz) / 14.0) ** 2 + ((y - (cy - 3)) / 18.0) ** 2 + ((x - (cx + 7)) / 5.0) ** 2 <= 0.9
            volume[vent_enlarge_left | vent_enlarge_right] = 0.15
        else:
            volume[sn_left] = 0.62
            volume[sn_right] = 0.62

        # Add Gaussian noise
        noise = self.rng.normal(0, noise_level, size=shape)
        volume = np.clip(volume + noise * (volume > 0.05), 0.0, 1.0)

        return volume.astype(np.float32)

    def export_full_multimodal_dataset(
        self, output_dir: str, n_samples: int = 120, pd_ratio: float = 0.6
    ) -> pd.DataFrame:
        """
        Generate and persist a complete multimodal cohort with synced patient IDs:
        - raw/audio/{subject_id}.wav
        - raw/mri/{subject_id}.npy (and .nii format)
        - raw/clinical_data.csv
        """
        os.makedirs(os.path.join(output_dir, "raw", "audio"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "raw", "mri"), exist_ok=True)

        df_clinical = self.generate_clinical_cohort(n_samples=n_samples, pd_ratio=pd_ratio)
        csv_path = os.path.join(output_dir, "raw", "clinical_data.csv")
        df_clinical.to_csv(csv_path, index=False)

        for _, row in df_clinical.iterrows():
            sub_id = row["subject_id"]
            is_pd = bool(row["is_parkinsons"])

            # 1. Generate & Save Audio (.wav)
            audio = self.generate_synthetic_audio(is_parkinsons=is_pd)
            audio_path = os.path.join(output_dir, "raw", "audio", f"{sub_id}.wav")
            sf.write(audio_path, audio, 16000)

            # 2. Generate & Save 3D MRI (.npy volume)
            mri_vol = self.generate_synthetic_3d_mri(is_parkinsons=is_pd, shape=(96, 96, 96))
            mri_path = os.path.join(output_dir, "raw", "mri", f"{sub_id}.npy")
            np.save(mri_path, mri_vol)

        return df_clinical
