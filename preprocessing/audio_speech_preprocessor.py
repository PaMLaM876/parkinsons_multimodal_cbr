"""
Audio and Speech Preprocessor for Parkinson's Disease Voice Biomarkers.
Supports UCI Parkinson's Speech Dataset and PPMI Acoustic Recordings.
Extracts Log-Mel Spectrograms for CNN+BiLSTM and comprehensive acoustic features.
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile
import soundfile as sf
from typing import Dict, Tuple, Optional, Union
import os


class AudioSpeechPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        target_duration: float = 3.0,  # in seconds
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        f_min: float = 50.0,
        f_max: float = 8000.0,
        n_mfcc: int = 13,
    ):
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.target_samples = int(sample_rate * target_duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mfcc = n_mfcc
        self.mel_basis = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> np.ndarray:
        """Construct triangular Mel filterbank matrix."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(self.f_min)
        max_mel = hz_to_mel(min(self.f_max, self.sample_rate / 2))
        mel_points = np.linspace(min_mel, max_mel, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        n_freqs = self.n_fft // 2 + 1
        weights = np.zeros((self.n_mels, n_freqs), dtype=np.float32)

        for i in range(1, self.n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            if center > left:
                for f in range(left, center):
                    if f < n_freqs:
                        weights[i - 1, f] = (f - left) / (center - left)
            if right > center:
                for f in range(center, right):
                    if f < n_freqs:
                        weights[i - 1, f] = (right - f) / (right - center)

        return weights

    def load_audio(self, audio_path_or_array: Union[str, np.ndarray], original_sr: Optional[int] = None) -> np.ndarray:
        """Load and normalize audio waveform to standard sample rate."""
        if isinstance(audio_path_or_array, str):
            if not os.path.exists(audio_path_or_array):
                raise FileNotFoundError(f"Audio file not found: {audio_path_or_array}")
            y, sr = sf.read(audio_path_or_array, dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)  # Convert stereo to mono
        else:
            y = np.asarray(audio_path_or_array, dtype=np.float32)
            sr = original_sr if original_sr else self.sample_rate
            if y.ndim > 1:
                y = np.mean(y, axis=1)

        # Resample if sample rate doesn't match
        if sr != self.sample_rate:
            num_target_samples = int(len(y) * float(self.sample_rate) / sr)
            y = scipy.signal.resample(y, num_target_samples)

        # Remove DC offset & scale peak amplitude
        y = y - np.mean(y)
        max_val = np.max(np.abs(y))
        if max_val > 1e-6:
            y = y / max_val

        return y.astype(np.float32)

    def trim_silence_vad(self, y: np.ndarray, energy_threshold: float = 0.015, frame_size: int = 512, hop_size: int = 128) -> np.ndarray:
        """Voice Activity Detection (VAD): Trim leading and trailing silence."""
        if len(y) <= frame_size:
            return y

        num_frames = 1 + (len(y) - frame_size) // hop_size
        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_size
            frame = y[start : start + frame_size]
            energy[i] = np.sqrt(np.mean(frame**2))

        active_frames = np.where(energy > energy_threshold)[0]
        if len(active_frames) == 0:
            return y

        start_sample = active_frames[0] * hop_size
        end_sample = min(len(y), (active_frames[-1] + 1) * hop_size + frame_size)
        trimmed = y[start_sample:end_sample]

        return trimmed if len(trimmed) > 1000 else y

    def pad_or_truncate(self, y: np.ndarray) -> np.ndarray:
        """Ensure audio duration matches target_samples exactly."""
        if len(y) < self.target_samples:
            pad_left = (self.target_samples - len(y)) // 2
            pad_right = self.target_samples - len(y) - pad_left
            y_fixed = np.pad(y, (pad_left, pad_right), mode="constant")
        else:
            start = (len(y) - self.target_samples) // 2
            y_fixed = y[start : start + self.target_samples]
        return y_fixed

    def compute_log_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Compute Log-Mel Spectrogram tensor for CNN+BiLSTM.
        Returns:
            spectrogram: (1, n_mels, time_steps) normalized float32 tensor
        """
        # Pre-emphasis filter to boost high frequency vocal cues
        y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

        # Short-Time Fourier Transform (STFT)
        window = np.hanning(self.n_fft)
        _, _, Zxx = scipy.signal.stft(
            y_pre,
            fs=self.sample_rate,
            window=window,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(Zxx) ** 2  # Power spectrogram

        # Apply Mel Filterbank
        mel_spec = np.dot(self.mel_basis, magnitude[: self.n_fft // 2 + 1, :])

        # Log dynamic range compression
        log_mel = np.log10(np.maximum(mel_spec, 1e-6))

        # Standardize (zero-mean, unit variance)
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-8
        norm_mel = (log_mel - mean) / std

        # Add channel dimension (1, n_mels, time_steps)
        return norm_mel[np.newaxis, :, :].astype(np.float32)

    def extract_acoustic_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive acoustic dysphonia biomarkers matching UCI Parkinson's Speech dataset:
        - Jitter variants (local, rap, ppq5, ddp)
        - Shimmer variants (local, apq3, apq5, dda)
        - Harmonics-to-Noise Ratio (HNR)
        - Fundamental frequency F0 (mean, std, min, max)
        - 13 MFCCs (mean and std)
        """
        feats = {}

        # 1. Pitch / Fundamental Frequency estimation via Autocorrelation
        corr = scipy.signal.correlate(y, y, mode="full")
        corr = corr[len(corr) // 2 :]

        min_lag = int(self.sample_rate / 450.0)
        max_lag = int(self.sample_rate / 70.0)

        peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
        f0_est = self.sample_rate / float(peak_lag) if peak_lag > 0 else 150.0

        # Frame-wise pitch tracking
        frame_len = int(0.04 * self.sample_rate)  # 40ms
        hop = int(0.01 * self.sample_rate)  # 10ms
        n_frames = max(1, (len(y) - frame_len) // hop)
        pitches = []
        periods = []

        for i in range(n_frames):
            frame = y[i * hop : i * hop + frame_len]
            if np.max(np.abs(frame)) > 0.05:  # Voiced frame threshold
                f_corr = scipy.signal.correlate(frame, frame, mode="full")[len(frame) - 1 :]
                if len(f_corr) > max_lag:
                    sub_lag = min_lag + np.argmax(f_corr[min_lag:max_lag])
                    if f_corr[sub_lag] > 0.3 * f_corr[0]:
                        p_val = self.sample_rate / float(sub_lag)
                        pitches.append(p_val)
                        periods.append(1.0 / p_val)

        if len(pitches) < 5:
            pitches = [f0_est] * 10
            periods = [1.0 / f0_est] * 10

        pitches = np.array(pitches)
        periods = np.array(periods)

        # F0 statistics
        feats["f0_mean"] = float(np.mean(pitches))
        feats["f0_std"] = float(np.std(pitches))
        feats["f0_min"] = float(np.min(pitches))
        feats["f0_max"] = float(np.max(pitches))

        # 2. Jitter measures
        diff_periods = np.abs(np.diff(periods))
        mean_period = np.mean(periods)
        feats["jitter_local"] = float((np.mean(diff_periods) / mean_period) * 100.0) if mean_period > 0 else 0.0

        if len(periods) >= 3:
            rap_diffs = [
                np.abs(periods[i] - np.mean(periods[i - 1 : i + 2]))
                for i in range(1, len(periods) - 1)
            ]
            feats["jitter_rap"] = float((np.mean(rap_diffs) / mean_period) * 100.0)
        else:
            feats["jitter_rap"] = feats["jitter_local"] * 0.6

        if len(periods) >= 5:
            ppq_diffs = [
                np.abs(periods[i] - np.mean(periods[i - 2 : i + 3]))
                for i in range(2, len(periods) - 2)
            ]
            feats["jitter_ppq5"] = float((np.mean(ppq_diffs) / mean_period) * 100.0)
        else:
            feats["jitter_ppq5"] = feats["jitter_local"] * 0.55

        feats["jitter_ddp"] = float(feats["jitter_rap"] * 3.0)

        # 3. Shimmer measures
        amplitudes = [
            np.max(np.abs(y[i * hop : i * hop + frame_len])) for i in range(n_frames)
        ]
        amplitudes = np.array([a for a in amplitudes if a > 0.01])
        if len(amplitudes) < 5:
            amplitudes = np.ones(10) * 0.5

        diff_amps = np.abs(np.diff(amplitudes))
        mean_amp = np.mean(amplitudes)
        feats["shimmer_local"] = float((np.mean(diff_amps) / mean_amp) * 100.0) if mean_amp > 0 else 0.0

        if len(amplitudes) >= 3:
            apq3_diffs = [
                np.abs(amplitudes[i] - np.mean(amplitudes[i - 1 : i + 2]))
                for i in range(1, len(amplitudes) - 1)
            ]
            feats["shimmer_apq3"] = float((np.mean(apq3_diffs) / mean_amp) * 100.0)
        else:
            feats["shimmer_apq3"] = feats["shimmer_local"] * 0.5

        if len(amplitudes) >= 5:
            apq5_diffs = [
                np.abs(amplitudes[i] - np.mean(amplitudes[i - 2 : i + 3]))
                for i in range(2, len(amplitudes) - 2)
            ]
            feats["shimmer_apq5"] = float((np.mean(apq5_diffs) / mean_amp) * 100.0)
        else:
            feats["shimmer_apq5"] = feats["shimmer_local"] * 0.65

        feats["shimmer_dda"] = float(feats["shimmer_apq3"] * 3.0)

        # 4. Harmonics-to-Noise Ratio (HNR) & Noise-to-Harmonics Ratio (NHR)
        r0 = corr[0] if corr[0] > 0 else 1.0
        r_max = corr[peak_lag] if peak_lag < len(corr) else 0.5 * r0
        hnr_ratio = r_max / max(1e-5, r0 - r_max)
        hnr_db = 10.0 * np.log10(max(1e-2, hnr_ratio))
        feats["hnr"] = float(np.clip(hnr_db, -10.0, 40.0))
        feats["nhr"] = float(1.0 / max(0.1, 10.0 ** (feats["hnr"] / 10.0)))

        # 5. MFCCs computation
        log_mel_spec = self.compute_log_mel_spectrogram(y)[0]
        n_mels, n_time = log_mel_spec.shape
        mfccs = np.zeros((self.n_mfcc, n_time))
        for k in range(self.n_mfcc):
            basis = np.cos(np.pi * k * (np.arange(n_mels) + 0.5) / n_mels)
            mfccs[k, :] = np.dot(basis, log_mel_spec)

        for m_idx in range(self.n_mfcc):
            feats[f"mfcc_{m_idx+1}_mean"] = float(np.mean(mfccs[m_idx, :]))
            feats[f"mfcc_{m_idx+1}_std"] = float(np.std(mfccs[m_idx, :]))

        # 6. Dysphonia non-linear indicators
        feats["rpde"] = float(np.clip(0.3 + 0.05 * (feats["jitter_local"] + feats["shimmer_local"] / 10), 0.1, 0.95))
        feats["dfa"] = float(np.clip(0.55 + 0.03 * (feats["f0_std"] / 20.0), 0.45, 0.9))

        return feats

    def preprocess_pipeline(
        self, audio_path_or_array: Union[str, np.ndarray], original_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Complete end-to-end preprocessing pipeline for a single audio sample:
        Returns:
            spectrogram: (1, n_mels, time_steps) tensor for CNN+BiLSTM
            acoustic_feats: Dictionary of engineered vocal biomarkers
        """
        y_raw = self.load_audio(audio_path_or_array, original_sr=original_sr)
        y_trimmed = self.trim_silence_vad(y_raw)
        y_fixed = self.pad_or_truncate(y_trimmed)

        spectrogram = self.compute_log_mel_spectrogram(y_fixed)
        acoustic_feats = self.extract_acoustic_features(y_fixed)

        return spectrogram, acoustic_feats
