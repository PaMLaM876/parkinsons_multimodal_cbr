"""
3D Structural MRI (sMRI) Preprocessor for PPMI T1-weighted Neuroimaging.
Handles NIfTI loading, RAS+ reorientation, skull stripping, 3D isometric resampling,
robust intensity normalization, and multi-planar slice generation for 3D ResNet-50.
"""

import numpy as np
import scipy.ndimage
import nibabel as nib
from typing import Tuple, Optional, Union, Dict
import os


class MRI3DPreprocessor:
    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (96, 96, 96),
        clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        apply_skull_strip: bool = True,
    ):
        self.target_shape = target_shape
        self.clip_percentiles = clip_percentiles
        self.apply_skull_strip = apply_skull_strip

    def load_nifti(self, file_path_or_array: Union[str, np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load a 3D MRI volume from NIfTI file or numpy array.
        Returns:
            volume: 3D numpy array (D, H, W)
            affine: 4x4 affine matrix if available
        """
        if isinstance(file_path_or_array, str):
            if not os.path.exists(file_path_or_array):
                raise FileNotFoundError(f"MRI file not found: {file_path_or_array}")
            nii = nib.load(file_path_or_array)
            # Reorient to RAS+ canonical orientation
            nii = nib.as_closest_canonical(nii)
            vol = nii.get_fdata(dtype=np.float32)
            affine = nii.affine
        else:
            vol = np.asarray(file_path_or_array, dtype=np.float32)
            affine = np.eye(4)

        # Squeeze 4D (e.g. 1, D, H, W or D, H, W, 1) down to 3D
        if vol.ndim == 4:
            vol = np.squeeze(vol)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D MRI volume, got shape {vol.shape}")

        return vol, affine

    def skull_strip_otsu_morphology(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automated 3D brain extraction / skull-stripping using adaptive intensity thresholding
        and 3D morphological operations (dilation/erosion/connected components).
        Returns:
            masked_volume: 3D volume with skull and non-brain tissue zeroed out
            brain_mask: Boolean mask of brain tissue
        """
        # Exclude background zero air voxels for threshold estimation
        non_zero = volume[volume > 0.05 * np.max(volume)]
        if len(non_zero) == 0:
            return volume, np.ones_like(volume, dtype=bool)

        # Otsu's thresholding in 3D
        hist, bin_edges = np.histogram(non_zero, bins=128)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        weight1 = np.cumsum(hist) / float(len(non_zero))
        weight2 = 1.0 - weight1

        mean1 = np.cumsum(hist * bin_centers) / np.maximum(1, np.cumsum(hist))
        mean2 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / np.maximum(1, len(non_zero) - np.cumsum(hist))

        variance = weight1[:-1] * weight2[:-1] * ((mean1[:-1] - mean2[:-1]) ** 2)
        thresh_idx = np.argmax(variance)
        thresh = bin_centers[thresh_idx] * 0.7  # Conservative brain threshold

        # Generate preliminary binary mask
        raw_mask = volume > thresh

        # 3D Morphological opening (erosion followed by dilation) to remove thin skull bridges
        struct_elem = scipy.ndimage.generate_binary_structure(3, 1)  # 6-connectivity
        opened_mask = scipy.ndimage.binary_opening(raw_mask, structure=struct_elem, iterations=2)

        # Retain the largest 3D connected component (the brain cerebrum/cerebellum)
        labeled_array, num_features = scipy.ndimage.label(opened_mask)
        if num_features > 0:
            sizes = scipy.ndimage.sum(opened_mask, labeled_array, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            brain_mask = labeled_array == largest_label
        else:
            brain_mask = raw_mask

        # 3D Morphological closing and hole filling to encompass ventricles and deep grey nuclei
        brain_mask = scipy.ndimage.binary_closing(brain_mask, structure=struct_elem, iterations=3)
        brain_mask = scipy.ndimage.binary_fill_holes(brain_mask)

        masked_volume = volume * brain_mask
        return masked_volume, brain_mask

    def resample_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resample 3D MRI volume to standard isometric target shape (e.g. 96x96x96).
        Uses 3rd-order spline interpolation for smooth anatomic preservation.
        """
        current_shape = volume.shape
        if current_shape == self.target_shape:
            return volume

        zoom_factors = [
            self.target_shape[i] / float(current_shape[i]) for i in range(3)
        ]
        resampled = scipy.ndimage.zoom(
            volume, zoom=zoom_factors, order=2, mode="nearest", prefilter=True
        )
        return resampled.astype(np.float32)

    def normalize_intensity(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Robust MR intensity normalization:
        1. Clip outliers at 1st and 99th percentiles (mitigate scanner artifacts/bias fields)
        2. Standardize brain voxel intensities to zero mean, unit variance (Z-score)
        3. Min-Max scale into [0, 1] range for neural network convergence
        """
        active_voxels = volume[volume > 0] if mask is None else volume[mask]
        if len(active_voxels) == 0:
            return volume

        # Percentile clipping
        p_low, p_high = np.percentile(active_voxels, self.clip_percentiles)
        clipped = np.clip(volume, p_low, p_high)

        # Z-score standardization on brain voxels
        active_clipped = clipped[clipped > 0]
        mean_val = np.mean(active_clipped)
        std_val = np.std(active_clipped) + 1e-8
        z_scored = np.where(volume > 0, (clipped - mean_val) / std_val, 0.0)

        # Scale into [-1, 1] for 3D ResNet input
        min_v = np.min(z_scored[volume > 0]) if np.any(volume > 0) else -1.0
        max_v = np.max(z_scored[volume > 0]) if np.any(volume > 0) else 1.0
        norm_vol = np.where(volume > 0, 2.0 * (z_scored - min_v) / (max_v - min_v + 1e-8) - 1.0, 0.0)

        return norm_vol.astype(np.float32)

    def extract_multiplanar_slices(
        self, volume: np.ndarray, slice_indices: Optional[Dict[str, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract central or specified slices along three orthogonal anatomic planes:
        - Axial (Transverse): Superior to Inferior
        - Coronal (Frontal): Anterior to Posterior
        - Sagittal: Left to Right
        """
        D, H, W = volume.shape
        if slice_indices is None:
            slice_indices = {
                "axial": D // 2,
                "coronal": H // 2,
                "sagittal": W // 2,
            }

        slices = {
            "axial": volume[slice_indices["axial"], :, :],
            "coronal": volume[:, slice_indices["coronal"], :],
            "sagittal": volume[:, :, slice_indices["sagittal"]],
        }
        return slices

    def preprocess_pipeline(
        self, file_path_or_array: Union[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Complete 3D MRI preprocessing pipeline:
        1. Ingest NIfTI or numpy 3D volume
        2. Skull-strip / extract brain tissue mask
        3. Resample isometrically to target 3D shape (e.g. 96x96x96)
        4. Normalize intensity (Z-score + Percentile clipping)
        5. Format as (1, D, H, W) tensor for 3D ResNet-50
        6. Extract multi-planar inspection slices
        """
        raw_vol, _ = self.load_nifti(file_path_or_array)

        if self.apply_skull_strip:
            brain_vol, mask = self.skull_strip_otsu_morphology(raw_vol)
        else:
            brain_vol = raw_vol
            mask = raw_vol > 0

        resampled_vol = self.resample_3d(brain_vol)
        norm_vol = self.normalize_intensity(resampled_vol)

        # Tensor formatted as (1, D, H, W) for PyTorch Conv3D
        tensor_3d = norm_vol[np.newaxis, :, :].astype(np.float32)

        # Multi-planar orthogonal views for clinical interpretation
        slices = self.extract_multiplanar_slices(norm_vol)

        return tensor_3d, slices
