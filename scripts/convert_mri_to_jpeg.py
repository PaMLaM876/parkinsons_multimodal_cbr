"""
Convert 3D MRI Volumes to JPEG Slices for Model Training.

Reads .npy (NumPy) 3D brain volumes from data/raw/mri/ and exports
central axial, coronal, and sagittal 2D slices as JPEG images.

Also supports DICOM (.dcm) → JPEG conversion if pydicom is installed.

Output structure:
    data/raw/mri_jpeg/
    ├── axial/
    │   ├── PPMI_1000_axial.jpg
    │   ├── PPMI_1001_axial.jpg
    │   └── ...
    ├── coronal/
    │   ├── PPMI_1000_coronal.jpg
    │   └── ...
    └── sagittal/
        ├── PPMI_1000_sagittal.jpg
        └── ...

Usage:
    python scripts/convert_mri_to_jpeg.py
    python scripts/convert_mri_to_jpeg.py --input_dir data/raw/mri --output_dir data/raw/mri_jpeg
    python scripts/convert_mri_to_jpeg.py --quality 95 --resize 224
    python scripts/convert_mri_to_jpeg.py --dicom_dir path/to/dicoms  (requires pydicom)
"""

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Optional DICOM support
try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D float array to uint8 [0, 255] for JPEG encoding.
    Handles both [0, 1] and arbitrary-range volumes with robust min/max scaling.
    """
    arr = slice_2d.astype(np.float64)

    # Clip outliers at 1st / 99th percentile for better contrast
    non_zero = arr[arr > 0]
    if len(non_zero) > 0:
        p_low = np.percentile(non_zero, 1.0)
        p_high = np.percentile(non_zero, 99.0)
        arr = np.clip(arr, p_low, p_high)

    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)

    normalized = (arr - arr_min) / (arr_max - arr_min) * 255.0
    return normalized.astype(np.uint8)


def extract_central_slices(volume: np.ndarray) -> dict:
    """
    Extract the central axial, coronal, and sagittal slices from a 3D volume.

    Args:
        volume: 3D numpy array with shape (D, H, W)

    Returns:
        Dictionary mapping plane name → 2D slice array
    """
    if volume.ndim == 4:
        volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    D, H, W = volume.shape
    return {
        "axial": volume[D // 2, :, :],       # Transverse plane (superior–inferior)
        "coronal": volume[:, H // 2, :],      # Frontal plane (anterior–posterior)
        "sagittal": volume[:, :, W // 2],     # Left–right plane
    }


def save_slice_as_jpeg(
    slice_2d: np.ndarray,
    output_path: str,
    quality: int = 90,
    resize: int = None,
) -> None:
    """
    Save a 2D array as a grayscale JPEG image.

    Args:
        slice_2d:    2D numpy array (float or int)
        output_path: Destination file path (.jpg)
        quality:     JPEG quality [1–95], higher = better quality / larger file
        resize:      If set, resize the image to (resize x resize) pixels
    """
    img_data = normalize_to_uint8(slice_2d)
    img = Image.fromarray(img_data, mode="L")  # "L" = 8-bit grayscale

    if resize is not None and resize > 0:
        img = img.resize((resize, resize), Image.LANCZOS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "JPEG", quality=quality)


def convert_npy_volumes(
    input_dir: str,
    output_dir: str,
    quality: int = 90,
    resize: int = None,
) -> dict:
    """
    Convert all .npy 3D MRI volumes in input_dir to JPEG slices.

    Returns:
        Summary dict with counts and any errors.
    """
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not npy_files:
        print(f"[!] No .npy files found in {input_dir}")
        return {"total": 0, "converted": 0, "errors": []}

    print(f"[*] Found {len(npy_files)} .npy MRI volumes in {input_dir}")
    print(f"[*] Output directory: {output_dir}")
    print(f"[*] JPEG quality: {quality}" + (f"  |  Resize: {resize}x{resize}" if resize else ""))
    print()

    converted = 0
    errors = []

    for npy_path in npy_files:
        subject_id = os.path.splitext(os.path.basename(npy_path))[0]  # e.g. "PPMI_1000"

        try:
            volume = np.load(npy_path)
            slices = extract_central_slices(volume)

            for plane_name, slice_2d in slices.items():
                out_path = os.path.join(output_dir, plane_name, f"{subject_id}_{plane_name}.jpg")
                save_slice_as_jpeg(slice_2d, out_path, quality=quality, resize=resize)

            converted += 1
            if converted % 20 == 0 or converted == len(npy_files):
                print(f"    [{converted}/{len(npy_files)}] Converted {subject_id}")

        except Exception as e:
            err_msg = f"Error processing {subject_id}: {e}"
            print(f"    [!] {err_msg}")
            errors.append(err_msg)

    summary = {
        "total": len(npy_files),
        "converted": converted,
        "output_files": converted * 3,  # 3 planes per volume
        "planes": ["axial", "coronal", "sagittal"],
        "errors": errors,
    }
    return summary


def convert_dicom_files(
    dicom_dir: str,
    output_dir: str,
    quality: int = 90,
    resize: int = None,
) -> dict:
    """
    Convert DICOM (.dcm) files to JPEG.
    Each DICOM file is treated as a single 2D slice.

    Requires: pip install pydicom

    Returns:
        Summary dict with counts and errors.
    """
    if not HAS_PYDICOM:
        print("[!] pydicom is not installed. Install it with:")
        print("    pip install pydicom")
        return {"total": 0, "converted": 0, "errors": ["pydicom not installed"]}

    dcm_files = sorted(
        glob.glob(os.path.join(dicom_dir, "**", "*.dcm"), recursive=True)
        + glob.glob(os.path.join(dicom_dir, "**", "*.DCM"), recursive=True)
    )

    # Also search for extensionless DICOM files (common in clinical datasets)
    for root, dirs, files in os.walk(dicom_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if fpath not in dcm_files and not os.path.splitext(f)[1]:
                # Check if it's DICOM by reading magic bytes
                try:
                    with open(fpath, "rb") as fp:
                        fp.seek(128)
                        if fp.read(4) == b"DICM":
                            dcm_files.append(fpath)
                except Exception:
                    pass

    if not dcm_files:
        print(f"[!] No DICOM files found in {dicom_dir}")
        return {"total": 0, "converted": 0, "errors": []}

    print(f"[*] Found {len(dcm_files)} DICOM files in {dicom_dir}")
    print(f"[*] Output directory: {output_dir}")

    converted = 0
    errors = []

    for dcm_path in dcm_files:
        try:
            ds = pydicom.dcmread(dcm_path)
            pixel_array = ds.pixel_array.astype(np.float64)

            # Apply DICOM windowing if available
            if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
                width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
                low = center - width / 2
                high = center + width / 2
                pixel_array = np.clip(pixel_array, low, high)

            # Apply RescaleSlope / RescaleIntercept if present
            if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
                pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

            # Build output filename from DICOM metadata or path
            patient_id = getattr(ds, "PatientID", "unknown")
            instance_num = getattr(ds, "InstanceNumber", "0")
            series_desc = getattr(ds, "SeriesDescription", "scan").replace(" ", "_")
            out_name = f"{patient_id}_{series_desc}_{instance_num}.jpg"
            out_path = os.path.join(output_dir, "dicom_converted", out_name)

            save_slice_as_jpeg(pixel_array, out_path, quality=quality, resize=resize)
            converted += 1

            if converted % 50 == 0:
                print(f"    [{converted}/{len(dcm_files)}] Converting...")

        except Exception as e:
            err_msg = f"Error: {os.path.basename(dcm_path)}: {e}"
            errors.append(err_msg)

    summary = {
        "total": len(dcm_files),
        "converted": converted,
        "errors": errors,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D MRI volumes (.npy) and DICOM (.dcm) files to JPEG slices."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join("data", "raw", "mri"),
        help="Directory containing .npy 3D MRI volumes (default: data/raw/mri)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "raw", "mri_jpeg"),
        help="Output directory for JPEG slices (default: data/raw/mri_jpeg)",
    )
    parser.add_argument(
        "--dicom_dir",
        type=str,
        default=None,
        help="Optional: Directory containing DICOM files to convert (requires pydicom)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG quality [1-95] (default: 90)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize slices to NxN pixels (e.g. 224 for ResNet input). Default: keep original size.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MRI -> JPEG Conversion Tool")
    print("=" * 60)
    print()

    # ── Convert .npy volumes ──────────────────────────────────
    npy_summary = convert_npy_volumes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        quality=args.quality,
        resize=args.resize,
    )

    # ── Convert DICOM files (optional) ────────────────────────
    dcm_summary = None
    if args.dicom_dir:
        print()
        dcm_summary = convert_dicom_files(
            dicom_dir=args.dicom_dir,
            output_dir=args.output_dir,
            quality=args.quality,
            resize=args.resize,
        )

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  CONVERSION COMPLETE")
    print("=" * 60)

    if npy_summary["total"] > 0:
        print(f"  NumPy (.npy) volumes:  {npy_summary['converted']}/{npy_summary['total']} converted")
        print(f"  JPEG files generated:  {npy_summary['output_files']} (3 planes × {npy_summary['converted']} subjects)")
        print(f"  Planes:                {', '.join(npy_summary['planes'])}")
        if npy_summary["errors"]:
            print(f"  Errors:                {len(npy_summary['errors'])}")

    if dcm_summary and dcm_summary["total"] > 0:
        print(f"  DICOM files:           {dcm_summary['converted']}/{dcm_summary['total']} converted")
        if dcm_summary["errors"]:
            print(f"  DICOM Errors:          {len(dcm_summary['errors'])}")

    print(f"  Output directory:      {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
