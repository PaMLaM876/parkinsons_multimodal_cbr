"""
Convert Real PPMI DICOM MRI scans to isotropic 3D NumPy arrays (.npy) for interactive slicing.
"""
import os
import glob
import numpy as np
import pydicom
import scipy.ndimage

def normalize_image(img):
    """Normalize volume to 0-255 uint8."""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img.astype(np.uint8)

def convert_dicoms_to_numpy(dicom_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all DICOM files
    dcm_files = glob.glob(os.path.join(dicom_dir, "**", "*.dcm"), recursive=True)
    if not dcm_files:
        print(f"[!] No DICOM files found in {dicom_dir}")
        return
        
    print(f"[*] Found {len(dcm_files)} DICOM files. Grouping by series...")
    
    # Group by SeriesInstanceUID
    series = {}
    patient_series = {}
    
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            uid = ds.SeriesInstanceUID
            if uid not in series:
                series[uid] = []
            series[uid].append(f)
            
            pid = ds.PatientID if hasattr(ds, "PatientID") else "Unknown"
            if pid not in patient_series:
                patient_series[pid] = []
            if (uid, series[uid]) not in patient_series[pid]:
                patient_series[pid].append((uid, series[uid]))
        except Exception:
            pass
            
    print(f"[*] Found {len(series)} distinct MRI series.")
    
    # For each patient, pick the series with the most slices (the full 3D T1 scan, not a 16-slice localizer)
    best_series = {}
    for pid, s_list in patient_series.items():
        # Update the lists with the full files list from `series` since we only appended references initially
        s_list_full = [(uid, series[uid]) for uid, _ in s_list]
        best_uid, best_files = max(s_list_full, key=lambda x: len(x[1]))
        best_series[pid] = best_files
        print(f"    -> Patient {pid}: Selected series with {len(best_files)} slices (discarded {len(s_list_full)-1} smaller series)")

    for idx, (pid, files) in enumerate(best_series.items()):
        print(f"    -> Processing High-Res Series for Patient {pid} ({len(files)} slices)")
        slices = []
        for f in files:
            ds = pydicom.dcmread(f)
            slices.append(ds)
            
        # Sort by physical Z-coordinate (ImagePositionPatient[2]) to prevent interleaving artifacts
        def get_z_pos(s):
            if hasattr(s, "ImagePositionPatient"):
                return float(s.ImagePositionPatient[2])
            return int(getattr(s, "InstanceNumber", 0))
            
        slices.sort(key=get_z_pos)
        
        # Build 3D volume
        try:
            volume = np.stack([s.pixel_array for s in slices])
        except Exception as e:
            print(f"       [!] Failed to stack pixel arrays (mismatched shapes). Skipping.")
            continue
            
        # Read voxel dimensions for correct aspect ratio scaling
        try:
            px_spacing = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
            slice_thickness = getattr(slices[0], "SliceThickness", 1.0)
            if not isinstance(slice_thickness, (int, float)):
                slice_thickness = float(slice_thickness)
            dx = float(px_spacing[0])
            dy = float(px_spacing[1])
            dz = slice_thickness
        except Exception:
            dx, dy, dz = 1.0, 1.0, 1.0
            
        # Interpolate volume to make voxels isotropic (cubic)
        # We target the finest resolution (usually X/Y)
        target_res = min(dx, dy, dz)
        if target_res <= 0: target_res = 1.0
        
        zoom_z = dz / target_res
        zoom_y = dy / target_res
        zoom_x = dx / target_res
        
        print(f"       Resampling volume (original shape: {volume.shape}, zooms: Z={zoom_z:.2f}, Y={zoom_y:.2f}, X={zoom_x:.2f})")
        
        try:
            # We use order=1 (bilinear) for speed, order=3 (cubic) would look slightly better but take much longer
            iso_volume = scipy.ndimage.zoom(volume, (zoom_z, zoom_y, zoom_x), order=1)
        except Exception as e:
            print(f"       [!] Resampling failed: {e}. Using original volume.")
            iso_volume = volume
            
        # Normalize and save
        iso_volume = normalize_image(iso_volume)
        out_file = os.path.join(out_dir, f"{pid}.npy")
        np.save(out_file, iso_volume)
            
    print(f"[+] All isotropic NumPy volumes saved to {out_dir}")

if __name__ == "__main__":
    dicom_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "real", "ppmi", "mri"))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "real", "ppmi", "mri_numpy"))
    convert_dicoms_to_numpy(dicom_dir, out_dir)
