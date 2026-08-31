"""
Convert Real PPMI DICOM MRI scans to JPEG images for the showcase.
"""
import os
import glob
import numpy as np
import pydicom
from PIL import Image

def normalize_image(img):
    """Normalize pixel array to 0-255 uint8."""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img.astype(np.uint8)

def convert_dicoms_to_jpeg(dicom_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all DICOM files
    dcm_files = glob.glob(os.path.join(dicom_dir, "**", "*.dcm"), recursive=True)
    if not dcm_files:
        print(f"[!] No DICOM files found in {dicom_dir}")
        return
        
    print(f"[*] Found {len(dcm_files)} DICOM files. Grouping by series...")
    
    # Group by SeriesInstanceUID
    series = {}
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            uid = ds.SeriesInstanceUID
            if uid not in series:
                series[uid] = []
            series[uid].append(f)
        except Exception as e:
            pass
            
    print(f"[*] Found {len(series)} distinct MRI series.")
    
    # Group series by patient to find the highest-resolution scan (avoiding localizers)
    patient_series = {}
    for uid, files in series.items():
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            pid = ds.PatientID if hasattr(ds, "PatientID") else "Unknown"
            if pid not in patient_series:
                patient_series[pid] = []
            patient_series[pid].append((uid, files))
        except Exception:
            pass
            
    # For each patient, pick the series with the most slices (the full 3D T1 scan, not a 16-slice localizer)
    best_series = {}
    for pid, s_list in patient_series.items():
        best_uid, best_files = max(s_list, key=lambda x: len(x[1]))
        best_series[best_uid] = best_files
        print(f"    -> Patient {pid}: Selected series with {len(best_files)} slices (discarded {len(s_list)-1} smaller series)")

    for idx, (uid, files) in enumerate(best_series.items()):
        print(f"    -> Processing High-Res Series {idx+1}/{len(best_series)} ({len(files)} slices)")
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
            
        # Extract central slices
        z, y, x = volume.shape
        axial = volume[z // 2, :, :]
        coronal = volume[:, y // 2, :]
        sagittal = volume[:, :, x // 2]
        
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
        
        # Normalize and save
        subject_id = slices[0].PatientID if hasattr(slices[0], "PatientID") else f"Subject_{idx}"
        
        planes = {"axial": axial, "coronal": coronal, "sagittal": sagittal}
        
        # Calculate physical dimensions
        physical_x = x * dx
        physical_y = y * dy
        physical_z = z * dz
        
        for plane, img_data in planes.items():
            img_normalized = normalize_image(img_data)
            im = Image.fromarray(img_normalized)
            
            if plane == "axial":
                pass # Axial is usually isotropic or close to it, keep original size
            elif plane == "coronal":
                scale_factor = x / physical_x if physical_x > 0 else 1.0
                target_width = x
                target_height = int(physical_z * scale_factor)
                im = im.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Pad to square to prevent UI from stretching it
                max_dim = max(target_width, target_height)
                square = Image.new("L", (max_dim, max_dim), color=0)
                square.paste(im, ((max_dim - target_width) // 2, (max_dim - target_height) // 2))
                im = square
                
            elif plane == "sagittal":
                scale_factor = y / physical_y if physical_y > 0 else 1.0
                target_width = y
                target_height = int(physical_z * scale_factor)
                im = im.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Pad to square to prevent UI from stretching it
                max_dim = max(target_width, target_height)
                square = Image.new("L", (max_dim, max_dim), color=0)
                square.paste(im, ((max_dim - target_width) // 2, (max_dim - target_height) // 2))
                im = square
                
            out_file = os.path.join(out_dir, f"{subject_id}_{idx}_{plane}.jpg")
            im.save(out_file)
            
    print(f"[+] All JPEGs saved to {out_dir}")

if __name__ == "__main__":
    dicom_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "real", "ppmi", "mri"))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "real", "ppmi", "mri_jpeg"))
    convert_dicoms_to_jpeg(dicom_dir, out_dir)
