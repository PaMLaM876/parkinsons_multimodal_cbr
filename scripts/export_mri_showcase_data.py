"""
Export per-subject MRI analysis data for the showcase dashboard.

Runs the trained model on test data and exports:
- MRI gate weights per subject
- MRI encoder embedding statistics
- Per-subject prediction results with MRI contribution

Usage:
    python scripts/export_mri_showcase_data.py
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import MultimodalPDModel, CBREngine


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "processed", "test_multimodal.npz")
    ckpt_path = os.path.join(project_root, "checkpoints", "best_model.pt")
    output_path = os.path.join(project_root, "data", "processed", "mri_showcase_results.json")
    mri_jpeg_dir = os.path.join(project_root, "data", "raw", "mri_jpeg")

    print("=" * 60)
    print("  MRI Showcase Data Export")
    print("=" * 60)

    # Load test data
    data = np.load(data_path, allow_pickle=True)
    test_data = {
        "speech_spec": torch.tensor(data["speech_specs"], dtype=torch.float32),
        "acoustic_vec": torch.tensor(data["acoustic_feats"], dtype=torch.float32),
        "mri_tensor": torch.tensor(data["mri_tensors"], dtype=torch.float32),
        "clinical_vec": torch.tensor(data["clinical_vecs"], dtype=torch.float32),
        "modality_mask": torch.tensor(data["modality_masks"], dtype=torch.float32),
        "label": torch.tensor(data["labels"], dtype=torch.long),
        "subject_id": data["subject_ids"].tolist(),
    }
    print(f"[+] Loaded {len(test_data['subject_id'])} test subjects")

    # Load model
    device = torch.device("cpu")
    model = MultimodalPDModel(
        clinical_input_dim=test_data["clinical_vec"].shape[1],
        acoustic_input_dim=test_data["acoustic_vec"].shape[1],
        use_light_mri=True,
    ).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[+] Loaded model from epoch {ckpt['epoch']}")

    # Also load training data for gate statistics
    train_path = os.path.join(project_root, "data", "processed", "train_multimodal.npz")
    train_data_np = np.load(train_path, allow_pickle=True)

    # Run inference on all test subjects
    subjects_data = []
    all_gate_speech = []
    all_gate_mri = []
    all_gate_clinical = []

    with torch.no_grad():
        for i in range(len(test_data["subject_id"])):
            sid = test_data["subject_id"][i]

            speech = test_data["speech_spec"][i:i+1].to(device)
            acoustic = test_data["acoustic_vec"][i:i+1].to(device)
            mri = test_data["mri_tensor"][i:i+1].to(device)
            clinical = test_data["clinical_vec"][i:i+1].to(device)
            mask = test_data["modality_mask"][i:i+1].to(device)

            output = model(speech, acoustic, mri, clinical, mask)
            probs = output["probabilities"][0].cpu().numpy()
            gates = output["gate_values"][0].cpu().numpy()
            embedding = output["embedding"][0].cpu().numpy()

            # MRI encoder embedding (before fusion)
            mri_emb = model.mri_encoder(mri)[0].cpu().numpy()

            pred_class = int(output["logits"].argmax(dim=1).item())
            true_label = int(test_data["label"][i].item())

            # Check if MRI JPEG slices exist
            has_axial = os.path.exists(os.path.join(mri_jpeg_dir, "axial", f"{sid}_axial.jpg"))
            has_coronal = os.path.exists(os.path.join(mri_jpeg_dir, "coronal", f"{sid}_coronal.jpg"))
            has_sagittal = os.path.exists(os.path.join(mri_jpeg_dir, "sagittal", f"{sid}_sagittal.jpg"))

            # MRI volume statistics
            mri_vol = test_data["mri_tensor"][i, 0].numpy()  # (96, 96, 96)
            brain_mask = mri_vol > 0.05
            brain_coverage = float(brain_mask.sum() / brain_mask.size)

            subject_entry = {
                "subject_id": sid,
                "true_label": "PD" if true_label == 1 else "HC",
                "predicted_label": "PD" if pred_class == 1 else "HC",
                "correct": pred_class == true_label,
                "confidence": float(probs[pred_class]),
                "pd_probability": float(probs[1]),
                "gate_weights": {
                    "speech": round(float(gates[0]), 4),
                    "mri": round(float(gates[1]), 4),
                    "clinical": round(float(gates[2]), 4),
                },
                "mri_analysis": {
                    "embedding_l2_norm": round(float(np.linalg.norm(mri_emb)), 4),
                    "embedding_mean": round(float(mri_emb.mean()), 6),
                    "embedding_std": round(float(mri_emb.std()), 6),
                    "brain_coverage_pct": round(brain_coverage * 100, 1),
                    "volume_mean_intensity": round(float(mri_vol[brain_mask].mean()), 4) if brain_mask.any() else 0,
                    "volume_std_intensity": round(float(mri_vol[brain_mask].std()), 4) if brain_mask.any() else 0,
                    "has_jpeg_slices": has_axial and has_coronal and has_sagittal,
                },
            }
            subjects_data.append(subject_entry)
            all_gate_speech.append(gates[0])
            all_gate_mri.append(gates[1])
            all_gate_clinical.append(gates[2])

    # Aggregate statistics
    correct_count = sum(1 for s in subjects_data if s["correct"])
    total = len(subjects_data)

    y_true = [1 if s["true_label"] == "PD" else 0 for s in subjects_data]
    y_pred = [1 if s["predicted_label"] == "PD" else 0 for s in subjects_data]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    result = {
        "model_epoch": ckpt["epoch"],
        "test_accuracy": round(correct_count / total, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "n_subjects": total,
        "aggregate_gate_weights": {
            "speech_mean": round(float(np.mean(all_gate_speech)), 4),
            "mri_mean": round(float(np.mean(all_gate_mri)), 4),
            "clinical_mean": round(float(np.mean(all_gate_clinical)), 4),
            "speech_std": round(float(np.std(all_gate_speech)), 4),
            "mri_std": round(float(np.std(all_gate_mri)), 4),
            "clinical_std": round(float(np.std(all_gate_clinical)), 4),
        },
        "subjects": subjects_data,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[+] Exported MRI showcase data: {output_path}")
    print(f"    Test accuracy: {result['test_accuracy']}")
    print(f"    Gate weights (mean): Speech={result['aggregate_gate_weights']['speech_mean']:.3f}"
          f" | MRI={result['aggregate_gate_weights']['mri_mean']:.3f}"
          f" | Clinical={result['aggregate_gate_weights']['clinical_mean']:.3f}")
    print(f"    Subjects with JPEG slices: {sum(1 for s in subjects_data if s['mri_analysis']['has_jpeg_slices'])}/{total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
