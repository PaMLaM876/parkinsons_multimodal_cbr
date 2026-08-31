"""
Model Evaluation Script — Comprehensive Test Metrics & Visualizations.

Loads a trained checkpoint and generates:
  1. Test set metrics (Accuracy, F1, AUC-ROC, Sensitivity, Specificity)
  2. ROC curve plot
  3. Confusion matrix heatmap
  4. Per-modality ablation study (train with each modality alone)
  5. Modality gate weight analysis
  6. CBR patient twin retrieval demo

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_auc_score,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import MultimodalPDModel, CBREngine


def load_npz_split(path: str) -> dict:
    """Load a preprocessed .npz split."""
    data = np.load(path, allow_pickle=True)
    return {
        "speech_spec": torch.tensor(data["speech_specs"], dtype=torch.float32),
        "acoustic_vec": torch.tensor(data["acoustic_feats"], dtype=torch.float32),
        "mri_tensor": torch.tensor(data["mri_tensors"], dtype=torch.float32),
        "clinical_vec": torch.tensor(data["clinical_vecs"], dtype=torch.float32),
        "modality_mask": torch.tensor(data["modality_masks"], dtype=torch.float32),
        "label": torch.tensor(data["labels"], dtype=torch.long),
        "subject_id": data["subject_ids"].tolist(),
    }


@torch.no_grad()
def get_predictions(model, data, device, modality_mask_override=None):
    """Run inference and collect predictions, probabilities, gate values."""
    model.eval()
    speech = data["speech_spec"].to(device)
    acoustic = data["acoustic_vec"].to(device)
    mri = data["mri_tensor"].to(device)
    clinical = data["clinical_vec"].to(device)

    if modality_mask_override is not None:
        mask = modality_mask_override.expand(speech.size(0), -1).to(device)
    else:
        mask = data["modality_mask"].to(device)

    labels = data["label"].numpy()

    # Process in batches to avoid OOM
    batch_size = 4
    all_probs = []
    all_preds = []
    all_gates = []
    all_embeddings = []

    for i in range(0, len(labels), batch_size):
        j = min(i + batch_size, len(labels))
        out = model(speech[i:j], acoustic[i:j], mri[i:j], clinical[i:j], mask[i:j])
        all_probs.append(out["probabilities"][:, 1].cpu().numpy())
        all_preds.append(out["logits"].argmax(dim=1).cpu().numpy())
        all_gates.append(out["gate_values"].cpu().numpy())
        emb = model.get_embedding(speech[i:j], acoustic[i:j], mri[i:j], clinical[i:j], mask[i:j])
        all_embeddings.append(emb.cpu().numpy())

    return {
        "labels": labels,
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
        "gates": np.concatenate(all_gates),
        "embeddings": np.concatenate(all_embeddings),
    }


def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve with AUC."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2.5,
            label=f"Multimodal Model (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — PD vs HC Classification", fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[+] ROC curve saved: {save_path}")
    return roc_auc


def plot_confusion_matrix(labels, preds, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    cmap = LinearSegmentedColormap.from_list("custom", ["#FFFFFF", "#1976D2"])
    im = ax.imshow(cm, cmap=cmap, aspect="auto")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=24, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["HC (Predicted)", "PD (Predicted)"], fontsize=12)
    ax.set_yticklabels(["HC (True)", "PD (True)"], fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=15, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[+] Confusion matrix saved: {save_path}")


def plot_gate_weights(gates, labels, save_path):
    """Plot modality gate weight distributions for PD vs HC."""
    modality_names = ["Speech", "MRI", "Clinical"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Modality Gate Weights — PD vs HC", fontsize=15, fontweight="bold")

    for i, (name, ax) in enumerate(zip(modality_names, axes)):
        pd_gates = gates[labels == 1, i]
        hc_gates = gates[labels == 0, i]

        ax.hist(hc_gates, bins=15, alpha=0.6, color="#4CAF50", label="HC", density=True)
        ax.hist(pd_gates, bins=15, alpha=0.6, color="#F44336", label="PD", density=True)
        ax.axvline(pd_gates.mean(), color="#F44336", linestyle="--", linewidth=2,
                   label=f"PD mean={pd_gates.mean():.3f}")
        ax.axvline(hc_gates.mean(), color="#4CAF50", linestyle="--", linewidth=2,
                   label=f"HC mean={hc_gates.mean():.3f}")
        ax.set_title(f"{name} Gate", fontsize=13)
        ax.set_xlabel("Gate Weight")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[+] Gate weights plot saved: {save_path}")


def modality_ablation(model, test_data, device):
    """Run ablation: test with each modality alone and all combined."""
    results = {}

    configs = {
        "All Modalities": torch.tensor([[1.0, 1.0, 1.0]]),
        "Speech Only": torch.tensor([[1.0, 0.0, 0.0]]),
        "MRI Only": torch.tensor([[0.0, 1.0, 0.0]]),
        "Clinical Only": torch.tensor([[0.0, 0.0, 1.0]]),
        "Speech + Clinical": torch.tensor([[1.0, 0.0, 1.0]]),
        "MRI + Clinical": torch.tensor([[0.0, 1.0, 1.0]]),
        "Speech + MRI": torch.tensor([[1.0, 1.0, 0.0]]),
    }

    for name, mask in configs.items():
        preds_data = get_predictions(model, test_data, device, modality_mask_override=mask)
        acc = accuracy_score(preds_data["labels"], preds_data["preds"])
        f1 = f1_score(preds_data["labels"], preds_data["preds"], average="binary", zero_division=0)
        try:
            auc_val = roc_auc_score(preds_data["labels"], preds_data["probs"])
        except ValueError:
            auc_val = 0.5
        results[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "auc_roc": round(auc_val, 4)}
        print(f"    {name:25s} → Acc={acc:.3f} | F1={f1:.3f} | AUC={auc_val:.3f}")

    return results


def plot_ablation(ablation_results, save_path):
    """Bar chart of ablation study results."""
    names = list(ablation_results.keys())
    accs = [ablation_results[n]["accuracy"] for n in names]
    aucs = [ablation_results[n]["auc_roc"] for n in names]
    f1s = [ablation_results[n]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, accs, width, label="Accuracy", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, aucs, width, label="AUC-ROC", color="#4CAF50", alpha=0.85)
    bars3 = ax.bar(x + width, f1s, width, label="F1 Score", color="#FF9800", alpha=0.85)

    ax.set_xlabel("Modality Configuration", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Modality Ablation Study — Test Set", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[+] Ablation chart saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Trained Multimodal PD Model")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--reports_dir", type=str, default="./reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  MODEL EVALUATION")
    print(f"{'='*60}\n")

    # Load test data
    print("[1/6] Loading test data...")
    test_data = load_npz_split(os.path.join(args.data_dir, "test_multimodal.npz"))

    # Load model
    print("[2/6] Loading trained model...")
    clinical_dim = test_data["clinical_vec"].shape[1]
    acoustic_dim = test_data["acoustic_vec"].shape[1]

    model = MultimodalPDModel(
        clinical_input_dim=clinical_dim,
        acoustic_input_dim=acoustic_dim,
        use_light_mri=True,
    ).to(device)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"    Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val AUC={ckpt.get('val_auc', 'N/A')}, F1={ckpt.get('val_f1', 'N/A')})")

    # Full evaluation
    print("\n[3/6] Running inference on test set...")
    results = get_predictions(model, test_data, device)

    acc = accuracy_score(results["labels"], results["preds"])
    f1 = f1_score(results["labels"], results["preds"], average="binary", zero_division=0)
    auc_val = roc_auc_score(results["labels"], results["probs"])

    print(f"\n    Test Accuracy:    {acc:.4f}")
    print(f"    Test F1:          {f1:.4f}")
    print(f"    Test AUC-ROC:     {auc_val:.4f}")
    print(f"\n    Classification Report:")
    print(classification_report(results["labels"], results["preds"],
                                target_names=["HC", "PD"], zero_division=0))

    # Plots
    os.makedirs(args.reports_dir, exist_ok=True)

    print("[4/6] Generating ROC curve...")
    plot_roc_curve(results["labels"], results["probs"],
                   os.path.join(args.reports_dir, "roc_curve.png"))

    print("[5/6] Generating confusion matrix...")
    plot_confusion_matrix(results["labels"], results["preds"],
                          os.path.join(args.reports_dir, "confusion_matrix.png"))

    plot_gate_weights(results["gates"], results["labels"],
                      os.path.join(args.reports_dir, "gate_weights.png"))

    # Ablation study
    print("\n[6/6] Running modality ablation study...")
    ablation = modality_ablation(model, test_data, device)
    plot_ablation(ablation, os.path.join(args.reports_dir, "ablation_study.png"))

    # Save all results
    eval_results = {
        "test_accuracy": acc,
        "test_f1": f1,
        "test_auc_roc": auc_val,
        "ablation": ablation,
        "gate_means": {
            "speech": float(results["gates"][:, 0].mean()),
            "mri": float(results["gates"][:, 1].mean()),
            "clinical": float(results["gates"][:, 2].mean()),
        },
    }
    results_path = os.path.join(args.reports_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[+] Evaluation results saved: {results_path}")

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Reports: {os.path.abspath(args.reports_dir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
