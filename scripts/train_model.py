"""
Model Training Script — End-to-End Multimodal PD Classification.

Loads preprocessed .npz data splits, trains the MultimodalPDModel
(Speech CNN+BiLSTM + 3D ResNet MRI + Clinical MLP → GMU → Classifier),
logs metrics per epoch, saves checkpoints, and generates training curves.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 100 --batch_size 8 --lr 0.001
    python scripts/train_model.py --use_full_resnet  (for GPU training)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import MultimodalPDModel, CBREngine


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_npz_split(path: str) -> dict:
    """Load a preprocessed .npz split into tensors."""
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


def make_dataloader(split_data: dict, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from a split dict."""
    dataset = TensorDataset(
        split_data["speech_spec"],
        split_data["acoustic_vec"],
        split_data["mri_tensor"],
        split_data["clinical_vec"],
        split_data["modality_mask"],
        split_data["label"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ═══════════════════════════════════════════════════════════════
# Training & Evaluation Functions
# ═══════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device, gate_reg=0.1):
    """Train for one epoch, return average loss and predictions."""
    model.train()
    total_loss = 0.0
    total_gate_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        speech, acoustic, mri, clinical, mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        output = model(speech, acoustic, mri, clinical, mask)
        ce_loss = criterion(output["logits"], labels)

        # Gate entropy regularization to prevent modality collapse
        gate_entropy_loss = model.fusion.compute_gate_entropy_loss()
        loss = ce_loss + gate_reg * gate_entropy_loss

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += ce_loss.item() * labels.size(0)
        total_gate_loss += gate_entropy_loss.item() * labels.size(0)
        preds = output["logits"].argmax(dim=1).cpu().numpy()
        probs = output["probabilities"][:, 1].detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = total_loss / len(all_labels)
    avg_gate_loss = total_gate_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    metrics["gate_entropy_loss"] = avg_gate_loss
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on val/test set, return metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_gate_values = []

    for batch in dataloader:
        speech, acoustic, mri, clinical, mask, labels = [b.to(device) for b in batch]

        output = model(speech, acoustic, mri, clinical, mask)
        loss = criterion(output["logits"], labels)

        total_loss += loss.item() * labels.size(0)
        preds = output["logits"].argmax(dim=1).cpu().numpy()
        probs = output["probabilities"][:, 1].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        all_gate_values.append(output["gate_values"].cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    metrics["gate_values_mean"] = np.concatenate(all_gate_values, axis=0).mean(axis=0).tolist()
    return metrics


def compute_metrics(labels, preds, probs):
    """Compute classification metrics."""
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5  # Only one class in batch

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn + 1e-8)  # True Positive Rate (Recall for PD)
    specificity = tn / (tn + fp + 1e-8)  # True Negative Rate

    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ═══════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, save_path: str):
    """Generate training/validation loss and metric curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Multimodal PD Model — Training Progress", fontsize=16, fontweight="bold")

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], "r--", label="Val", linewidth=2)
    axes[0, 0].set_title("Loss (CrossEntropy)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, history["val_acc"], "r--", label="Val", linewidth=2)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUC-ROC
    axes[0, 2].plot(epochs, history["train_auc"], "b-", label="Train", linewidth=2)
    axes[0, 2].plot(epochs, history["val_auc"], "r--", label="Val", linewidth=2)
    axes[0, 2].set_title("AUC-ROC")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylim([0, 1.05])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(epochs, history["train_f1"], "b-", label="Train", linewidth=2)
    axes[1, 0].plot(epochs, history["val_f1"], "r--", label="Val", linewidth=2)
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Sensitivity & Specificity
    axes[1, 1].plot(epochs, history["val_sensitivity"], "g-", label="Sensitivity (Recall)", linewidth=2)
    axes[1, 1].plot(epochs, history["val_specificity"], "m--", label="Specificity", linewidth=2)
    axes[1, 1].set_title("Sensitivity & Specificity (Val)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim([0, 1.05])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Modality Gate Weights
    if history.get("gate_speech"):
        axes[1, 2].plot(epochs, history["gate_speech"], label="Speech", linewidth=2, color="#2196F3")
        axes[1, 2].plot(epochs, history["gate_mri"], label="MRI", linewidth=2, color="#4CAF50")
        axes[1, 2].plot(epochs, history["gate_clinical"], label="Clinical", linewidth=2, color="#FF9800")
        axes[1, 2].set_title("Modality Gate Weights (Val)")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Training curves saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Multimodal PD Model")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                        help="Directory containing train/val/test .npz files")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (small due to 3D MRI memory)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--gate_reg", type=float, default=0.1,
                        help="Gate entropy regularization weight (0=disabled)")
    parser.add_argument("--use_full_resnet", action="store_true",
                        help="Use full 3D ResNet-50 (GPU recommended)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--reports_dir", type=str, default="./reports",
                        help="Directory for training curve plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MULTIMODAL PARKINSON'S DISEASE MODEL TRAINING")
    print(f"{'='*60}")
    print(f"  Device:      {device}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch Size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Gate Reg:    {args.gate_reg}")
    print(f"  MRI Encoder: {'Full 3D ResNet-50' if args.use_full_resnet else 'Light 3D CNN'}")
    print(f"{'='*60}\n")

    # ─── Load Data ───
    print("[1/5] Loading preprocessed data splits...")
    train_data = load_npz_split(os.path.join(args.data_dir, "train_multimodal.npz"))
    val_data = load_npz_split(os.path.join(args.data_dir, "val_multimodal.npz"))
    test_data = load_npz_split(os.path.join(args.data_dir, "test_multimodal.npz"))

    train_loader = make_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_data, args.batch_size, shuffle=False)
    test_loader = make_dataloader(test_data, args.batch_size, shuffle=False)

    n_train = len(train_data["label"])
    n_val = len(val_data["label"])
    n_test = len(test_data["label"])
    print(f"    Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"    Train PD/HC: {(train_data['label']==1).sum()}/{(train_data['label']==0).sum()}")

    # ─── Build Model ───
    print("\n[2/5] Building model architecture...")
    clinical_dim = train_data["clinical_vec"].shape[1]
    acoustic_dim = train_data["acoustic_vec"].shape[1]

    model = MultimodalPDModel(
        clinical_input_dim=clinical_dim,
        acoustic_input_dim=acoustic_dim,
        use_light_mri=not args.use_full_resnet,
        dropout=0.3,
        modality_dropout=0.2,
    ).to(device)

    param_counts = model.count_parameters()
    print(f"    Parameter counts:")
    for name, count in param_counts.items():
        print(f"      {name:20s}: {count:>10,}")

    # ─── Loss, Optimizer, Scheduler ───
    print("\n[3/5] Setting up training...")

    # Class weights for imbalanced PD/HC
    n_pd = (train_data["label"] == 1).sum().item()
    n_hc = (train_data["label"] == 0).sum().item()
    class_weights = torch.tensor([n_train / (2 * n_hc + 1e-8),
                                  n_train / (2 * n_pd + 1e-8)],
                                 dtype=torch.float32).to(device)
    print(f"    Class weights: HC={class_weights[0]:.3f}, PD={class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential learning rates: lower for speech (already learned),
    # higher for MRI encoder (needs to catch up)
    param_groups = [
        {"params": model.speech_encoder.parameters(), "lr": args.lr * 0.5},
        {"params": model.mri_encoder.parameters(), "lr": args.lr * 2.0},
        {"params": model.clinical_encoder.parameters(), "lr": args.lr * 1.5},
        {"params": model.fusion.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                      eta_min=1e-6)

    # ─── Training Loop ───
    print(f"\n[4/5] Training for {args.epochs} epochs...")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "train_auc": [], "val_auc": [],
        "val_sensitivity": [], "val_specificity": [],
        "gate_speech": [], "gate_mri": [], "gate_clinical": [],
        "lr": [],
    }

    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device, gate_reg=args.gate_reg)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_auc"].append(train_metrics["auc_roc"])
        history["val_auc"].append(val_metrics["auc_roc"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["val_specificity"].append(val_metrics["specificity"])
        history["lr"].append(current_lr)

        if val_metrics.get("gate_values_mean"):
            gate_means = val_metrics["gate_values_mean"]
            history["gate_speech"].append(gate_means[0])
            history["gate_mri"].append(gate_means[1])
            history["gate_clinical"].append(gate_means[2])

        # Composite validation score: balances ranking quality (AUC),
        # classification quality (F1), and overall accuracy.
        # This prevents saving a model that has perfect AUC but predicts all one class.
        val_composite = (0.4 * val_metrics["auc_roc"]
                         + 0.4 * val_metrics["f1"]
                         + 0.2 * val_metrics["accuracy"])

        # Print epoch summary
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.3f}/{val_metrics['accuracy']:.3f} | "
              f"AUC: {train_metrics['auc_roc']:.3f}/{val_metrics['auc_roc']:.3f} | "
              f"F1: {val_metrics['f1']:.3f} | "
              f"Score: {val_composite:.3f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        # Checkpoint best model (composite score ensures F1 > 0)
        if val_composite > best_val_score:
            best_val_score = val_composite
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_metrics["auc_roc"],
                "val_f1": val_metrics["f1"],
                "val_composite": val_composite,
                "val_metrics": val_metrics,
                "param_counts": param_counts,
            }
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"    * New best model saved (Score={val_composite:.4f}, "
                  f"AUC={val_metrics['auc_roc']:.3f}, F1={val_metrics['f1']:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  [!] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best model: epoch {best_epoch} with val Score={best_val_score:.4f}")

    # ─── Final Test Evaluation ───
    print(f"\n[5/5] Final evaluation on test set...")

    # Load best model
    ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_model.pt"),
                       weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS (Best Model — Epoch {best_epoch})")
    print(f"{'='*60}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:    {test_metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {test_metrics['auc_roc']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f} (PD recall)")
    print(f"  Specificity: {test_metrics['specificity']:.4f} (HC recall)")
    print(f"  Confusion Matrix:")
    print(f"               Pred HC  Pred PD")
    print(f"    True HC:    {test_metrics['tn']:4d}    {test_metrics['fp']:4d}")
    print(f"    True PD:    {test_metrics['fn']:4d}    {test_metrics['tp']:4d}")
    if test_metrics.get("gate_values_mean"):
        g = test_metrics["gate_values_mean"]
        print(f"  Gate Weights: Speech={g[0]:.3f} | MRI={g[1]:.3f} | Clinical={g[2]:.3f}")
    print(f"{'='*60}")

    # ─── Build CBR Case Library ───
    print("\n[*] Building CBR case library from training data...")
    cbr = CBREngine(embedding_dim=model.fusion_output_dim, k=5, similarity="cosine")

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            speech, acoustic, mri, clinical, mask, labels = [b.to(device) for b in batch]
            embeddings = model.get_embedding(speech, acoustic, mri, clinical, mask)
            for i in range(len(labels)):
                cbr.add_case(
                    embedding=embeddings[i].cpu().numpy(),
                    label=labels[i].item(),
                    subject_id=train_data["subject_id"][i] if i < len(train_data["subject_id"]) else f"train_{i}",
                )

    cbr.save(os.path.join(args.checkpoint_dir, "cbr_case_library.npz"))

    # Demo CBR query on first test sample
    with torch.no_grad():
        test_batch = next(iter(test_loader))
        speech, acoustic, mri, clinical, mask, labels = [b.to(device) for b in test_batch]
        test_emb = model.get_embedding(speech[:1], acoustic[:1], mri[:1], clinical[:1], mask[:1])
        cbr_result = cbr.query(test_emb[0].cpu().numpy())

    print(f"\n  CBR Demo — Test Patient (True label: {'PD' if labels[0]==1 else 'HC'}):")
    print(f"    CBR Prediction: {cbr_result['prediction']['predicted_label_str']} "
          f"(confidence={cbr_result['prediction']['confidence']:.3f})")
    print(f"    Retrieved {len(cbr_result['retrieved_cases'])} patient twins:")
    for twin in cbr_result["retrieved_cases"][:3]:
        print(f"      #{twin['rank']} {twin['subject_id']}: {twin['label_str']} "
              f"(similarity={twin['similarity']:.4f})")

    # ─── Save Training Curves ───
    os.makedirs(args.reports_dir, exist_ok=True)
    plot_training_curves(history, os.path.join(args.reports_dir, "training_curves.png"))

    # Save history JSON
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[+] Training history saved: {history_path}")

    # Save test results JSON
    results_path = os.path.join(args.checkpoint_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "test_metrics": test_metrics,
            "param_counts": param_counts,
            "training_time_seconds": round(total_time, 1),
            "device": str(device),
            "args": vars(args),
        }, f, indent=2)
    print(f"[+] Test results saved: {results_path}")

    print(f"\n{'='*60}")
    print(f"  TRAINING PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Best checkpoint : {os.path.abspath(os.path.join(args.checkpoint_dir, 'best_model.pt'))}")
    print(f"  Training curves : {os.path.abspath(os.path.join(args.reports_dir, 'training_curves.png'))}")
    print(f"  CBR library     : {os.path.abspath(os.path.join(args.checkpoint_dir, 'cbr_case_library.npz'))}")
    print(f"  Test results    : {os.path.abspath(results_path)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
