"""Quick smoke test: verify model instantiates and forward pass works with correct tensor shapes."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

print("=" * 60)
print("  MODEL ARCHITECTURE SMOKE TEST")
print("=" * 60)

# 1. Test individual encoders
print("\n[1/4] Testing Clinical Encoder...")
from models.clinical_encoder import ClinicalTabularEncoder
clin_enc = ClinicalTabularEncoder(input_dim=37, embed_dim=64)
x_clin = torch.randn(4, 37)
e_clin = clin_enc(x_clin)
print(f"    Input: {x_clin.shape} → Output: {e_clin.shape}")
assert e_clin.shape == (4, 64), f"Expected (4, 64), got {e_clin.shape}"
print("    ✓ Clinical Encoder OK")

print("\n[2/4] Testing Speech Encoder...")
from models.speech_encoder import SpeechEncoder
speech_enc = SpeechEncoder(spec_channels=1, n_mels=64, acoustic_dim=22, embed_dim=256)
x_spec = torch.randn(4, 1, 64, 184)
x_acoustic = torch.randn(4, 22)
e_speech, attn_w = speech_enc(x_spec, x_acoustic)
print(f"    Spec: {x_spec.shape} + Acoustic: {x_acoustic.shape} → Embed: {e_speech.shape}, Attn: {attn_w.shape}")
assert e_speech.shape == (4, 256), f"Expected (4, 256), got {e_speech.shape}"
print("    ✓ Speech Encoder OK")

print("\n[3/4] Testing MRI Encoder (Light)...")
from models.mri_encoder import MRIResNet3DLight
mri_enc = MRIResNet3DLight(in_channels=1, embed_dim=256)
x_mri = torch.randn(2, 1, 96, 96, 96)
e_mri = mri_enc(x_mri)
print(f"    Input: {x_mri.shape} → Output: {e_mri.shape}")
assert e_mri.shape == (2, 256), f"Expected (2, 256), got {e_mri.shape}"
print("    ✓ MRI Encoder (Light) OK")

print("\n[4/4] Testing Full Multimodal Model...")
from models.multimodal_model import MultimodalPDModel
model = MultimodalPDModel(
    clinical_input_dim=37, acoustic_input_dim=42,
    use_light_mri=True, dropout=0.3
)

# Simulate a forward pass with batch of 2
B = 2
inputs = {
    "speech_spec": torch.randn(B, 1, 64, 184),
    "acoustic_vec": torch.randn(B, 22),
    "mri_tensor": torch.randn(B, 1, 96, 96, 96),
    "clinical_vec": torch.randn(B, 37),
    "modality_mask": torch.ones(B, 3),
}

output = model(**inputs)
print(f"    Logits: {output['logits'].shape}")
print(f"    Probabilities: {output['probabilities'].shape}")
print(f"    Embedding: {output['embedding'].shape}")
print(f"    Gate values: {output['gate_values'].shape}")
print(f"    Attn weights: {output['attn_weights'].shape}")
assert output["logits"].shape == (B, 2)
assert output["embedding"].shape == (B, 256)
print("    ✓ Full Model Forward Pass OK")

# Parameter counts
params = model.count_parameters()
print(f"\n  Parameter Summary:")
for name, count in params.items():
    print(f"    {name:20s}: {count:>10,}")

# Test backward pass
loss = torch.nn.CrossEntropyLoss()(output["logits"], torch.tensor([0, 1]))
loss.backward()
print(f"\n  ✓ Backward pass OK (loss={loss.item():.4f})")

# Test with real data shapes
print("\n[*] Testing with actual .npz data shapes...")
data = np.load("data/processed/train_multimodal.npz", allow_pickle=True)
print(f"    speech_specs: {data['speech_specs'].shape}")
print(f"    acoustic_feats: {data['acoustic_feats'].shape}")
print(f"    mri_tensors: {data['mri_tensors'].shape}")
print(f"    clinical_vecs: {data['clinical_vecs'].shape}")
print(f"    labels: {data['labels'].shape}")

# Run model on first 2 real samples
real_out = model(
    speech_spec=torch.tensor(data["speech_specs"][:2], dtype=torch.float32),
    acoustic_vec=torch.tensor(data["acoustic_feats"][:2], dtype=torch.float32),
    mri_tensor=torch.tensor(data["mri_tensors"][:2], dtype=torch.float32),
    clinical_vec=torch.tensor(data["clinical_vecs"][:2], dtype=torch.float32),
    modality_mask=torch.tensor(data["modality_masks"][:2], dtype=torch.float32),
)
print(f"    Real data forward pass → logits: {real_out['logits'].detach()}")
print(f"    Predictions: {real_out['probabilities'].detach().numpy()}")
print(f"    Gate weights: {real_out['gate_values'].detach().numpy()}")

print("\n" + "=" * 60)
print("  ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
