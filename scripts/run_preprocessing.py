"""
Main CLI Runner for Parkinson's Multimodal Preprocessing Pipeline.
Processes speech audio, 3D structural MRI, and PPMI/UCI clinical data.
If raw datasets are not detected in data/raw, it automatically generates
a realistic 120-subject benchmark cohort matching PPMI/UCI schemas.
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing import (
    SyntheticParkinsonsDatasetGenerator,
    MultimodalDatasetBuilder,
    AudioSpeechPreprocessor,
    MRI3DPreprocessor,
    ClinicalTabularPreprocessor,
)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Preprocessing for Parkinson's Disease CDSS")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root data directory")
    parser.add_argument("--n_synthetic_samples", type=int, default=120, help="Number of cohort samples if generating")
    parser.add_argument("--force_generate", action="store_true", help="Force regenerate synthetic cohort")
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, "raw")
    processed_dir = os.path.join(args.data_dir, "processed")
    clinical_csv = os.path.join(raw_dir, "clinical_data.csv")
    audio_dir = os.path.join(raw_dir, "audio")
    mri_dir = os.path.join(raw_dir, "mri")

    # Step 1: Check or generate cohort
    if args.force_generate or not os.path.exists(clinical_csv) or not os.path.exists(audio_dir) or not os.path.exists(mri_dir):
        print("[*] Generating benchmark multimodal cohort matching PPMI & UCI schemas...")
        generator = SyntheticParkinsonsDatasetGenerator(random_seed=42)
        df_clinical = generator.export_full_multimodal_dataset(
            output_dir=args.data_dir,
            n_samples=args.n_synthetic_samples,
            pd_ratio=0.60,
        )
        print(f"[+] Successfully generated cohort with {len(df_clinical)} subjects.")
    else:
        print(f"[*] Found existing raw dataset in: {raw_dir}")

    # Step 2: Initialize Preprocessors
    audio_prep = AudioSpeechPreprocessor(
        sample_rate=16000,
        target_duration=3.0,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
        n_mfcc=13,
    )
    mri_prep = MRI3DPreprocessor(
        target_shape=(96, 96, 96),
        clip_percentiles=(1.0, 99.0),
        apply_skull_strip=True,
    )
    clinical_prep = ClinicalTabularPreprocessor(target_label_col="diagnosis")

    builder = MultimodalDatasetBuilder(
        audio_preprocessor=audio_prep,
        mri_preprocessor=mri_prep,
        clinical_preprocessor=clinical_prep,
    )

    # Step 3: Run Full Multimodal Preprocessing & Stratified Splitting
    datasets = builder.build_and_preprocess(
        clinical_csv_path=clinical_csv,
        audio_dir=audio_dir,
        mri_dir=mri_dir,
        output_processed_dir=processed_dir,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    # Step 4: Validate Preprocessed Dataset
    train_ds = datasets["train"]
    val_ds = datasets["val"]
    test_ds = datasets["test"]

    sample_item = train_ds[0]
    print("\n========================================================")
    print("      MULTIMODAL PREPROCESSING COMPLETED SUCCESSFULLY    ")
    print("========================================================")
    print(f"Total Cohort Size     : {len(train_ds) + len(val_ds) + len(test_ds)}")
    print(f"Training Set Size     : {len(train_ds)} subjects")
    print(f"Validation Set Size   : {len(val_ds)} subjects")
    print(f"Test Set Size         : {len(test_ds)} subjects")
    print("\n--- Tensor Specifications for Deep Learning Encoders ---")
    print(f"Speech Spectrogram (CNN+BiLSTM) : {tuple(sample_item['speech_spec'].shape)}  dtype={sample_item['speech_spec'].dtype}")
    print(f"Acoustic Features (UCI Schema)  : {tuple(sample_item['acoustic_vec'].shape)}   dtype={sample_item['acoustic_vec'].dtype}")
    print(f"3D MRI Scan (3D ResNet-50)      : {tuple(sample_item['mri_tensor'].shape)} dtype={sample_item['mri_tensor'].dtype}")
    print(f"Clinical Profile (Tabular MLP)  : {tuple(sample_item['clinical_vec'].shape)}   dtype={sample_item['clinical_vec'].dtype}")
    print(f"Modality Presence Mask          : {sample_item['modality_mask'].numpy().tolist()} [Speech, MRI, Clinical]")
    print(f"Diagnosis Ground Truth          : {sample_item['label'].item()} ({'PD' if sample_item['label'].item()==1 else 'HC'})")
    print(f"Saved Processed Directory       : {os.path.abspath(processed_dir)}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
