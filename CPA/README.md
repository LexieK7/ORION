# CPA — Pathology Analysis

This repository is a framework adapted for H&E whole-slide image (WSI) analysis and classification. It contains utilities for WSI segmentation, patch extraction, feature embedding, training models, and evaluation.

This README explains how to prepare data, extract features (including using the CONCH model), create splits, train models, and run evaluation.

Table of Contents
- Overview
- Requirements
- Installation
- Data preparation (WSI -> patches -> HDF5 / PT)
- Feature extraction (ResNet / CONCH)
- Creating train/val/test splits
- Training (CLAM / MIL)
- Evaluation and inference
- Tips, placeholders and troubleshooting
- Citation

Overview
- The repo follows the CLAM design for weakly-supervised MIL on WSI-level labels. Key scripts:
  - `create_patches_fp.py`: WSI segmentation and patch extraction (produces .h5 patch bags).
  - `extract_features.py` / `extract_features_fp.py`: compute patch embeddings and save as `.h5` and `.pt` per-slide files.
  - `create_splits_seq.py`: generate k-fold train/val/test splits.
  - `main.py`: training orchestration (runs per-fold training with configurable options).
  - `external.py`, `eval.py`: evaluation helpers and scripts for fold-wise evaluation and ensembling.
  - `conch/open_clip_custom`: CONCH model factory, transforms and utilities.

Requirements
- Python 3.10
- PyTorch (compatible with your CUDA), torchvision
- numpy, pandas, scikit-learn, openpyxl, h5py, Pillow, openslide-python, timm, huggingface_hub
- Optional: tensorboardX for logging

Install (example)

```bash
conda env create -f env.yml
```

Data preparation
1. Raw WSIs: put your whole-slide files (.svs/.ndpi/.tiff) in a directory, e.g. `data/wsi/`.
2. Create patches and tissue masks: use `create_patches_fp.py`. This script performs segmentation, optional masking, patching, and stitching. Example usage:

```bash
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 1024 --seg --patch --stitch 
```

Output structure (example under `data/processed`):
- `patches/` : HDF5 bag files for each slide (slide_id.h5)
- `masks/` : visualization masks (png)
- `stitches/` : stitched heatmap images


Feature extraction

- CONCH  (recommended for semantic-rich embeddings)
  - The repo includes a CONCH/CoCa implementation under `conch/open_clip_custom`. Use `create_model_from_pretrained()` to load a CONCH visual model and its preprocess transform. 

  Minimal example snippet (Python) to extract features with CONCH:

```python
from conch.open_clip_custom.factory import create_model_from_pretrained
from torch.utils.data import DataLoader
import torch

# choose config name that exists in conch/open_clip_custom/model_configs
cfg_name = 'conch_ViT-B-16'  # example config filename without .json
checkpoint_path = 'hf_hub:owner/repo'  # or local checkpoint path
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, preprocess = create_model_from_pretrained(cfg_name, checkpoint_path=checkpoint_path, device=device, return_transform=True)
model.eval()

# Your dataset should apply `preprocess` to image patches and return a batch tensor
loader = DataLoader(my_patch_dataset, batch_size=128, collate_fn=my_collate)

with torch.no_grad():
    for batch, coords in loader:
        batch = batch.to(device)
        feats = model.visual.forward_no_head(batch)  # or model(batch) depending on config
        # save feats.cpu().numpy() and coords similarly to the standard extract_features script
```

Integration tips:
- `create_model_from_pretrained` will return a preprocessing transform (`preprocess`) you should use when constructing your patch dataset.
- If using HF hub URIs, set `hf_auth_token` in `create_model_from_pretrained` or configure `huggingface-cli`.

HDF5 / PT layout
- The `extract_features*.py` scripts save per-slide HDF5 files with two datasets: `features` (N x C) and `coords` (N x 2). They also save a PyTorch tensor `.pt` per slide under `feat_dir/pt_files/slide_id.pt` for faster downstream loading by the MIL dataset.

Creating splits
- Use `create_splits_seq.py` to create k-fold train/val/test splits (patient-stratified when appropriate). Example:

```bash
python create_splits_seq.py --task task_2_tumor_subtyping --k 5 --val_frac 0.1 --test_frac 0.1
```

This writes split CSVs under `splits/<task>_<label_frac>/` that are consumed by `main.py`.

Training
- Use `main.py` to train CLAM/MIL models. The script supports many command-line arguments — core usage:

```bash
python main.py --data_root_dir data/features --task task_2_tumor_subtyping --model_type clam_mb --k 5 --max_epochs 200 --lr 1e-5 --exp_code my_experiment
```

Key relevant arguments (see `main.py` for the full list):
- `--data_root_dir`: root where per-slide `.pt` feature files live.
- `--task`: `task_1_tumor_vs_normal` or `task_2_tumor_subtyping` (affects label dictionary and n_classes)
- `--model_type`: `clam_sb`, `clam_mb`, or `mil`.
- `--model_size`: `small` or `big` for CLAM visual head config.
- `--bag_loss`: `svm` or `ce` (slide-level loss).
- `--inst_loss`: instance-level clustering loss for CLAM (`svm` or `ce`)
- `--B`: number of sampled positive/negative patches in CLAM instance sampling.
- `--results_dir`: directory to store training outputs and checkpoints.

After training, checkpoints are saved as `s_{fold}_checkpoint.pt` in the designated results directory.

Evaluation
- Use `external.py` or `eval.py` to run per-fold evaluation and produce an Excel summary (per-fold sheets, ensemble mean/vote and a Summary sheet). Edit the file-level placeholders at the top (e.g., `data_path`, `csv_path`, `weight_dir`, `output_path`) before running, or adapt the script into a function for programmatic use.

Example evaluation command pattern:

```bash
python external.py
```

Tips, placeholders and troubleshooting
- Replace all placeholder strings (e.g., `DATA-PATH`, `MODEL-WEIGHT-DIR`, `PATH-TO-CTA-RESULT`) in scripts with real paths before running.
- If using CONCH models from Hugging Face Hub, ensure the appropriate checkpoint name and `hf_auth_token` are provided when loading.
- If you plan to use CONCH as embedding backbone, prefer the FP extractor (`extract_features_fp.py`) or adapt `extract_features.py` to apply the CONCH `preprocess` transform when building patch batches.
- For large cohorts, parallelize feature extraction by distributing slides across nodes or by batching slides per worker.
- GPU memory: tune `--batch_size` in `extract_features*.py` and `--B`/model_size in `main.py` to fit your GPU.

Repository structure (high level)

- `create_patches.py` — segmentation and patch generation using `wsi_core` utilities
- `extract_features.py`, `extract_features_fp.py` — embedders (ResNet checkpoint by default)
- `conch/` — CONCH / CoCa visual model and factory for loading pretrained checkpoints
- `datasets/` — dataset classes for HDF5 / PT bags and split handling
- `models/` — CLAM/MIL model implementations and utilities
- `utils/` — helper utilities: file IO, training loops, evaluation summary
- `splits/`, `dataset_csv/`, `vis_utils/`, `wsi_core/` — supporting assets and utilities

Citation
- This repository and code base are adapted from CLAM: Mahmoodlab/CLAM (Open source tools for computational pathology). Please cite the original work for academic use: https://github.com/mahmoodlab/CLAM
