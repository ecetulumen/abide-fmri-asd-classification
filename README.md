# ABIDE ASD Classification with `rois_ho` Functional Connectivity

This repository contains an ASD-versus-TD classification pipeline based on
Pearson functional-connectivity features derived from the ABIDE dataset. It
also creates group-level ASD and TD connectivity maps for the Harvard-Oxford
(`rois_ho`) atlas.

## Pipeline

1. Load precomputed Pearson-FC vectors from an NPZ file.
2. Apply ANOVA F-score feature selection inside each training fold.
3. Standardize features using statistics learned from the training fold only.
4. Train a class-weighted MLP with Gaussian input noise, label smoothing,
   dropout, AdamW and early stopping.
5. Evaluate with stratified 10-fold cross-validation.
6. Save accuracy, balanced accuracy, macro F1, ROC-AUC, confusion matrix,
   training curves and fold-level results.
7. Reconstruct 111 x 111 FC matrices from the 6,105 upper-triangle edges and
   visualize ASD mean, TD mean, ASD-TD difference and edge distributions.

## Repository structure

```text
abide-rois-ho-classification/
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters
│   ├── data.py            # NPZ loading and FC matrix reconstruction
│   ├── metrics.py         # Evaluation and history utilities
│   ├── models.py          # PyTorch MLP
│   ├── plots.py           # Training and Pearson-map figures
│   └── training.py        # One-fold training and inference
├── data/
│   └── README.md          # Expected input format
├── docs/
│   └── figures/           # Selected figures for the public README
├── outputs/               # Generated locally; ignored by Git
├── train_model.py         # 10-fold training entry point
├── visualize_fc.py        # Pearson-map entry point
├── requirements.txt
└── GITHUB_UPLOAD_GUIDE_TR.md
```

## Expected data

The dataset is not included. The input NPZ file must contain:

- `X`: shape `(n_subjects, 6105)`, Pearson-FC upper-triangle vectors
- `y`: shape `(n_subjects,)`, where `0=TD` and `1=ASD`
- `subjects`: shape `(n_subjects,)`, unique subject identifiers

See [`data/README.md`](data/README.md) for details.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Training

Run from the repository root:

```bash
python train_model.py \
  --data-path data/processed/cpac_rois_ho_FILTERED_corr_features_FIXED.npz \
  --output-dir outputs/training
```

Use `--device cuda` to require a CUDA GPU, `--device cpu` for CPU, or leave the
default `--device auto`.

## Pearson FC maps

```bash
python visualize_fc.py \
  --data-path data/processed/cpac_rois_ho_FILTERED_corr_features_FIXED.npz \
  --output-dir outputs/pearson_maps
```

## Google Colab

After cloning the repository, mount Drive and run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/KULLANICI_ADIN/ABIDE-rois-ho-classification.git
%cd /content/abide-rois-ho-classification
!pip install -r requirements.txt

DATA_PATH = "/content/drive/MyDrive/ABIDE_MULTI_ATLAS/CPAC_FILTERED_MULTI_ATLAS_RESULTS_FIXED/feature_cache/cpac_rois_ho_FILTERED_corr_features_FIXED.npz"
TRAIN_OUTPUT = "/content/drive/MyDrive/ABIDE_MULTI_ATLAS/FINAL_ROIS_HO_PEARSON_MLP_BALANCED_LOSS"
MAP_OUTPUT = "/content/drive/MyDrive/ABIDE_MULTI_ATLAS/PEARSON_MAP_ROIS_HO"

!python train_model.py --data-path "$DATA_PATH" --output-dir "$TRAIN_OUTPUT" --device auto
!python visualize_fc.py --data-path "$DATA_PATH" --output-dir "$MAP_OUTPUT"
```

## Evaluation note

The unbiased primary summary uses out-of-fold probabilities with the fixed
threshold `0.50`. The code retains the original fold-wise threshold search as
an explicitly **exploratory** result because each threshold is selected and
scored on the same validation fold. For a paper or thesis claim based on an
optimized threshold, select the threshold in an inner validation loop and keep
the outer fold untouched.

The current early-stopping checkpoint is also selected from each validation
fold. A fully unbiased model-selection estimate requires nested
cross-validation.

## Reproducibility and privacy

- Feature selection and scaling are fitted only on each training fold.
- Duplicate subject identifiers cause an error to reduce leakage risk.
- Raw ABIDE files, cached NPZ features, trained weights and generated outputs
  are excluded through `.gitignore`.
- Add only a few non-sensitive result figures to `docs/figures/` when preparing
  the repository as a portfolio project.

