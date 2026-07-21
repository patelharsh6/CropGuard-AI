# CropGuard AI

CropGuard AI is a lightweight crop-disease image classifier for **Tomato, Potato, and Corn (maize)** leaves, built on the PlantVillage dataset. It fine-tunes a **MobileNetV3-Small** backbone into a 17-class classifier, then post-training-quantizes it to a compact INT8 TFLite model targeting on-device (mobile) inference.

The pipeline is deliberately mobile-first: a tiny classification head, realism-oriented augmentation to close the lab-to-field gap, and full INT8 quantization to keep the deployed model small and fast.

## Status at a glance

| Stage | Script | State |
|-------|--------|-------|
| Dataset split (stratified 70/15/15) | `src/data_pipeline.py` | Done — `data/dataset_split.csv` |
| Realism augmentation pipeline | `src/augmentation.py` | Done |
| Pipeline throughput benchmark | `src/benchmark_pipeline.py` | Done |
| Two-phase transfer-learning training | `src/train.py` | Done — `models/cropguard_v1.keras` |
| Resume / extended fine-tuning | `src/resume_phase2.py` | Done |
| Test-set evaluation + reports | `src/evaluate.py` | Done — `outputs/` |
| INT8 TFLite quantization | `src/quantize.py` | Ready to run |
| Mobile/app integration | `app/` | Not started (empty) |

**Headline metrics (float32 `cropguard_v1.keras`):**

- Test accuracy: **0.9465** (3,625 test images)
- Top-3 accuracy: **0.9961**
- Best validation accuracy across all phases: **0.9481**
- Macro-avg F1: **0.9402** | Weighted-avg F1: **0.9467**

## Project structure

```
cropguard-ai/
├── app/                        # (empty) future mobile/app integration
├── data/                       # gitignored — see "Dataset" below
│   ├── plantvillage_dataset/   # PlantVillage (color/grayscale/segmented)
│   ├── backgrounds/            # 12 procedural bg textures (soil/grass/wood/hand)
│   └── dataset_split.csv       # reproducible train/val/test split (24,164 rows)
├── models/
│   ├── cropguard_v1.keras      # best checkpoint (~10.1 MB, float32)
│   └── cropguard_v1_int8.tflite  # produced by src/quantize.py
├── notebooks/                  # (empty)
├── outputs/
│   ├── aug_preview.png         # augmentation sanity-check grids
│   ├── aug_preview_affine.png
│   ├── training_history.json   # per-epoch metrics, both phases + resume
│   ├── training_curves.png     # accuracy/loss curves with phase boundaries
│   ├── classification_report.json
│   └── confusion_matrix.png    # labeled 17×17 heatmap
├── src/
│   ├── config.py               # shared paths/constants
│   ├── data_pipeline.py        # dataset scan, split CSV, tf.data pipelines
│   ├── augmentation.py         # albumentations realism pipeline
│   ├── benchmark_pipeline.py   # input-pipeline throughput benchmark
│   ├── train.py                # two-phase transfer learning
│   ├── resume_phase2.py        # continue fine-tuning from checkpoint
│   ├── evaluate.py             # full test-set evaluation + reports
│   └── quantize.py             # INT8 TFLite conversion + eval + latency
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Key dependencies: TensorFlow 2.21 / Keras 3.15, Albumentations 2.0.8, OpenCV (headless), scikit-learn, pandas, matplotlib. Full pinned list in `requirements.txt`.

### Dataset

The [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) must be placed at `data/plantvillage_dataset/` (the `color/` variant is used). The `data/` directory is gitignored; the split CSV makes the exact train/val/test partition reproducible once the images are in place.

All scripts are run as modules from the project root:

```bash
python -m src.data_pipeline     # scan + create split CSV
python -m src.train             # two-phase training
python -m src.resume_phase2     # optional: extend phase-2 fine-tuning
python -m src.evaluate          # full test-set evaluation
python -m src.quantize          # INT8 TFLite conversion + re-eval + latency
```

## 1. Data pipeline (`src/data_pipeline.py`)

- Scans `data/plantvillage_dataset/color/` and keeps only the **Tomato, Potato, and Corn** class folders → **17 classes**, 24,164 images total.
- Creates a **stratified 70/15/15 split** (`train_test_split`, `random_state=42`): **16,914 train / 3,625 val / 3,625 test**, saved to `data/dataset_split.csv` (`image_path,label,split`) so every later script uses the identical partition.
- Builds `tf.data.Dataset` pipelines: read → `decode_jpeg` → resize to **224×224** → normalize to **[0, 1]** float32 → batch (32) → prefetch. Only the train split is shuffled, so val/test predictions stay aligned with CSV row order.
- Label mapping is `sorted(unique labels) → index`, used consistently across training, evaluation, and quantization.

The dataset is heavily imbalanced — e.g. `Potato___healthy` has ~152 training images vs ~5,357 for `Tomato___Tomato_Yellow_Leaf_Curl_Virus`. This is handled at training time via class weights (below), not oversampling.

## 2. Augmentation (`src/augmentation.py`)

PlantVillage photos are clean lab shots (single leaf, uniform background); real farm photos are not. The training-only Albumentations pipeline simulates field conditions, with all probabilities/intensities in a single tunable `AUG_CONFIG` dict:

- **Standard:** horizontal/vertical flips, affine scale ±20% + rotation ±30°, brightness/contrast jitter (±0.3, p=0.8).
- **Camera/environment realism:** motion blur (p=0.5), JPEG compression down to quality 40 (p=0.6), hue/saturation jitter as a color-temperature proxy (p=0.7).
- **Occlusion:** CoarseDropout — up to 4 black patches of 32×32 px (p=0.4), simulating shadows/obstructions.
- **Synthetic background paste (p=0.4)** — the highest-effort realism step: a custom `BackgroundReplace` transform segments the leaf (GrabCut on a downsized 128px copy, with an Otsu saturation-threshold fallback), feathers the alpha mask, and composites the leaf onto one of 12 procedurally generated textures (soil, grass, wood, hand/skin) in `data/backgrounds/`. Textures are auto-generated on first use, so the pipeline has no external downloads. Background paste runs *first* so subsequent blur/compression/lighting unify the composite.

Integration with `tf.data` is via `tf.numpy_function` (uint8 round-trip for Albumentations, back to [0,1] float32). Val/test data is **never** augmented. `visualize_augmentations()` writes a sanity-check grid to `outputs/aug_preview.png`.

`src/benchmark_pipeline.py` times the input pipeline in three configurations (no augmentation / augmentation without background paste / full) to isolate whether GrabCut is a throughput bottleneck.

## 3. Training (`src/train.py`)

Two-phase transfer learning on **MobileNetV3-Small** (ImageNet weights, `include_top=False`), chosen for its small footprint ahead of the <10 MB quantization goal.

**Architecture:** `Input(224,224,3)` → `Rescaling(255.)` (the pipeline outputs [0,1] but MobileNetV3's built-in preprocessing expects [0,255]) → MobileNetV3-Small backbone (called with `training=False` so BatchNorm stays in inference mode) → GlobalAveragePooling → Dropout(0.3) → Dense(17, softmax).

- **Phase 1 — head only** (backbone frozen): Adam @ 1e-3, up to 50 epochs. Ran **26 epochs**.
- **Phase 2 — fine-tune**: last ~25% of backbone layers unfrozen (BatchNorm kept frozen for stability), recompiled with Adam @ 1e-5, up to 30 epochs.
- **Class imbalance:** `'balanced'` class weights computed from the train split, passed to `model.fit(class_weight=...)`.
- **Callbacks:** ModelCheckpoint on best `val_accuracy` (phase 2 only overwrites the file if it beats phase 1's best, via `initial_value_threshold`), EarlyStopping (patience 6, restore best), ReduceLROnPlateau (×0.5, patience 3, min 1e-6), plus a per-epoch phase-tagged logger.
- **Artifacts:** best model → `models/cropguard_v1.keras`; both phases' history → `outputs/training_history.json`; combined accuracy/loss curves with a phase-boundary marker → `outputs/training_curves.png`.

> **Save format note:** the native Keras 3 `.keras` format is used instead of legacy `.h5`, because the MobileNetV3 backbone fails to round-trip through `.h5` under Keras 3 (breaks inside the built-in hard-swish activation). `.keras` saves/loads reliably and converts to TFLite via `from_keras_model()`.

### Resumed fine-tuning (`src/resume_phase2.py`)

The original phase 2 hit its 30-epoch cap while `val_accuracy` was still climbing (best 0.9454), so fine-tuning was resumed from the checkpoint for up to 15 more epochs at Adam @ 5e-6 (where ReduceLROnPlateau had already brought the LR). Same unfreeze config and checkpoint-threshold guard, so a worse run can never clobber the saved model. New epochs are appended to the phase-2 history and the curves are regenerated with a resume marker. Phase 2 totals **45 epochs** including the resumed segment; final best **val_accuracy 0.9481**.

## 4. Evaluation (`src/evaluate.py`)

Single unaugmented pass over the full held-out test split (3,625 images):

| Metric | Value |
|--------|-------|
| Overall test accuracy | **0.9465** |
| Top-3 accuracy | **0.9961** |
| Macro-avg F1 | 0.9402 |
| Weighted-avg F1 | 0.9467 |

Per-class highlights (full table in `outputs/classification_report.json`):

- `Corn___Common_rust_` is perfect (F1 1.000); most classes sit above 0.93 F1.
- Weakest classes: `Tomato___Early_blight` (F1 0.870) and `Tomato___Target_Spot` (F1 0.870) — visually similar brown-lesion diseases.
- The class-weighting spotlight: the two smallest training classes held up well — `Potato___healthy` F1 0.917 (support 23) and `Tomato___Tomato_mosaic_virus` F1 0.949 with perfect recall (support 56).

The script also writes a labeled 17×17 confusion-matrix heatmap (`outputs/confusion_matrix.png`) and prints the most-confused class pairs.

<!-- __REST2__ -->



