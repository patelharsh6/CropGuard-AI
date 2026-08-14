"""Diagnostic: inspect representative_dataset() yield values and class coverage.

Checks:
1. Value range and shape of actual yielded samples
2. Whether each call to converter.representative_dataset returns a FRESH generator
3. Class distribution of the 400 calibration indices

Run:  python -m scripts.diag_repr_dataset
"""

import numpy as np
import pandas as pd

from src.quantize import (
    SEED, N_REPRESENTATIVE,
    _load_split, _read_image, representative_dataset,
)

print("=" * 70)
print("Diagnostic 1: representative_dataset() inspection")
print("=" * 70)

train_df, label_map = _load_split('train')
print(f"\nTrain split: {len(train_df)} rows, {train_df['label'].nunique()} classes")

# ── 1. Print first 5 yielded samples ─────────────────────────────────────
print("\n--- First 5 yielded samples (shape, dtype, min, max) ---")
for i, sample in enumerate(representative_dataset(train_df)):
    arr = sample[0]          # the single array in the list
    print(f"  sample {i}: shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}")
    if i == 4:
        break

# ── 2. Fresh-generator check ──────────────────────────────────────────────
print("\n--- Fresh-generator check ---")
# Simulate what TFLiteConverter does: it calls the callable twice.
gen_factory = lambda: representative_dataset(train_df)

gen1 = gen_factory()
gen2 = gen_factory()

count1 = sum(1 for _ in gen1)
count2 = sum(1 for _ in gen2)
print(f"  Calling gen_factory() twice yields {count1} and {count2} samples.")
print(f"  Generators are {'DIFFERENT objects (fresh ✓)' if gen1 is not gen2 else 'SAME object (stale ✗)'}")

# ── 3. Class distribution of the 400 calibration indices ─────────────────
print("\n--- Class distribution of calibration samples ---")
rng = np.random.default_rng(SEED)
indices = rng.choice(len(train_df), size=min(N_REPRESENTATIVE, len(train_df)), replace=False)
calib_df = train_df.iloc[indices]
class_counts = calib_df['label'].value_counts().sort_index()
print(f"  {len(indices)} calibration images across {len(class_counts)} classes:")
for label, cnt in class_counts.items():
    print(f"    {label:35s}  {cnt:3d} images")

print(f"\n  Min samples/class: {class_counts.min()}  "
      f"Max: {class_counts.max()}  "
      f"Mean: {class_counts.mean():.1f}")
