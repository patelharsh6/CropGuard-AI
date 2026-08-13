"""Diagnostic: dynamic-range quantization (weights-only, float32 I/O).

Converts cropguard_v1.keras with only Optimize.DEFAULT — no representative
dataset, no forced int8 I/O types.  Evaluates accuracy on the full test split
using the same _read_image / label_map logic as the INT8 eval.

Run:  python -m scripts.diag_dynamic_range
"""

import os
import numpy as np
import tensorflow as tf

from src.quantize import (
    MODEL_PATH, FLOAT32_BASELINE_ACC,
    _load_split, _read_image,
)

DYNRANGE_PATH = os.path.join('models', 'cropguard_v1_dynrange.tflite')

print("=" * 70)
print("Diagnostic 2: dynamic-range quantization")
print("=" * 70)

# ── Convert (or reuse) ────────────────────────────────────────────────────
if os.path.exists(DYNRANGE_PATH):
    print(f"\nFound {DYNRANGE_PATH} — skipping conversion.")
    with open(DYNRANGE_PATH, 'rb') as f:
        tflite_bytes = f.read()
else:
    print(f"\nLoading {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Converting with dynamic-range quantization (weights only) …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # No representative_dataset, no inference_input_type / output_type
    tflite_bytes = converter.convert()
    with open(DYNRANGE_PATH, 'wb') as f:
        f.write(tflite_bytes)
    print(f"Saved {DYNRANGE_PATH}  ({len(tflite_bytes)/1e6:.2f} MB)")

size_mb = os.path.getsize(DYNRANGE_PATH) / 1e6
print(f"File: {DYNRANGE_PATH}  ({size_mb:.2f} MB)")

# ── Inspect I/O types ─────────────────────────────────────────────────────
interp = tf.lite.Interpreter(model_content=tflite_bytes)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"\nInput  dtype={inp['dtype']}  quantization={inp['quantization']}")
print(f"Output dtype={out['dtype']}  quantization={out['quantization']}")

# ── Evaluate ──────────────────────────────────────────────────────────────
test_df, label_map = _load_split('test')
n = len(test_df)
print(f"\nEvaluating {n} test images …", flush=True)

correct = 0
PROGRESS = 500
for i, (_, row) in enumerate(test_df.iterrows(), start=1):
    img = _read_image(row['image_path'])
    interp.set_tensor(inp['index'], img[np.newaxis])   # float32, no quantize step
    interp.invoke()
    scores = interp.get_tensor(out['index'])[0]        # float32 directly
    if int(scores.argmax()) == label_map[row['label']]:
        correct += 1
    if i % PROGRESS == 0:
        print(f"  {i}/{n}  running acc = {correct/i:.4f}", flush=True)

acc = correct / n
print(f"\nRESULT: dynamic-range accuracy = {acc:.4f}  "
      f"(float32 baseline {FLOAT32_BASELINE_ACC:.4f}, "
      f"delta {acc - FLOAT32_BASELINE_ACC:+.4f})", flush=True)
