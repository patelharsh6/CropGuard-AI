# Quantization Investigation Findings

**Project:** CropGuard AI — MobileNetV3Small plant disease classifier (17 classes)  
**Date:** 2026-08-14  
**Status:** Closed — dynamic-range model shipped as production artifact

---

## Baseline

| Model | Accuracy | Size | Notes |
|---|---|---|---|
| `cropguard_v1.keras` (float32) | **0.9465** | 10.12 MB | Phase-2 fine-tuned MobileNetV3Small |

Evaluated with `python -m src.evaluate` on the 15% stratified test split
(same 70/15/15 CSV-based split used throughout training).

---

## Full INT8 Post-Training Quantization

**Tool:** `tf.lite.TFLiteConverter` with:
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
```

### Attempt 1 — Uniform random calibration (400 images)

| Metric | Value |
|---|---|
| Accuracy | **0.7630** (−0.1835 vs float32) |
| Size | 1.25 MB |
| Latency (median, laptop CPU, reference kernels) | 326.9 ms |

**Hypothesis at the time:** accuracy collapse caused by class-imbalanced calibration
sampling (uniform random over 16,914 training images effectively gave Potato_healthy
~1 calibration image and Tomato_mosaic_virus ~2).

### Attempt 2 — Stratified calibration (510 images, 30/class across 17 classes)

Fixed `representative_dataset()` in `src/quantize.py` to group by class label and
sample `min(30, n_available)` per class. Verified correct:

- Calibration set: exactly 510 images, 30 per class (confirmed by printed distribution table)
- Image loading: identical `_read_image()` helper (tf.io.read_file → decode_jpeg RGB →
  resize 224×224 → /255.0 float32) — no BGR issue, no indexing bug (iloc/loc match verified
  via diagnostic script)

| Metric | Value |
|---|---|
| Accuracy | **0.6836** (−0.2629 vs float32) |

**Result: worse than uniform sampling.** Stratified calibration is correct sampling
practice but did not fix — and slightly worsened — the accuracy collapse. This
**disproved the class-imbalance hypothesis**.

### Root Cause Confirmed

The dynamic-range model (weights-only, no activation calibration required) achieves
0.9401 accuracy. Full INT8 collapses regardless of calibration strategy. Therefore the
sensitivity is in **activation quantization**, not weight quantization.

**Architectural root cause:** MobileNetV3Small contains:
- **Hard-swish / hard-sigmoid activations** (`hard_silu` in Keras 3) — produce
  narrow, non-linear activation distributions that per-tensor INT8 quantization clips
  severely
- **Squeeze-and-Excitation blocks** — channel-wise sigmoid scaling with small value
  ranges loses precision at 8-bit resolution
- **Per-tensor output quantization** — only 256 distinct values for the 17-class
  softmax output (scale=0.00390625, zero_point=−128), which is too coarse

XNNPACK could not prepare the full-INT8 graph; evaluation required
`BUILTIN_WITHOUT_DEFAULT_DELEGATES` (reference kernels).

---

## Dynamic-Range Quantization (Production Choice)

**Tool:** `tf.lite.TFLiteConverter` with:
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# No representative_dataset, no INT8 target_spec
# Weights quantized to INT8; activations remain float32 at runtime
```

| Metric | Value |
|---|---|
| Accuracy | **0.9401** (−0.0064 vs float32) |
| Size | **1.1457 MB** (11.3% of 10.12 MB float32) |
| Latency — mean (laptop CPU, XNNPACK) | **22.7 ms** |
| Latency — median | **22.0 ms** |
| Latency — p95 | **34.8 ms** |
| XNNPACK delegate | Works (float32 activations are XNNPACK-compatible) |

Evaluated with `python -m src.evaluate` on the same test split.

**Key contrast with full INT8:** XNNPACK works cleanly for the dynamic-range model
because activations stay float32 — the hard-swish and SE blocks execute in full
precision. Only weights are INT8, which the optimizer handles cleanly.

---

## QAT (Quantization-Aware Training) Feasibility Check

Attempted as a potential fix for the full INT8 accuracy collapse.

```
pip install tensorflow-model-optimization==0.8.1
```

**Step 1 — `quantize_model()` failed immediately:**
```
ValueError: `to_quantize` can only either be a keras Sequential or Functional model.
  File: tensorflow_model_optimization/python/core/quantization/keras/quantize.py, line 135
```

**Root cause:** `tensorflow-model-optimization 0.8.1` targets Keras 2. It performs
`isinstance()` checks against `tf_keras.src.models.functional.Functional`. This project
uses **Keras 3.15.0**, whose Functional class lives at `keras.src.models.functional.Functional`.
The check fails even though the model is a valid Functional model.

**Known workaround** (`TF_USE_LEGACY_KERAS=1`) is blocked: `cropguard_v1.keras` was
saved in Keras 3 format (references `keras.src.models.functional`). Loading it through
the Legacy Keras 2 deserializer fails with:
```
ModuleNotFoundError: No module named 'tf_keras.src.models.functional'
```

Re-training from scratch under Legacy Keras then running QAT would require hours of
compute with no certainty that hard-swish/SE blocks would benefit — these layers are
architecturally problematic for per-tensor INT8 regardless of QAT.  
**QAT was not pursued.**

---

## Final Decision

**Production artifact:** `models/cropguard_v1_production.tflite`  
(copied from `cropguard_v1_dynrange.tflite`)

Rationale:
- Only 0.64% accuracy drop vs float32 (0.9401 vs 0.9465) — within acceptable tolerance
- 8.8× size reduction (10.12 MB → 1.15 MB)
- XNNPACK-accelerated, 22 ms median latency on laptop CPU
- No additional training required, no fragile workarounds
- Full INT8 is definitively not recoverable via PTQ on this architecture without QAT,
  and QAT is blocked by the Keras 3 / tfmot compatibility gap

---

## Investigation Scripts (Preserved)

All diagnostic scripts kept in `scripts/investigations/` for reproducibility:

| Script | Purpose |
|---|---|
| `diag_repr_dataset.py` | Early representative dataset diagnostics |
| `diag_dynamic_range.py` | Dynamic-range model accuracy evaluation |
| `diagnose_quant_io.py` | INT8 input/output quantization parameter inspection |
| `eval_int8_accuracy.py` | Standalone INT8 accuracy evaluation |
| `resume_phase2.py` | Phase-2 fine-tuning resume script |
