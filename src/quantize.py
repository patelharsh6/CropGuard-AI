"""
CropGuard AI — full INT8 post-training quantization of the trained Keras model.

Converts models/cropguard_v1.keras to a fully integer-quantized TFLite model
(models/cropguard_v1_int8.tflite) with tf.lite.TFLiteConverter, then:

- Reports file size before/after.
- Re-evaluates accuracy on the FULL test split (the same split evaluate.py
  uses) by running each image through tf.lite.Interpreter, and prints the
  delta against the float32 baseline (0.9465).
- Benchmarks single-image inference latency (mean/median/p95 over 100 runs
  after warmup) and flags if median exceeds the 2000 ms mobile target.
  NOTE: this is a laptop-CPU proxy, not a real low-end-Android measurement.

Run:  python -m src.quantize
"""

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_pipeline import CSV_PATH, IMG_SIZE

MODEL_PATH = os.path.join('models', 'cropguard_v1.keras')
TFLITE_PATH = os.path.join('models', 'cropguard_v1_int8.tflite')

FLOAT32_BASELINE_ACC = 0.9465   # reported test accuracy of the float32 .keras model
LATENCY_TARGET_MS = 2000.0      # project mobile latency target (median)
N_REPRESENTATIVE = 400          # train images sampled for calibration
N_LATENCY_RUNS = 100
N_WARMUP_RUNS = 5
SEED = 42


def _load_split(split: str):
    df = pd.read_csv(CSV_PATH)
    label_map = {lbl: i for i, lbl in enumerate(sorted(df['label'].unique()))}
    split_df = df[df['split'] == split].reset_index(drop=True)
    return split_df, label_map


def _read_image(path: str) -> np.ndarray:
    """Resize + normalize to [0,1] float32 — matches load_and_preprocess_image."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img.numpy()


def representative_dataset(train_df: pd.DataFrame):
    """Yields ~N_REPRESENTATIVE single-image batches for INT8 calibration."""
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(train_df), size=min(N_REPRESENTATIVE, len(train_df)),
                         replace=False)
    for idx in indices:
        img = _read_image(train_df.iloc[idx]['image_path'])
        yield [img[np.newaxis].astype(np.float32)]   # shape (1, H, W, 3)


def convert_to_int8(model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    train_df, _ = _load_split('train')
    converter.representative_dataset = lambda: representative_dataset(train_df)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def _make_interpreter(tflite_bytes: bytes) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    return interp


def _quantize_input(img: np.ndarray, input_detail: dict) -> np.ndarray:
    """Scale float32 [0,1] image to INT8 using interpreter's quant params."""
    scale, zero_point = input_detail['quantization']
    q = np.round(img / scale + zero_point).astype(np.int8)
    return q[np.newaxis]   # (1, H, W, 3)


def evaluate_tflite(tflite_bytes: bytes) -> float:
    """Run full test split through TFLite interpreter; return accuracy."""
    test_df, label_map = _load_split('test')
    interp = _make_interpreter(tflite_bytes)
    input_detail = interp.get_input_details()[0]
    output_detail = interp.get_output_details()[0]

    correct = 0
    for _, row in test_df.iterrows():
        img = _read_image(row['image_path'])
        inp = _quantize_input(img, input_detail)
        interp.set_tensor(input_detail['index'], inp)
        interp.invoke()
        logits = interp.get_tensor(output_detail['index'])[0]   # int8
        if int(logits.argmax()) == label_map[row['label']]:
            correct += 1

    return correct / len(test_df)


def benchmark_latency(tflite_bytes: bytes) -> dict:
    """Time N_LATENCY_RUNS single-image forward passes; return stats in ms."""
    interp = _make_interpreter(tflite_bytes)
    input_detail = interp.get_input_details()[0]
    output_detail = interp.get_output_details()[0]

    # Use a fixed dummy image for consistent timing.
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.int8)

    for _ in range(N_WARMUP_RUNS):
        interp.set_tensor(input_detail['index'], dummy)
        interp.invoke()

    times_ms = []
    for _ in range(N_LATENCY_RUNS):
        t0 = time.perf_counter()
        interp.set_tensor(input_detail['index'], dummy)
        interp.invoke()
        _ = interp.get_tensor(output_detail['index'])
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        'mean':   float(arr.mean()),
        'median': float(np.median(arr)),
        'p95':    float(np.percentile(arr, 95)),
    }


def main():
    print("=" * 70)
    print("CropGuard AI — INT8 Post-Training Quantization")
    print("=" * 70)

    # ── 1. Load Keras model ──────────────────────────────────────────────────
    print(f"\nLoading {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)

    keras_size = os.path.getsize(MODEL_PATH)
    print(f"Keras model size: {keras_size / 1e6:.2f} MB")

    # ── 2-3. Convert to INT8 TFLite ──────────────────────────────────────────
    print(f"\nConverting to INT8 TFLite (calibrating on ~{N_REPRESENTATIVE} "
          "train images) …")
    tflite_bytes = convert_to_int8(model)

    # ── 4. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_bytes)
    tflite_size = os.path.getsize(TFLITE_PATH)
    print(f"Saved {TFLITE_PATH}  ({tflite_size / 1e6:.2f} MB)")

    # ── 5. Accuracy evaluation ───────────────────────────────────────────────
    print("\nEvaluating accuracy on full test split …")
    int8_acc = evaluate_tflite(tflite_bytes)
    acc_delta = int8_acc - FLOAT32_BASELINE_ACC

    # ── 6. Latency benchmark ─────────────────────────────────────────────────
    print(f"\nBenchmarking latency ({N_WARMUP_RUNS} warmup + "
          f"{N_LATENCY_RUNS} timed runs) …")
    lat = benchmark_latency(tflite_bytes)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QUANTIZATION SUMMARY")
    print("=" * 70)
    print(f"File size   : {keras_size / 1e6:.2f} MB  (float32 .keras)  →  "
          f"{tflite_size / 1e6:.2f} MB  (INT8 .tflite)  "
          f"[{tflite_size / keras_size * 100:.1f}% of original]")
    print(f"Accuracy    : {FLOAT32_BASELINE_ACC:.4f} (float32 baseline)  →  "
          f"{int8_acc:.4f} (INT8)  "
          f"[delta {acc_delta:+.4f}]")
    print(f"Latency     : mean {lat['mean']:.1f} ms  |  "
          f"median {lat['median']:.1f} ms  |  "
          f"p95 {lat['p95']:.1f} ms")
    if lat['median'] > LATENCY_TARGET_MS:
        print(f"  ⚠  MEDIAN LATENCY EXCEEDS {LATENCY_TARGET_MS:.0f} ms TARGET")
    print("  (laptop-CPU proxy — not a real low-end-Android measurement)")
    print("=" * 70)


if __name__ == '__main__':
    main()

