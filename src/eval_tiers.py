"""
CropGuard AI — empirical check of the confidence-tier system.

The frontend (web/src/lib/confidenceTier.ts) gates its UI on top-1 softmax
confidence: HIGH >= 0.85 (full diagnosis + treatment), MODERATE >= 0.50
(caution banner + top-3), LOW otherwise (no diagnosis, no treatment panel).
Those thresholds were chosen by intuition. This script measures what they
actually buy, by running models/cropguard_v1_production.tflite (the exact
artifact the browser loads) over a split and reporting, per tier:

  - coverage (share of images that land in the tier)
  - top-1 accuracy and top-3 accuracy inside the tier
  - the "dangerous" cases: HIGH-tier images that are wrong

Also prints a coverage/accuracy sweep over candidate HIGH thresholds — a
preview of the risk-coverage analysis Phase 2 does properly.

Run:  python -m src.eval_tiers            # test split (default)
      python -m src.eval_tiers --split val
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_pipeline import CSV_PATH, IMG_SIZE

TFLITE_PATH = os.path.join('models', 'cropguard_v1_production.tflite')
REPORT_PATH = os.path.join('outputs', 'tier_report.json')

# Must mirror TIER_THRESHOLDS in web/src/lib/confidenceTier.ts.
HIGH_THRESHOLD = 0.85
MODERATE_THRESHOLD = 0.50

SWEEP = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


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


def run_inference(split_df: pd.DataFrame, label_map: dict):
    """Return (probs [N,17] float32, y_true [N] int) from the production model."""
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    # Production artifact is float32 in/out with a static batch of 1 — no
    # scale/zero-point math needed (see docs/quantization_findings.md).
    assert inp_d['dtype'] == np.float32, inp_d['dtype']
    assert out_d['dtype'] == np.float32, out_d['dtype']

    probs = np.zeros((len(split_df), out_d['shape'][-1]), dtype=np.float32)
    for i, row in split_df.iterrows():
        interp.set_tensor(inp_d['index'], _read_image(row['image_path'])[None, ...])
        interp.invoke()
        probs[i] = interp.get_tensor(out_d['index'])[0]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(split_df)} images")

    y_true = np.array([label_map[lbl] for lbl in split_df['label']])
    return probs, y_true


def tier_of(conf: float) -> str:
    if conf >= HIGH_THRESHOLD:
        return 'HIGH'
    if conf >= MODERATE_THRESHOLD:
        return 'MODERATE'
    return 'LOW'


def summarize(probs: np.ndarray, y_true: np.ndarray, class_names: list):
    top1 = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = top1 == y_true
    top3 = probs.argsort(axis=1)[:, -3:]
    in_top3 = np.any(top3 == y_true[:, None], axis=1)
    tiers = np.array([tier_of(c) for c in conf])

    result = {
        'n_images': int(len(y_true)),
        'thresholds': {'HIGH': HIGH_THRESHOLD, 'MODERATE': MODERATE_THRESHOLD},
        'overall_accuracy': float(correct.mean()),
        'overall_top3_accuracy': float(in_top3.mean()),
        'mean_confidence': float(conf.mean()),
        'tiers': {},
        'high_tier_errors': [],
        'threshold_sweep': [],
    }

    for name in ('HIGH', 'MODERATE', 'LOW'):
        m = tiers == name
        n = int(m.sum())
        result['tiers'][name] = {
            'n': n,
            'coverage': float(m.mean()),
            'accuracy': float(correct[m].mean()) if n else None,
            'top3_accuracy': float(in_top3[m].mean()) if n else None,
            'mean_confidence': float(conf[m].mean()) if n else None,
        }

    # The failure mode the tier system is supposed to prevent: confident + wrong.
    for idx in np.where((tiers == 'HIGH') & ~correct)[0]:
        result['high_tier_errors'].append({
            'true': class_names[y_true[idx]],
            'predicted': class_names[top1[idx]],
            'confidence': float(conf[idx]),
        })
    result['high_tier_errors'].sort(key=lambda e: -e['confidence'])

    for t in SWEEP:
        m = conf >= t
        result['threshold_sweep'].append({
            'threshold': t,
            'coverage': float(m.mean()),
            'accuracy': float(correct[m].mean()) if m.any() else None,
        })

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    args = ap.parse_args()

    print('=' * 70)
    print(f'CropGuard AI — confidence-tier check ({args.split} split)')
    print('=' * 70)

    split_df, label_map = _load_split(args.split)
    class_names = list(label_map.keys())
    print(f'Model: {TFLITE_PATH}  ({os.path.getsize(TFLITE_PATH) / 1e6:.2f} MB)')
    print(f'Images: {len(split_df)}')

    probs, y_true = run_inference(split_df, label_map)
    res = summarize(probs, y_true, class_names)
    res['split'] = args.split

    print(f"\nOverall: acc {res['overall_accuracy']:.4f}  "
          f"top-3 {res['overall_top3_accuracy']:.4f}  "
          f"mean confidence {res['mean_confidence']:.4f}")

    print(f"\n{'Tier':<10}{'n':>7}{'coverage':>11}{'top-1 acc':>12}"
          f"{'top-3 acc':>12}{'mean conf':>12}")
    for name in ('HIGH', 'MODERATE', 'LOW'):
        t = res['tiers'][name]
        fmt = lambda v: f'{v:.4f}' if v is not None else '     -'
        print(f"{name:<10}{t['n']:>7}{t['coverage']:>11.4f}"
              f"{fmt(t['accuracy']):>12}{fmt(t['top3_accuracy']):>12}"
              f"{fmt(t['mean_confidence']):>12}")

    errs = res['high_tier_errors']
    print(f"\nHIGH-tier errors (confident and wrong): {len(errs)} "
          f"({len(errs) / res['n_images'] * 100:.2f}% of all images)")
    for e in errs[:10]:
        print(f"  {e['confidence']:.4f}  {e['true']} -> {e['predicted']}")
    if len(errs) > 10:
        print(f"  ... {len(errs) - 10} more")

    print(f"\n{'HIGH threshold':>15}{'coverage':>11}{'accuracy':>11}")
    for s in res['threshold_sweep']:
        acc = f"{s['accuracy']:.4f}" if s['accuracy'] is not None else '    -'
        print(f"{s['threshold']:>15.2f}{s['coverage']:>11.4f}{acc:>11}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {os.path.abspath(REPORT_PATH)}")
    print('=' * 70)


if __name__ == '__main__':
    main()
