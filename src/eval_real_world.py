"""
CropGuard AI — Phase 1: honest generalization number on self-taken phone photos.

Everything the model has ever seen is a single detached leaf on a uniform studio
background (PlantVillage). This script scores the *production* artifact —
models/cropguard_v1_production.tflite, the exact file the browser loads — on
photos taken and labelled by hand, and reports the drop against the clean-test
baseline of 0.9401. That gap is the domain-shift number.

Expected layout (folder name = ground-truth label):

    real_world_test/
        Tomato___Early_blight/IMG_0001.jpg
        Potato___healthy/IMG_0002.jpg
        Corn_(maize)___Common_rust_/IMG_0003.jpg
        _ood/...                 <- ignored here (leading underscore); Phase 4

Folder names may use either the PlantVillage directory names above or the
display-sanitized names from web/src/lib/constants.ts (e.g. `Tomato_Early_blight`).
See real_world_test/README.md for shooting and labelling rules.

Run:  python -m src.eval_real_world
      python -m src.eval_real_world --dir real_world_test --json outputs/real_world_report.json
      python -m src.eval_real_world --calibrated   # what the shipped UI now does
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (LEGACY_TIER_HIGH, LEGACY_TIER_MODERATE, TEMPERATURE,
                        TIER_HIGH, TIER_MODERATE)
from src.data_pipeline import CSV_PATH, IMG_SIZE

TFLITE_PATH = os.path.join('models', 'cropguard_v1_production.tflite')
DEFAULT_DIR = 'real_world_test'
DEFAULT_JSON = os.path.join('outputs', 'real_world_report.json')

# Clean held-out test split, same artifact — the thing we are measuring against.
CLEAN_TEST_ACCURACY = 0.9401
CLEAN_TEST_TOP3 = 0.9953

# Module-level so tier_of() stays a one-liner; --calibrated rebinds them to the
# thresholds derived by src/calibration.py, which is what the frontend ships.
HIGH_THRESHOLD = LEGACY_TIER_HIGH
MODERATE_THRESHOLD = LEGACY_TIER_MODERATE

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def _label_map() -> dict:
    """The training label order: sorted unique PlantVillage folder names."""
    df = pd.read_csv(CSV_PATH)
    return {lbl: i for i, lbl in enumerate(sorted(df['label'].unique()))}


def _sanitize(label: str) -> str:
    """PlantVillage folder name -> the display name used in constants.ts.

    Mirrors the hand-written CLASS_NAMES list, so a photo folder can be named
    either way without silently failing to match.
    """
    return (label.replace('_(maize)', '')
                 .replace('Tomato___Tomato_', 'Tomato___')
                 .replace('Spider_mites Two-spotted_spider_mite', 'Spider_mites')
                 .replace('Cercospora_leaf_spot Gray_leaf_spot', 'Cercospora_Gray_leaf_spot')
                 .replace('___', '_')
                 .rstrip('_'))


def _folder_lookup(label_map: dict) -> dict:
    """Accept both the PlantVillage name and the sanitized name as folder names."""
    lookup = {}
    for label, idx in label_map.items():
        lookup[label.lower()] = (label, idx)
        lookup[_sanitize(label).lower()] = (label, idx)
    return lookup


def discover(root: str, label_map: dict):
    """Walk root/<ClassName>/*.jpg. Returns (samples, unmatched_folders)."""
    lookup = _folder_lookup(label_map)
    samples, unmatched = [], []

    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        # Leading underscore = not a class folder (_ood/ is Phase 4's negative set).
        if not os.path.isdir(sub) or entry.startswith('_') or entry.startswith('.'):
            continue

        match = lookup.get(entry.lower())
        if match is None:
            unmatched.append(entry)
            continue

        label, idx = match
        for fname in sorted(os.listdir(sub)):
            if fname.lower().endswith(IMAGE_EXTS):
                samples.append({'path': os.path.join(sub, fname),
                                'label': label, 'y_true': idx})
    return samples, unmatched


def _read_image(path: str) -> np.ndarray:
    """Resize + normalize to [0,1] — identical to the training/eval preprocessing.

    decode_image (not decode_jpeg) so hand-taken PNG/HEIC-exported files work too.
    """
    img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                             expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return (tf.cast(img, tf.float32) / 255.0).numpy()


def run_inference(samples: list) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    probs = np.zeros((len(samples), out_d['shape'][-1]), dtype=np.float32)
    for i, s in enumerate(samples):
        interp.set_tensor(inp_d['index'], _read_image(s['path'])[None, ...])
        interp.invoke()
        probs[i] = interp.get_tensor(out_d['index'])[0]
    return probs


def _p(line: str = ''):
    """Print safely: Windows consoles default to cp1252 and photo filenames are
    routinely not ASCII (accents, Polish/German characters)."""
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'))


def tier_of(conf: float) -> str:
    if conf >= HIGH_THRESHOLD:
        return 'HIGH'
    if conf >= MODERATE_THRESHOLD:
        return 'MODERATE'
    return 'LOW'


def summarize(samples: list, probs: np.ndarray, class_names: list) -> dict:
    y_true = np.array([s['y_true'] for s in samples])
    top1 = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = top1 == y_true
    order = probs.argsort(axis=1)[:, ::-1]
    in_top3 = np.any(order[:, :3] == y_true[:, None], axis=1)
    tiers = np.array([tier_of(c) for c in conf])

    per_photo = []
    for i, s in enumerate(samples):
        per_photo.append({
            'path': s['path'].replace('\\', '/'),
            'true': s['label'],
            'predicted': class_names[top1[i]],
            'confidence': float(conf[i]),
            'tier': tiers[i],
            'correct': bool(correct[i]),
            'in_top3': bool(in_top3[i]),
            'top3': [{'class': class_names[j], 'p': float(probs[i, j])}
                     for j in order[i, :3]],
        })

    per_class = {}
    by_label = defaultdict(list)
    for i, s in enumerate(samples):
        by_label[s['label']].append(i)
    for label, idxs in sorted(by_label.items()):
        per_class[label] = {
            'n': len(idxs),
            'accuracy': float(correct[list(idxs)].mean()),
            'mean_confidence': float(conf[list(idxs)].mean()),
        }

    tier_stats = {}
    for name in ('HIGH', 'MODERATE', 'LOW'):
        m = tiers == name
        n = int(m.sum())
        tier_stats[name] = {
            'n': n,
            'coverage': float(m.mean()),
            'accuracy': float(correct[m].mean()) if n else None,
        }

    acc = float(correct.mean())
    top3_acc = float(in_top3.mean())
    return {
        'n_photos': int(len(samples)),
        'accuracy': acc,
        'top3_accuracy': top3_acc,
        'mean_confidence': float(conf.mean()),
        'clean_test_baseline': {
            'accuracy': CLEAN_TEST_ACCURACY,
            'top3_accuracy': CLEAN_TEST_TOP3,
        },
        'domain_shift_drop': {
            'accuracy': CLEAN_TEST_ACCURACY - acc,
            'top3_accuracy': CLEAN_TEST_TOP3 - top3_acc,
        },
        'tiers': tier_stats,
        'confident_and_wrong': [p for p in per_photo
                                if p['tier'] == 'HIGH' and not p['correct']],
        'per_class': per_class,
        'per_photo': per_photo,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR)
    ap.add_argument('--json', default=DEFAULT_JSON)
    ap.add_argument('--calibrated', action='store_true',
                    help='apply temperature scaling and the calibrated tier '
                         'thresholds from src/calibration.py — i.e. score these '
                         'photos the way the shipped UI now gates them')
    args = ap.parse_args()

    global HIGH_THRESHOLD, MODERATE_THRESHOLD
    if args.calibrated:
        HIGH_THRESHOLD, MODERATE_THRESHOLD = TIER_HIGH, TIER_MODERATE

    _p('=' * 74)
    _p('CropGuard AI — real-world (self-taken phone photo) evaluation')
    _p('=' * 74)

    if not os.path.isdir(args.dir):
        raise SystemExit(
            f"No '{args.dir}/' directory. Create it and add photos as\n"
            f"  {args.dir}/<ClassName>/photo.jpg\n"
            f"See {os.path.join(args.dir, 'README.md')} for the rules.")

    label_map = _label_map()
    class_names = list(label_map.keys())
    samples, unmatched = discover(args.dir, label_map)

    if unmatched:
        _p(f"WARNING: {len(unmatched)} folder(s) did not match any class and were "
              f"skipped: {', '.join(unmatched)}")

    if not samples:
        raise SystemExit(
            f"\nNo labelled photos found under '{args.dir}/'.\n"
            f"Phase 1 is blocked until you take and label them yourself — stock and\n"
            f"internet images are not trustworthy ground truth. Target: 15-25 photos,\n"
            f"single leaf filling the frame, natural light and background, phone camera.\n"
            f"See {os.path.join(args.dir, 'README.md')}.")

    _p(f"Model:  {TFLITE_PATH} ({os.path.getsize(TFLITE_PATH) / 1e6:.2f} MB)")
    _p(f"Photos: {len(samples)} across "
          f"{len({s['label'] for s in samples})} classes\n")

    probs = run_inference(samples)
    if args.calibrated:
        # p ** (1/T), renormalized — temperature scaling on the hidden logits.
        # Monotone, so accuracy is untouched; only confidences and tiers move.
        scaled = np.power(probs.astype(np.float64), 1.0 / TEMPERATURE)
        probs = (scaled / scaled.sum(axis=1, keepdims=True)).astype(np.float32)
    res = summarize(samples, probs, class_names)
    res['calibrated'] = bool(args.calibrated)
    res['temperature'] = TEMPERATURE if args.calibrated else None
    res['thresholds'] = {'HIGH': HIGH_THRESHOLD, 'MODERATE': MODERATE_THRESHOLD}

    _p(f"{'photo':<44}{'predicted':<34}{'conf':>7}{'tier':>10}  ok")
    for p in res['per_photo']:
        name = os.path.basename(p['path'])
        flag = 'YES' if p['correct'] else ('top3' if p['in_top3'] else 'NO')
        _p(f"{name[:43]:<44}{p['predicted'][:33]:<34}"
              f"{p['confidence']:>7.3f}{p['tier']:>10}  {flag}")

    d = res['domain_shift_drop']
    _p(f"\n{'':<26}{'real-world':>12}{'clean test':>12}{'drop':>10}")
    _p(f"{'top-1 accuracy':<26}{res['accuracy']:>12.4f}"
          f"{CLEAN_TEST_ACCURACY:>12.4f}{d['accuracy']:>10.4f}")
    _p(f"{'top-3 accuracy':<26}{res['top3_accuracy']:>12.4f}"
          f"{CLEAN_TEST_TOP3:>12.4f}{d['top3_accuracy']:>10.4f}")
    _p(f"{'mean confidence':<26}{res['mean_confidence']:>12.4f}")

    _p(f"\n{'Tier':<12}{'n':>5}{'coverage':>11}{'accuracy':>11}")
    for name in ('HIGH', 'MODERATE', 'LOW'):
        t = res['tiers'][name]
        acc = f"{t['accuracy']:.4f}" if t['accuracy'] is not None else '     -'
        _p(f"{name:<12}{t['n']:>5}{t['coverage']:>11.4f}{acc:>11}")

    cw = res['confident_and_wrong']
    _p(f"\nConfident and wrong (HIGH tier, incorrect): {len(cw)}")
    for p in cw:
        _p(f"  {p['confidence']:.3f}  {p['true']} -> {p['predicted']}  "
              f"({os.path.basename(p['path'])})")

    _p("\nPer class:")
    for label, c in res['per_class'].items():
        _p(f"  {label:<48} n={c['n']:<3} acc {c['accuracy']:.3f}  "
              f"mean conf {c['mean_confidence']:.3f}")

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump(res, f, indent=2)
    _p(f"\nSaved {os.path.abspath(args.json)}")
    _p('=' * 74)


if __name__ == '__main__':
    main()
