"""
CropGuard AI — synthetic domain-shift stress test.

Phase 1's real number needs hand-taken field photos (see real_world_test/README.md)
and cannot be faked. This script measures the *same kind* of degradation without
them: take clean held-out test images and corrupt them the way a phone photo in a
field differs from a PlantVillage studio shot — new background, motion blur, harsh
or dim light, warm/cool white balance, JPEG artifacts — then score the production
artifact on each corruption separately.

What it is good for: an ordered list of which shift the model is most fragile to,
and a defensible lower bound on the field drop.
What it is NOT: a replacement for real photos. These corruptions are drawn from the
same augmentation family the model was *trained* on, so the model has effectively
seen them and the measured drop is optimistic. Read every number here as "at least
this much."

Run:  python -m src.eval_domain_shift              # 600 test images, seed 42
      python -m src.eval_domain_shift --n 3625     # whole test split
"""

import argparse
import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from src.augmentation import BackgroundReplace, BACKGROUNDS_DIR
from src.data_pipeline import CSV_PATH, IMG_SIZE

TFLITE_PATH = os.path.join('models', 'cropguard_v1_production.tflite')
REPORT_PATH = os.path.join('outputs', 'domain_shift_report.json')

HIGH_THRESHOLD = 0.85

# Each corruption is deliberately single-factor except the last, so the ranking
# says which shift hurts, not just that shift hurts. p=1.0 throughout: this is a
# measurement, not training augmentation.
CORRUPTIONS = {
    'clean': [],
    'background_replace': [
        BackgroundReplace(backgrounds_dir=BACKGROUNDS_DIR, p=1.0)],
    'motion_blur': [A.MotionBlur(blur_limit=(7, 11), p=1.0)],
    'defocus_blur': [A.GaussianBlur(blur_limit=(5, 9), p=1.0)],
    'underexposed': [A.RandomBrightnessContrast(
        brightness_limit=(-0.45, -0.30), contrast_limit=(-0.2, 0.0), p=1.0)],
    'overexposed': [A.RandomBrightnessContrast(
        brightness_limit=(0.30, 0.45), contrast_limit=(-0.2, 0.0), p=1.0)],
    'white_balance': [A.HueSaturationValue(
        hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=0, p=1.0)],
    'jpeg_artifacts': [A.ImageCompression(quality_range=(15, 30), p=1.0)],
    'off_angle': [A.Affine(rotate=(-35, 35), scale=(0.75, 0.95), shear=(-12, 12),
                           border_mode=cv2.BORDER_REFLECT_101, p=1.0)],
    # std_range is in normalized units: 0.02-0.05 is ~5-13 grey levels, which is
    # realistic phone-sensor noise in low light. Albumentations 2.x defaults to
    # (0.2, 0.44) — 51-112 levels — which is not a photo, it is static.
    'sensor_noise': [A.GaussNoise(std_range=(0.02, 0.05), p=1.0)],
    'sensor_noise_severe': [A.GaussNoise(std_range=(0.10, 0.15), p=1.0)],
    # The realistic composite: everything a handheld outdoor shot stacks at once.
    'field_composite': [
        BackgroundReplace(backgrounds_dir=BACKGROUNDS_DIR, p=1.0),
        A.Affine(rotate=(-25, 25), scale=(0.8, 1.0),
                 border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.25, p=1.0),
        A.MotionBlur(blur_limit=(5, 9), p=0.7),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.5),
        A.ImageCompression(quality_range=(25, 55), p=1.0),
    ],
}


def _load_test_sample(n: int, seed: int):
    """Stratified subsample of the test split, so rare classes stay represented."""
    df = pd.read_csv(CSV_PATH)
    label_map = {lbl: i for i, lbl in enumerate(sorted(df['label'].unique()))}
    test_df = df[df['split'] == 'test']

    if n >= len(test_df):
        sample = test_df
    else:
        frac = n / len(test_df)
        parts = [g.sample(max(1, round(len(g) * frac)), random_state=seed)
                 for _, g in test_df.groupby('label', sort=True)]
        sample = pd.concat(parts)
    return sample.reset_index(drop=True), label_map


def _read_uint8(path: str) -> np.ndarray:
    """Decode + resize to 224x224 uint8 RGB. Corruptions run in uint8 space, then
    the same /255.0 as training is applied afterwards — so the only difference
    between 'clean' here and src.eval_tiers is the resize dtype rounding."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                             expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return np.clip(img.numpy(), 0, 255).astype(np.uint8)


def _make_interpreter():
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]


def evaluate_corruption(name, transforms, images, y_true, seed):
    """Score one corruption over the pre-decoded uint8 image list."""
    pipeline = A.Compose(transforms) if transforms else None
    # Reseed per corruption so every corruption sees the same image order and
    # comparable random draws.
    np.random.seed(seed)
    import random
    random.seed(seed)

    interp, inp_d, out_d = _make_interpreter()
    probs = np.zeros((len(images), out_d['shape'][-1]), dtype=np.float32)

    for i, img in enumerate(images):
        x = pipeline(image=img)['image'] if pipeline else img
        x = (x.astype(np.float32) / 255.0)[None, ...]
        interp.set_tensor(inp_d['index'], x)
        interp.invoke()
        probs[i] = interp.get_tensor(out_d['index'])[0]

    top1 = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = top1 == y_true
    in_top3 = np.any(probs.argsort(axis=1)[:, -3:] == y_true[:, None], axis=1)
    high = conf >= HIGH_THRESHOLD

    return {
        'accuracy': float(correct.mean()),
        'top3_accuracy': float(in_top3.mean()),
        'mean_confidence': float(conf.mean()),
        'high_tier_coverage': float(high.mean()),
        'high_tier_accuracy': float(correct[high].mean()) if high.any() else None,
        'confidently_wrong_rate': float((high & ~correct).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600,
                    help='test images to sample (stratified); default 600')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('=' * 78)
    print('CropGuard AI — synthetic domain-shift stress test')
    print('=' * 78)
    print('NOTE: a lower bound. These corruptions overlap the training augmentation,')
    print('      so the model has effectively seen them. Real field photos will be')
    print('      worse. See real_world_test/README.md.\n')

    sample_df, label_map = _load_test_sample(args.n, args.seed)
    y_true = np.array([label_map[lbl] for lbl in sample_df['label']])
    print(f"Model:  {TFLITE_PATH} ({os.path.getsize(TFLITE_PATH) / 1e6:.2f} MB)")
    print(f"Images: {len(sample_df)} stratified from the test split (seed {args.seed})")

    print('Decoding...', flush=True)
    images = [_read_uint8(p) for p in sample_df['image_path']]

    results = {}
    for name, transforms in CORRUPTIONS.items():
        print(f'  scoring {name} ...', flush=True)
        results[name] = evaluate_corruption(name, transforms, images, y_true,
                                            args.seed)

    base = results['clean']['accuracy']
    for name, r in results.items():
        r['accuracy_drop_vs_clean'] = base - r['accuracy']

    print(f"\n{'corruption':<22}{'top-1':>8}{'drop':>8}{'top-3':>8}"
          f"{'meanconf':>10}{'HIGH cov':>10}{'HIGH acc':>10}{'conf+wrong':>12}")
    ranked = sorted(results.items(), key=lambda kv: -kv[1]['accuracy_drop_vs_clean'])
    for name, r in ranked:
        ha = f"{r['high_tier_accuracy']:.3f}" if r['high_tier_accuracy'] is not None else '    -'
        print(f"{name:<22}{r['accuracy']:>8.4f}{r['accuracy_drop_vs_clean']:>8.4f}"
              f"{r['top3_accuracy']:>8.4f}{r['mean_confidence']:>10.4f}"
              f"{r['high_tier_coverage']:>10.4f}{ha:>10}"
              f"{r['confidently_wrong_rate']:>12.4f}")

    worst = ranked[0]
    print(f"\nMost damaging single shift: {worst[0]} "
          f"(-{worst[1]['accuracy_drop_vs_clean']:.4f} accuracy)")
    fc = results['field_composite']
    print(f"Stacked field-like composite: {fc['accuracy']:.4f} "
          f"(-{fc['accuracy_drop_vs_clean']:.4f} vs clean {base:.4f}); "
          f"confidently-wrong rate {fc['confidently_wrong_rate']:.4f} "
          f"vs {results['clean']['confidently_wrong_rate']:.4f} clean")

    out = {
        'n_images': int(len(sample_df)),
        'seed': args.seed,
        'clean_accuracy': base,
        'note': ('Lower bound. Corruptions overlap the training augmentation family; '
                 'real field photos are expected to be worse. Not a substitute for '
                 'real_world_test/.'),
        'corruptions': results,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {os.path.abspath(REPORT_PATH)}")
    print('=' * 78)


if __name__ == '__main__':
    main()
