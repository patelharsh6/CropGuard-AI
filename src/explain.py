"""
CropGuard AI — Phase 3: Grad-CAM explainability and the background-leakage check.

Every accuracy number in this repo says *how often* the model is right. None of
them say *what it looked at*. That matters here for one specific reason: the whole
training set is a detached leaf on a uniform studio background, so a network can
score 0.94 by reading the background instead of the lesion — the classic shortcut.
`BackgroundReplace` in src/augmentation.py exists to prevent exactly that, and was
never verified. This script verifies it.

What it does:

  1. Grad-CAM on the Keras model (models/cropguard_v1.keras) for four cohorts:
       correct_high  — correct, high-confidence clean test images (what "working"
                       attention looks like)
       cluster       — the tomato brown-lesion confusions (Early blight / Target
                       Spot / Septoria / Spider mites), the model's main error
                       mode. Rendered for BOTH the predicted and the true class.
       real_world    — the 17 field-style photos from Phase 1 (one per class)
       web_sourced   — the 20 vetted Wikimedia Commons photos
       ood           — the whole-plant / non-leaf images in */_ood, the known
                       out-of-distribution failure
     -> outputs/gradcam/<cohort>/*.jpg

  2. Quantifies background leakage: fraction of CAM mass falling inside a leaf
     mask, where the mask is the GrabCut segmentation already written for
     BackgroundReplace (src.augmentation._segment_leaf). Reported against the
     leaf's own area fraction, because a CAM that ignores the image entirely
     already scores mass == area. The ratio (`lift`) is the number that means
     something: > 1 = attention concentrates on the leaf, ~1 = indifferent,
     < 1 = the model is looking at the background.

  3. Repeats that measurement over a random sample of the test split (`--sweep`),
     split by correct vs wrong prediction — the cohorts are hand-picked and tiny,
     which is fine for looking at pictures and useless as evidence about the model.

  4. Everything -> outputs/gradcam_report.json; findings -> docs/EXPLAINABILITY.md.

Two implementation notes worth knowing before editing:

  * The graph is Rescaling -> MobileNetV3Small -> GAP -> Dropout -> Dense(softmax),
    so true logits are not an output. Rather than doing graph surgery on a nested
    Functional model, the head is re-applied by hand inside the tape
    (GAP -> kernel/bias matmul), which gives exact pre-softmax logits. The
    reconstruction is asserted against model.predict() at startup.
  * With GAP + a single Dense head, Grad-CAM on the FINAL conv activation is
    mathematically identical to plain CAM: d(logit_c)/d(A_k) is the constant
    W[k,c]/49, so the Grad-CAM channel weights *are* the classifier weights. The
    deep map therefore carries no more information than CAM, and it is only 7x7.
    That is why a second, genuinely non-trivial map is computed at a 14x14 layer
    (`activation_11`), where the gradient actually varies spatially.

Analysis runs on the Keras float model, not the shipped dynamic-range .tflite —
TFLite exposes no gradients. Agreement between the two is checked per image and
reported as `keras_tflite_agree`.

Run:  python -m src.explain
      python -m src.explain --cohort cluster --n 12
      python -m src.explain --no-mask          # skip segmentation (much faster)
      python -m src.explain --sweep 600        # bigger leakage sample
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import keras

from src.augmentation import _segment_leaf
from src.config import SEED, TEMPERATURE, TIER_HIGH
from src.data_pipeline import CSV_PATH, IMG_SIZE
from src.eval_real_world import _folder_lookup, _sanitize
from src.eval_tiers import TFLITE_PATH, _load_split

KERAS_PATH = os.path.join('models', 'cropguard_v1.keras')
OUT_DIR = os.path.join('outputs', 'gradcam')
REPORT_PATH = os.path.join('outputs', 'gradcam_report.json')
CACHE_DIR = os.path.join('outputs', 'cache')

# Nested backbone; its final activation is the model's 7x7x576 feature map.
BACKBONE = 'MobileNetV3Small'
# 'deep' is the last conv activation (== CAM, see module docstring); 'mid' is the
# 14x14 expand activation of block 8, the finest map where Grad-CAM is non-trivial.
LAYERS = {'deep': 'activation_17', 'mid': 'activation_11'}

# The confusion cluster src/evaluate.py flags as the model's main error mode.
BROWN_LESION_CLUSTER = [
    'Tomato___Early_blight',
    'Tomato___Target_Spot',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
]

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# A mask pixel counts as leaf above this alpha (_segment_leaf feathers its edges).
MASK_THRESHOLD = 0.5
# Panels are photographic overlays, so PNG is a poor fit: the same 57 panels cost 34 MB
# as PNG and ~4 MB as JPEG, and outputs/ is committed. Quality 88 is well above any
# artifact that would change how a heatmap reads.
PANEL_EXT = '.jpg'
PANEL_QUALITY = 88
# _segment_leaf assumes one leaf on a plain background. On a whole-plant or non-leaf
# photo it returns a sliver or nearly the whole frame, and "CAM mass inside the leaf"
# then measures nothing. Masks outside this area band are recorded but excluded from
# the cohort aggregates.
MASK_AREA_BAND = (0.05, 0.95)
PROB_FLOOR = 1e-30


# --------------------------------------------------------------- image loading

def load_image(path):
    """RGB float32 [0,1] at IMG_SIZE — matches data_pipeline / eval_tiers exactly.

    The TF decoder is not interchangeable with OpenCV's here: their JPEG output
    differs by up to ~4 grey levels, which is enough to flip a borderline
    prediction (measured: a cluster image at 0.49 Target_Spot under tf.io becomes
    0.66 Spider_mites under cv2.imdecode). Cohort selection reads predictions
    cached by src.calibration, which used tf.io — so this must too, or a picked
    "error" can render as correct. cv2 is only the fallback for formats
    tf.io.decode_image cannot read (e.g. webp).
    """
    try:
        raw = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                 expand_animations=False)
    except Exception:
        buf = np.fromfile(path, dtype=np.uint8)
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if dec is None:
            raise ValueError(f'unreadable image: {path}')
        raw = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(tf.cast(raw, tf.float32), IMG_SIZE)
    return (img / 255.0).numpy()


# ------------------------------------------------------------------- Grad-CAM

class Explainer:
    """Grad-CAM over the Keras model, with the head re-applied inside the tape."""

    def __init__(self, keras_path=KERAS_PATH):
        self.model = keras.models.load_model(keras_path)
        self.base = self.model.get_layer(BACKBONE)
        dense = self.model.get_layer('predictions')
        self.W = tf.convert_to_tensor(dense.kernel.numpy())
        self.b = tf.convert_to_tensor(dense.bias.numpy())
        # One tap per CAM layer: (intermediate activation, final feature map).
        self.taps = {
            key: keras.Model(self.base.inputs,
                             [self.base.get_layer(name).output, self.base.output])
            for key, name in LAYERS.items()
        }

    def forward(self, x, layer_key='deep'):
        """(conv activation, softmax probs) from the reconstructed head."""
        conv, feat = self.taps[layer_key](x * 255.0, training=False)
        logits = tf.matmul(tf.reduce_mean(feat, axis=[1, 2]), self.W) + self.b
        return conv, tf.nn.softmax(logits).numpy()

    def verify_head(self, img):
        """Max |softmax(reconstructed logits) - model.predict|. Must be ~0."""
        x = img[None, ...]
        ref = self.model.predict(x, verbose=0)[0]
        _, probs = self.forward(tf.convert_to_tensor(x), 'deep')
        return float(np.max(np.abs(probs[0] - ref)))

    def cam(self, img, class_idx, layer_key='deep'):
        """(cam [224,224] in [0,1], probs [17]) for one image and target class."""
        x = tf.convert_to_tensor(img[None, ...])
        with tf.GradientTape() as tape:
            conv, feat = self.taps[layer_key](x * 255.0, training=False)
            tape.watch(conv)
            logits = tf.matmul(tf.reduce_mean(feat, axis=[1, 2]), self.W) + self.b
            score = logits[:, class_idx]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=[1, 2])                  # [1, K]
        cam = tf.nn.relu(tf.einsum('bhwk,bk->bhw', conv, weights))[0].numpy()

        probs = tf.nn.softmax(logits).numpy()[0]
        peak = float(cam.max())
        if peak > 0:
            cam = cam / peak
        cam = cv2.resize(cam, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        return np.clip(cam, 0.0, 1.0), probs


# --------------------------------------------------------- leakage measurement

def leaf_mask(img):
    """Binary leaf mask + its area fraction, from the BackgroundReplace segmenter."""
    alpha = _segment_leaf((img * 255.0).astype(np.uint8))
    mask = alpha >= MASK_THRESHOLD
    return mask, float(mask.mean())


def cam_mass_in_mask(cam, mask):
    """Share of total CAM mass inside the mask, and whether the peak lands there."""
    total = float(cam.sum())
    if total <= 0:
        return None, None
    inside = float(cam[mask].sum()) / total
    peak_yx = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return inside, bool(mask[peak_yx])


# ------------------------------------------------------------------- rendering

def overlay(img, cam, alpha=0.45):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(img * (1 - alpha) + heat * alpha, 0, 1)


def with_contour(img, mask):
    if mask is None:
        return img
    out = (img * 255).astype(np.uint8).copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out.astype(np.float32) / 255.0


def render_panel(rec, img, mask, cams, out_path):
    """One figure per image: source (+ mask outline) then a column per CAM."""
    cols = 1 + len(cams)
    fig, axes = plt.subplots(1, cols, figsize=(3.1 * cols, 3.8))
    axes = np.atleast_1d(axes)

    axes[0].imshow(with_contour(img, mask))
    area = rec.get('leaf_area_frac')
    axes[0].set_title('source' + (f'\nleaf area {area:.2f}' if area else ''),
                      fontsize=9)

    for ax, (title, cam) in zip(axes[1:], cams):
        ax.imshow(overlay(img, cam))
        ax.set_title(title, fontsize=9)
    for ax in axes:
        ax.axis('off')

    head = f"pred {_sanitize(rec['pred_label'])} ({rec['confidence']:.3f})"
    if rec['true_label']:
        head += (f"  |  true {_sanitize(rec['true_label'])}"
                 f"  [{'OK' if rec['correct'] else 'WRONG'}]")
    fig.suptitle(head, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=110,
                pil_kwargs={'quality': PANEL_QUALITY, 'optimize': True})
    plt.close(fig)


# --------------------------------------------------------------------- cohorts

def _cached_tflite_probs(split):
    """Cached production-model probabilities for a split (written by calibration)."""
    p = os.path.join(CACHE_DIR, f'probs_{split}.npy')
    y = os.path.join(CACHE_DIR, f'labels_{split}.npy')
    if not (os.path.exists(p) and os.path.exists(y)):
        raise SystemExit(
            f'missing {p} — run `python -m src.calibration` first to populate the '
            'prediction cache (cohort selection reads the shipped model, not Keras).')
    return np.load(p), np.load(y)


def calibrate(probs):
    """Temperature scaling, identical to src/calibration and the shipped UI."""
    logits = np.log(np.clip(probs.astype(np.float64), PROB_FLOOR, 1.0)) / TEMPERATURE
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _pick_spread(idx, labels, n, seed=SEED):
    """Up to n indices, spreading across distinct labels before repeating one."""
    rng = np.random.default_rng(seed)
    by_label = {}
    for i in idx:
        by_label.setdefault(labels[i], []).append(int(i))
    for v in by_label.values():
        rng.shuffle(v)
    picked, keys = [], sorted(by_label)
    while len(picked) < n and any(by_label[k] for k in keys):
        for k in keys:
            if by_label[k] and len(picked) < n:
                picked.append(by_label[k].pop())
    return picked


def cohort_correct_high(n):
    """Correct, calibrated-HIGH test images — the 'attention works' reference."""
    split_df, _ = _load_split('test')
    probs, y_true = _cached_tflite_probs('test')
    cal = calibrate(probs)
    pred, conf = cal.argmax(1), cal.max(1)
    hits = np.where((pred == y_true) & (conf >= TIER_HIGH))[0]
    labels = list(split_df['label'])
    return [(split_df.loc[i, 'image_path'], labels[i])
            for i in _pick_spread(hits, labels, n or len(hits))]


def cohort_cluster(n):
    """Test images confused *within* the tomato brown-lesion cluster."""
    split_df, label_map = _load_split('test')
    probs, y_true = _cached_tflite_probs('test')
    pred = calibrate(probs).argmax(1)
    cluster = {label_map[c] for c in BROWN_LESION_CLUSTER if c in label_map}
    hits = np.where([(t in cluster) and (p in cluster) and (t != p)
                     for t, p in zip(y_true, pred)])[0]
    labels = list(split_df['label'])
    return [(split_df.loc[i, 'image_path'], labels[i])
            for i in _pick_spread(hits, labels, n or len(hits))]


def _walk_photos(root, ood=False):
    """(path, label_or_None) for a photo dir; ood=True reads only _* folders."""
    if not os.path.isdir(root):
        return []
    df = pd.read_csv(CSV_PATH)
    lookup = _folder_lookup({lbl: i for i, lbl in enumerate(sorted(df['label'].unique()))})
    out = []
    for folder in sorted(os.listdir(root)):
        sub = os.path.join(root, folder)
        if not os.path.isdir(sub) or folder.startswith('_') != ood:
            continue
        hit = lookup.get(folder.lower())
        for f in sorted(os.listdir(sub)):
            if f.lower().endswith(IMAGE_EXTS):
                out.append((os.path.join(sub, f), hit[0] if hit else None))
    return out


def _photo_cohort(*roots, ood=False):
    def selector(n):
        photos = [p for r in roots for p in _walk_photos(r, ood=ood)]
        return photos[:n] if n else photos
    return selector


# (selector, default image count; 0 = all available). The two photo sets stay
# separate cohorts because they are separate results in docs/DOMAIN_SHIFT.md:
# real_world_test is one image per class, web_sourced_test is Wikimedia Commons.
COHORTS = {
    'correct_high': (cohort_correct_high, 8),
    'cluster': (cohort_cluster, 8),
    'real_world': (_photo_cohort('real_world_test'), 0),
    'web_sourced': (_photo_cohort('web_sourced_test'), 0),
    # real_world_test/_ood is empty today; it is included so photos dropped there
    # later are picked up without an edit.
    'ood': (_photo_cohort('real_world_test', 'web_sourced_test', ood=True), 0),
}


# ------------------------------------------------------------------------ main

def make_tflite_predictor():
    """Predictor for the shipped artifact — used only for the agreement check."""
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    def predict(img):
        interp.set_tensor(inp_d['index'], img[None, ...])
        interp.invoke()
        return interp.get_tensor(out_d['index'])[0]
    return predict


def panel_stem(i, path, true_label):
    """Panel filename: index + source dir + label, so nothing collides.

    Photo sets use one file per class folder all named image.png, and the
    real_world cohort merges two directories.
    """
    src = path.replace(os.sep, '/').split('/')[0]
    tail = (_sanitize(true_label) if true_label
            else os.path.splitext(os.path.basename(path))[0])
    # 'data' is the training dataset root and says nothing; photo-set roots do.
    prefix = '' if src.startswith('data') else f'{src}_'
    stem = f'{i:02d}_{prefix}{tail}'
    return stem.encode('ascii', 'replace').decode().replace('?', '_')[:70]


def analyse_image(ex, path, true_label, label_map, class_names, want_mask, tfl):
    img = load_image(path)
    mask, area = leaf_mask(img) if want_mask else (None, None)

    _, probs = ex.forward(tf.convert_to_tensor(img[None, ...]), 'deep')
    probs = probs[0]
    pred_idx = int(np.argmax(probs))
    true_idx = label_map.get(true_label) if true_label else None

    rec = {
        'path': path.replace('\\', '/'),
        'true_label': true_label,
        'pred_label': class_names[pred_idx],
        'confidence': float(probs[pred_idx]),
        'calibrated_confidence': float(calibrate(probs[None, :])[0, pred_idx]),
        'correct': None if true_idx is None else bool(pred_idx == true_idx),
        'leaf_area_frac': area,
        'mask_plausible': None if area is None else
            bool(MASK_AREA_BAND[0] <= area <= MASK_AREA_BAND[1]),
        'keras_tflite_agree': bool(int(np.argmax(tfl(img))) == pred_idx),
        'layers': {},
    }

    cams = []
    for key in LAYERS:
        cam, _ = ex.cam(img, pred_idx, key)
        cams.append((f'{key} CAM — pred', cam))
        entry = {}
        if mask is not None:
            inside, peak_in = cam_mass_in_mask(cam, mask)
            entry = {
                'cam_mass_in_leaf': inside,
                'lift_vs_area': (inside / area) if (inside and area) else None,
                'peak_in_leaf': peak_in,
            }
        rec['layers'][key] = entry

    # For an error, also show where the evidence for the RIGHT answer sits.
    if true_idx is not None and true_idx != pred_idx:
        cam_true, _ = ex.cam(img, true_idx, 'deep')
        cams.append(('deep CAM — true class', cam_true))
        rec['true_class_confidence'] = float(probs[true_idx])

    return rec, img, mask, cams


def aggregate(records):
    """Per-cohort means of the leakage metrics, per CAM layer."""
    def mean(values):
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    # Only images whose leaf mask is plausible contribute to the leakage means.
    masked = [r for r in records if r.get('mask_plausible')]

    out = {}
    for key in LAYERS:
        per = [r['layers'][key] for r in masked]
        out[key] = {
            'n': len([p for p in per if p.get('cam_mass_in_leaf') is not None]),
            'mean_cam_mass_in_leaf': mean([p.get('cam_mass_in_leaf') for p in per]),
            'mean_lift_vs_area': mean([p.get('lift_vs_area') for p in per]),
            'peak_in_leaf_rate': mean([p.get('peak_in_leaf') for p in per]),
        }
    out['n_images'] = len(records)
    out['n_masks_plausible'] = len(masked)
    out['mean_leaf_area_frac'] = mean([r['leaf_area_frac'] for r in masked])
    out['accuracy'] = mean([r['correct'] for r in records])
    out['mean_confidence'] = mean([r['confidence'] for r in records])
    out['keras_tflite_agreement'] = mean([r['keras_tflite_agree'] for r in records])
    return out



# ------------------------------------------------------------- leakage sweep

def leakage_sweep(ex, tfl, label_map, class_names, n, seed=SEED):
    """Leakage metrics over a random test sample, no panels rendered.

    The cohorts above are hand-picked and tiny, which is fine for looking at
    pictures and useless for a claim about the model as a whole. This runs the
    same measurement over a plain random sample of the held-out test split and
    splits it by whether the prediction was right — the comparison that says
    whether wrong answers come with off-leaf attention (i.e. whether CAM mass
    could serve as an error signal).
    """
    split_df, _ = _load_split('test')
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(split_df), size=min(n, len(split_df)), replace=False)

    records = []
    for count, i in enumerate(idx, 1):
        rec, _, _, _ = analyse_image(ex, split_df.loc[int(i), 'image_path'],
                                     split_df.loc[int(i), 'label'],
                                     label_map, class_names, True, tfl)
        records.append(rec)
        if count % 25 == 0:
            print(f'  sweep {count}/{len(idx)}')

    correct = [r for r in records if r['correct']]
    wrong = [r for r in records if r['correct'] is False]
    return {
        'n_sampled': len(records),
        'seed': seed,
        'all': aggregate(records),
        'correct': aggregate(correct),
        'wrong': aggregate(wrong),
    }


def main():
    ap = argparse.ArgumentParser(description='Grad-CAM + background-leakage analysis')
    ap.add_argument('--cohort', choices=list(COHORTS), action='append',
                    help='cohort to run (repeatable; default: all)')
    ap.add_argument('--n', type=int, default=None,
                    help='images per cohort (default: cohort-specific; 0 = all)')
    ap.add_argument('--no-mask', action='store_true',
                    help='skip GrabCut segmentation and the leakage metrics')
    ap.add_argument('--sweep', type=int, default=200,
                    help='random test images for the leakage statistics '
                         '(no panels; 0 disables)')
    ap.add_argument('--json', default=REPORT_PATH)
    args = ap.parse_args()

    cohorts = args.cohort or list(COHORTS)
    want_mask = not args.no_mask

    _, label_map = _load_split('test')
    class_names = list(label_map.keys())

    print(f'loading {KERAS_PATH}')
    ex = Explainer()
    tfl = make_tflite_predictor()

    report = {'model': KERAS_PATH, 'layers': LAYERS,
              'mask_threshold': MASK_THRESHOLD, 'cohorts': {}}

    # The reconstructed head must reproduce the model's own softmax exactly.
    probe = load_image(COHORTS['real_world'][0](1)[0][0])
    delta = ex.verify_head(probe)
    print(f'head reconstruction max|delta| = {delta:.2e}')
    assert delta < 1e-5, f'logit reconstruction is wrong (delta={delta})'
    report['head_reconstruction_max_abs_error'] = delta

    for name in cohorts:
        fn, default_n = COHORTS[name]
        n = default_n if args.n is None else args.n
        photos = fn(n)
        cohort_dir = os.path.join(OUT_DIR, name)
        os.makedirs(cohort_dir, exist_ok=True)
        # Panel names encode the image index, so a shorter run would otherwise
        # leave stale panels from a longer one behind.
        for old in os.listdir(cohort_dir):
            if old.endswith(('.png', '.jpg')):
                os.remove(os.path.join(cohort_dir, old))
        print(f'\n[{name}] {len(photos)} images -> {cohort_dir}')

        records = []
        for i, (path, true_label) in enumerate(photos):
            rec, img, mask, cams = analyse_image(
                ex, path, true_label, label_map, class_names, want_mask, tfl)
            out_png = os.path.join(cohort_dir,
                                   panel_stem(i, path, true_label) + PANEL_EXT)
            render_panel(rec, img, mask, cams, out_png)
            rec['panel'] = out_png.replace('\\', '/')
            records.append(rec)

            mass = rec['layers'].get('deep', {}).get('cam_mass_in_leaf')
            # Windows consoles are cp1252; some source filenames are not.
            shown = os.path.basename(path).encode('ascii', 'replace').decode()
            line = (f"  {shown[:34]:34s} "
                    f"pred {_sanitize(rec['pred_label'])[:28]:28s} {rec['confidence']:.3f}")
            if mass is not None:
                line += f"  mass_in_leaf {mass:.3f} (area {rec['leaf_area_frac']:.3f})"
            print(line)

        summary = aggregate(records)
        report['cohorts'][name] = {'summary': summary, 'images': records}
        print(f"  -- deep: {json.dumps(summary['deep'])}")
        print(f"  -- mid:  {json.dumps(summary['mid'])}")
        print(f"  -- accuracy {summary['accuracy']} | keras/tflite agreement "
              f"{summary['keras_tflite_agreement']}")

    if args.sweep and want_mask:
        print(f'\n[leakage sweep] {args.sweep} random test images')
        sweep = leakage_sweep(ex, tfl, label_map, class_names, args.sweep)
        report['leakage_sweep'] = sweep
        for group in ('all', 'correct', 'wrong'):
            g = sweep[group]
            print(f"  {group:8s} n={g['n_images']:4d} masks={g['n_masks_plausible']:4d} "
                  f"deep mass {g['deep']['mean_cam_mass_in_leaf']} "
                  f"lift {g['deep']['mean_lift_vs_area']} "
                  f"peak_in_leaf {g['deep']['peak_in_leaf_rate']}")

    os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
    with open(args.json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
