"""
CropGuard AI — Phase 4: out-of-distribution detection ("is this even a leaf?").

The classifier is closed-set. Softmax over 17 leaf classes sums to 1 no matter what
comes in, so a photo of a chair, a face or a whole tomato plant still gets a
diagnosis — and `docs/EXPLAINABILITY.md` found one whole-plant scene scoring **0.983
calibrated, above the shipped HIGH gate**. Calibration (Phase 2) cannot reach this:
temperature scaling answers "how confident, given a leaf". Nothing in the pipeline
so far asks whether the input is a leaf at all.

Four detectors are compared on the same embeddings/logits:

  1. **MSP** — max softmax probability. The baseline everyone quotes, and the only
     one the shipped .tflite can already compute.
  2. **Energy** — `logsumexp(logits)` (the negative of the usual energy, so higher
     stays "more in-distribution"). Keeps the logit *magnitude*, which softmax
     normalizes away. It cannot be computed from the production artifact: that
     graph ends in softmax, and `log(p)` recovers logits only up to an additive
     constant — the one thing energy is not invariant to (`logsumexp(log p) == 0`
     for every input).
  3. **Mahalanobis** — distance to the nearest per-class Gaussian on the
     penultimate 576-d GAP embedding, shared covariance, Ledoit-Wolf shrinkage.
  4. **Cosine to the nearest class mean** — Mahalanobis with the covariance thrown
     away and the embedding L2-normalized. Not in the original plan; added because
     it beat the other three on the criterion that decides shippability (below)
     while needing 17x576 floats instead of a 576x576 precision matrix.

**The measurement that matters is not AUROC against the clean test split.** The two
embedding detectors score 0.9997 there, and that number is close to meaningless: the
clean split is studio-lit detached leaves on grey card, so a detector can ace it by
recognising the *studio*, not the leaf. The evidence is
`shifted` — the real field photos, which are in-scope inputs a gate must accept.
Scored against those, MSP and energy collapse to ~0.6 AUROC, and a threshold set
the textbook way (95% TPR on the clean validation split) rejects **every single
field photo**. So this module reports two operating points per detector:

    clean_val@95   the textbook threshold, kept as the before-picture
    field@90       threshold preserving 90% of the field photos — what ships

Populations:
  fit        — per-class subsample of the *train* split (class statistics only)
  id_val     — validation split; the clean-threshold population
  id_test    — test split; the clean ID number
  ood        — the 97-image negative set in real_world_test/_ood (+ any in
               web_sourced_test/_ood): faces, furniture, sky, text, walls, whole
               crop plants, other foliage, produce...
  shifted    — the real-world / web-sourced leaf photos. In-scope inputs from a
               shifted distribution; the field operating point is set on these and
               nothing else is fitted to them.

Everything runs on the **Keras float model** — the shipped TFLite artifact exposes
no logits or embeddings. `src/export_ood_model.py` re-exports a two-output artifact
and checks the winning score survives quantization.

Run:  python -m src.ood
      python -m src.ood --no-cache
      python -m src.ood --field-tpr 0.95 --fit-per-class 200
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.config import SEED, TEMPERATURE, TIER_HIGH, TIER_MODERATE
from src.data_pipeline import CSV_PATH
from src.eval_tiers import _load_split
from src.explain import BACKBONE, IMAGE_EXTS, KERAS_PATH, load_image

REPORT_PATH = os.path.join('outputs', 'ood_report.json')
SCORES_PLOT = os.path.join('outputs', 'ood_scores.png')
ROC_PLOT = os.path.join('outputs', 'ood_roc.png')
GATE_PARAMS_PATH = os.path.join('models', 'ood_gate.json')
WEB_COPIES = (os.path.join('web', 'public'),)
CACHE_DIR = os.path.join('outputs', 'cache')  # *.npy is gitignored

OOD_ROOTS = ('real_world_test', 'web_sourced_test')
LEAF_ROOTS = ('real_world_test', 'web_sourced_test')

BATCH = 32
FIT_PER_CLASS = 120
EMB_DIM = 576
# Clean-split TPR for the textbook operating point (and the FPR@95TPR convention).
CLEAN_TPR = 0.95
# Field TPR for the shipped operating point. 0.90 rather than 0.95 because the field
# set is n=37: the 5th percentile of 37 samples is one image, the 10th is four.
FIELD_TPR = 0.90
DETECTORS = ('msp', 'energy', 'mahalanobis', 'cosine')
# PCA widths probed for a cheaper Mahalanobis (see report['embedding_probe']).
PCA_DIMS = (32, 64, 128)


# ------------------------------------------------------------------ extraction

class Features:
    """Penultimate 576-d GAP embedding + true logits from the Keras float model.

    The graph is Rescaling -> MobileNetV3Small -> GAP -> Dropout -> Dense(softmax),
    so logits are not an output. src/explain.py re-applies the head by hand inside a
    gradient tape; the same trick without the tape is used here, and it hands us the
    embedding for free. Dropout is inference-only, so omitting it is exact — the
    reconstruction is asserted against model.predict().
    """

    def __init__(self, keras_path=KERAS_PATH):
        self.model = keras.models.load_model(keras_path)
        base = self.model.get_layer(BACKBONE)
        dense = self.model.get_layer('predictions')
        self.W = dense.kernel.numpy().astype(np.float64)
        self.b = dense.bias.numpy().astype(np.float64)
        self.backbone = keras.Model(base.inputs, base.output)

    def __call__(self, imgs: np.ndarray):
        """(embeddings [N,576], logits [N,17]) for a batch of [0,1] float images."""
        feat = self.backbone(imgs * 255.0, training=False).numpy()
        emb = feat.mean(axis=(1, 2)).astype(np.float64)
        return emb, emb @ self.W + self.b

    def verify_head(self, img: np.ndarray) -> float:
        """Max |softmax(reconstructed logits) - model.predict|. Must be ~0."""
        ref = self.model.predict(img[None, ...], verbose=0)[0]
        _, logits = self(img[None, ...])
        return float(np.max(np.abs(softmax(logits)[0] - ref)))


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def extract(paths, fx: Features, tag: str):
    """Run a list of image paths through the extractor, batched."""
    embs, logits = [], []
    for start in range(0, len(paths), BATCH):
        chunk = paths[start:start + BATCH]
        imgs = np.stack([load_image(p) for p in chunk])
        e, z = fx(imgs)
        embs.append(e)
        logits.append(z)
        if (start // BATCH) % 20 == 0 and len(paths) > BATCH:
            print(f'  {tag}: {min(start + BATCH, len(paths))}/{len(paths)}')
    if not embs:
        return np.zeros((0, EMB_DIM)), np.zeros((0, 17))
    return np.concatenate(embs), np.concatenate(logits)


def cached_extract(name: str, paths, fx_factory, use_cache=True):
    """extract() with an .npy cache keyed on population name and image count."""
    e_path = os.path.join(CACHE_DIR, f'ood_emb_{name}.npy')
    z_path = os.path.join(CACHE_DIR, f'ood_logits_{name}.npy')
    if use_cache and os.path.exists(e_path) and os.path.exists(z_path):
        emb, logits = np.load(e_path), np.load(z_path)
        if len(emb) == len(paths):
            print(f'  {name}: {len(emb)} cached')
            return emb, logits
        print(f'  {name}: cache stale ({len(emb)} != {len(paths)}), re-running')
    emb, logits = extract(paths, fx_factory(), name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(e_path, emb)
    np.save(z_path, logits)
    return emb, logits


# ------------------------------------------------------------------ populations

def split_paths(split: str):
    """(paths, y_true, class_names) for a dataset split."""
    split_df, label_map = _load_split(split)
    y = np.array([label_map[l] for l in split_df['label']])
    return list(split_df['image_path']), y, list(label_map.keys())


def fit_paths(per_class: int, seed=SEED):
    """A balanced per-class subsample of the train split, for the class statistics."""
    df = pd.read_csv(CSV_PATH)
    label_map = {l: i for i, l in enumerate(sorted(df['label'].unique()))}
    train = df[df['split'] == 'train']
    rng = np.random.default_rng(seed)
    paths, y = [], []
    for label, group in train.groupby('label'):
        idx = rng.permutation(len(group))[:per_class]
        for p in np.asarray(group['image_path'])[idx]:
            paths.append(p)
            y.append(label_map[label])
    return paths, np.array(y)


def walk_ood(roots=OOD_ROOTS):
    """(paths, categories) under */_ood, category = the folder holding the image.

    Both a flat `_ood/img.jpg` layout and `_ood/<category>/img.jpg` are read; the
    former is filed as 'uncategorised' so nothing is silently dropped.
    """
    paths, cats = [], []
    for root in roots:
        base = os.path.join(root, '_ood')
        if not os.path.isdir(base):
            continue
        for dirpath, _, files in os.walk(base):
            rel = os.path.relpath(dirpath, base)
            cat = 'uncategorised' if rel == '.' else rel.replace('\\', '/').split('/')[0]
            for f in sorted(files):
                if f.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(dirpath, f))
                    cats.append(cat)
    order = np.argsort(paths)
    return [paths[i] for i in order], [cats[i] for i in order]


def walk_leaves(roots=LEAF_ROOTS):
    """The shifted-but-in-scope leaf photos: every non-underscore class folder."""
    paths, srcs = [], []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for folder in sorted(os.listdir(root)):
            sub = os.path.join(root, folder)
            if not os.path.isdir(sub) or folder.startswith(('_', '.')):
                continue
            for f in sorted(os.listdir(sub)):
                if f.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(sub, f))
                    srcs.append(root)
    return paths, srcs


# -------------------------------------------------------------------- detectors

def msp_score(logits: np.ndarray) -> np.ndarray:
    """Max softmax probability. Higher = more in-distribution."""
    return softmax(logits).max(axis=1)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """`logsumexp(logits/T) * T` — negative free energy, so higher = ID.

    Liu et al. 2020. Unlike MSP this keeps the logit magnitude: a peaked but small
    logit vector (nothing in the image excites any class strongly) scores low even
    when its softmax maximum is high.
    """
    z = np.asarray(logits, dtype=np.float64) / T
    m = z.max(axis=1)
    return (m + np.log(np.exp(z - m[:, None]).sum(axis=1))) * T


class Mahalanobis:
    """Nearest-class Mahalanobis distance on embeddings, shared covariance.

    Lee et al. 2018. Ledoit-Wolf rather than the empirical covariance: 576 dimensions
    against ~2k fit images is close enough to ill-conditioned that the empirical
    inverse amplifies noise directions.
    """

    def __init__(self, emb: np.ndarray, y: np.ndarray, n_classes: int):
        self.means = np.stack([emb[y == c].mean(axis=0) for c in range(n_classes)])
        centered = emb - self.means[y]
        lw = LedoitWolf(assume_centered=True).fit(centered)
        self.precision = lw.precision_
        self.shrinkage = float(lw.shrinkage_)

    def distances(self, emb: np.ndarray) -> np.ndarray:
        """[N, C] squared Mahalanobis distance to every class mean."""
        out = np.empty((len(emb), len(self.means)))
        for c, mu in enumerate(self.means):
            d = emb - mu
            out[:, c] = np.einsum('ij,jk,ik->i', d, self.precision, d)
        return out

    def score(self, emb: np.ndarray) -> np.ndarray:
        """Negative distance to the nearest class. Higher = more in-distribution."""
        return -self.distances(emb).min(axis=1)


class ClassMeanCosine:
    """Cosine similarity to the nearest class-mean embedding.

    Mahalanobis with the covariance discarded and the embedding direction kept. Two
    reasons it is the one that ships:

      * it is the only detector that separates *field leaves* from non-leaves as well
        as it separates studio leaves from them — dropping the covariance drops
        exactly the part of the model that had memorised the studio distribution;
      * 17x576 floats (38 KB) against Mahalanobis's 576x576 precision matrix
        (1.3 MB), which would more than double the 1.15 MB the browser downloads.

    Scale-invariance is doing the work here. A field photo's embedding points in
    almost the same direction as a studio leaf's but with a different magnitude,
    and Mahalanobis charges for that magnitude while cosine does not.
    """

    def __init__(self, emb: np.ndarray, y: np.ndarray, n_classes: int):
        means = np.stack([emb[y == c].mean(axis=0) for c in range(n_classes)])
        self.means = means
        self.unit_means = means / np.linalg.norm(means, axis=1, keepdims=True)

    def score(self, emb: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(emb, axis=1, keepdims=True)
        return ((emb / np.maximum(n, 1e-12)) @ self.unit_means.T).max(axis=1)


# ---------------------------------------------------------------------- metrics

def threshold_at_tpr(id_scores: np.ndarray, tpr: float) -> float:
    """Score threshold that accepts `tpr` of the given in-distribution population."""
    return float(np.quantile(id_scores, 1.0 - tpr))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def evaluate(id_scores, ood_scores) -> dict:
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    s = np.r_[id_scores, ood_scores]
    thr = threshold_at_tpr(id_scores, CLEAN_TPR)
    return {
        'auroc': float(roc_auc_score(y, s)),
        'aupr_id': float(average_precision_score(y, s)),
        'aupr_ood': float(average_precision_score(1 - y, -s)),
        'fpr_at_95tpr': float(np.mean(np.asarray(ood_scores) >= thr)),
        'n_id': int(len(id_scores)),
        'n_ood': int(len(ood_scores)),
    }


def operating_point(scores: dict, thr: float, ood_cats) -> dict:
    """Accept rates for one threshold, overall and per OOD category."""
    cat_arr = np.array(ood_cats)
    return {
        'threshold': float(thr),
        'accept_rate': {name: float(np.mean(scores[name] >= thr))
                        for name in ('id_val', 'id_test', 'ood', 'shifted')},
        'ood_accept_by_category': {
            c: float(np.mean(scores['ood'][cat_arr == c] >= thr))
            for c in sorted(set(ood_cats))},
    }


# ----------------------------------------------------------------------- plots

def plot_scores(scores: dict, thresholds: dict, path=SCORES_PLOT):
    """Score distributions per detector: clean ID, field leaves, OOD."""
    fig, axes = plt.subplots(1, len(DETECTORS), figsize=(4.6 * len(DETECTORS), 4))
    for ax, det in zip(np.atleast_1d(axes), DETECTORS):
        s = scores[det]
        bins = np.histogram_bin_edges(
            np.r_[s['id_test'], s['ood'], s['shifted']], bins=40)
        ax.hist(s['id_test'], bins=bins, alpha=0.55, density=True,
                label=f"ID test (n={len(s['id_test'])})", color='#2c7fb8')
        ax.hist(s['shifted'], bins=bins, alpha=0.55, density=True,
                label=f"field leaves (n={len(s['shifted'])})", color='#31a354')
        ax.hist(s['ood'], bins=bins, alpha=0.55, density=True,
                label=f"OOD (n={len(s['ood'])})", color='#d95f0e')
        ax.axvline(thresholds[det]['field'], color='k', ls='--', lw=1.2,
                   label='field@90 gate')
        ax.axvline(thresholds[det]['clean'], color='k', ls=':', lw=1.2,
                   label='clean_val@95')
        ax.set_title(det)
        ax.set_xlabel('score (higher = in-distribution)')
        ax.set_ylabel('density')
        ax.legend(fontsize=7)
    fig.suptitle('OOD score distributions — Keras float model')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc(scores: dict, path=ROC_PLOT):
    """Two ROC panels: the flattering clean comparison, and the one that decides."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    panels = [('id_test', 'clean studio leaves vs OOD (flattering, and misleading)'),
              ('shifted', 'field leaves vs OOD (the shippable question)')]
    for ax, (pop, title) in zip(axes, panels):
        for det in DETECTORS:
            s = scores[det]
            y = np.r_[np.ones(len(s[pop])), np.zeros(len(s['ood']))]
            sc = np.r_[s[pop], s['ood']]
            fpr, tpr, _ = roc_curve(y, sc)
            ax.plot(fpr, tpr, lw=1.8,
                    label=f'{det} (AUROC {roc_auc_score(y, sc):.4f})')
        ax.plot([0, 1], [0, 1], 'k:', lw=1, label='chance')
        ax.set_xlabel('FPR — OOD images accepted')
        ax.set_ylabel('TPR — leaves accepted')
        ax.set_title(title, fontsize=10)
        ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------- shippable gate params

def write_gate_params(cos: ClassMeanCosine, threshold: float, class_names,
                      meta: dict):
    """Class-mean directions + threshold for web/src/lib/oodGate.ts to load.

    Unit-normalized means are stored, not raw ones, so the browser does one dot
    product per class and no normalization of its own. 17x576 float32 rounded to 6
    decimals is ~230 KB of JSON (~40 KB gzipped) — the reason the covariance-free
    detector is the one that can ship.
    """
    payload = {
        'detector': 'class_mean_cosine',
        'note': ('score = max_c cos(embedding, unit_mean_c); reject below threshold. '
                 'Derived by python -m src.ood; see docs/OOD.md.'),
        'embedding_dim': int(cos.unit_means.shape[1]),
        'threshold': float(threshold),
        'class_names': list(class_names),
        'unit_means': [[round(float(v), 6) for v in row] for row in cos.unit_means],
        **meta,
    }
    os.makedirs('models', exist_ok=True)
    with open(GATE_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    for dest in WEB_COPIES:
        if os.path.isdir(dest):
            with open(os.path.join(dest, 'ood_gate.json'), 'w', encoding='utf-8') as f:
                json.dump(payload, f)
    return GATE_PARAMS_PATH


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--fit-per-class', type=int, default=FIT_PER_CLASS)
    ap.add_argument('--field-tpr', type=float, default=FIELD_TPR)
    ap.add_argument('--clean-tpr', type=float, default=CLEAN_TPR)
    ap.add_argument('--no-write-params', action='store_true',
                    help='skip writing models/ood_gate.json')
    args = ap.parse_args()
    use_cache = not args.no_cache

    ood_paths, ood_cats = walk_ood()
    if not ood_paths:
        raise SystemExit(
            'no negatives found under */_ood — build the set first:\n'
            '  python scripts/harvest_ood.py --out <staging> --per-category 9\n'
            '  python scripts/curate_ood.py --staging <staging>')
    leaf_paths, leaf_srcs = walk_leaves()
    if not leaf_paths:
        raise SystemExit('no field leaf photos found — the field operating point '
                         'cannot be set (see real_world_test/README.md)')
    val_paths, y_val, class_names = split_paths('val')
    test_paths, y_test, _ = split_paths('test')
    f_paths, y_fit = fit_paths(args.fit_per_class)

    print(f'populations: fit={len(f_paths)} val={len(val_paths)} '
          f'test={len(test_paths)} ood={len(ood_paths)} field={len(leaf_paths)}')

    holder = {}

    def fx_factory():
        if 'fx' not in holder:
            print('loading Keras model...')
            holder['fx'] = Features()
        return holder['fx']

    emb_fit, log_fit = cached_extract(f'fit{args.fit_per_class}', f_paths,
                                      fx_factory, use_cache)
    emb_val, log_val = cached_extract('val', val_paths, fx_factory, use_cache)
    emb_test, log_test = cached_extract('test', test_paths, fx_factory, use_cache)
    emb_ood, log_ood = cached_extract('ood', ood_paths, fx_factory, use_cache)
    emb_shift, log_shift = cached_extract('shifted', leaf_paths, fx_factory, use_cache)

    head_err = None
    if 'fx' in holder:
        head_err = holder['fx'].verify_head(load_image(test_paths[0]))
        print(f'head reconstruction max|delta| = {head_err:.3g}')
        assert head_err < 1e-4, head_err

    maha = Mahalanobis(emb_fit, y_fit, len(class_names))
    cos = ClassMeanCosine(emb_fit, y_fit, len(class_names))
    print(f'Mahalanobis: Ledoit-Wolf shrinkage = {maha.shrinkage:.4f}')

    scorers = {
        'msp': lambda e, z: msp_score(z),
        'energy': lambda e, z: energy_score(z),
        'mahalanobis': lambda e, z: maha.score(e),
        'cosine': lambda e, z: cos.score(e),
    }
    pops = {'id_val': (emb_val, log_val), 'id_test': (emb_test, log_test),
            'ood': (emb_ood, log_ood), 'shifted': (emb_shift, log_shift)}

    scores, results, thresholds = {}, {}, {}
    for det, fn in scorers.items():
        s = {name: fn(e, z) for name, (e, z) in pops.items()}
        scores[det] = s
        thresholds[det] = {
            # Textbook: set on the clean validation split, never on test.
            'clean': threshold_at_tpr(s['id_val'], args.clean_tpr),
            # Shipped: set on the field photos, the population a user resembles.
            'field': threshold_at_tpr(s['shifted'], args.field_tpr),
        }
        results[det] = {
            'clean_vs_ood': evaluate(s['id_test'], s['ood']),
            'field_vs_ood': {
                'auroc': auroc(s['shifted'], s['ood']),
                'n_field': int(len(s['shifted'])), 'n_ood': int(len(s['ood'])),
            },
            'operating_points': {
                f'clean_val@{args.clean_tpr:g}':
                    operating_point(s, thresholds[det]['clean'], ood_cats),
                f'field@{args.field_tpr:g}':
                    operating_point(s, thresholds[det]['field'], ood_cats),
            },
            'mean_score': {name: float(np.mean(s[name])) for name in pops},
        }

    # Ranked by the field comparison: the clean one cannot separate the detectors
    # (all >= 0.99) and rewards recognising the studio rather than the leaf.
    winner = max(DETECTORS, key=lambda d: results[d]['field_vs_ood']['auroc'])
    ship_thr = thresholds[winner]['field']

    # Why the 576-d embedding ships whole. PCA compresses it cheaply and keeps the
    # clean number intact, which is exactly the trap: the leading components carry
    # the studio distribution, and the field-vs-OOD signal lives in the tail.
    mu = emb_fit.mean(axis=0)
    _, _, Vt = np.linalg.svd(emb_fit - mu, full_matrices=False)
    probe = {}
    for k in PCA_DIMS:
        P = Vt[:k]
        proj = lambda e: (e - mu) @ P.T
        mk = Mahalanobis(proj(emb_fit), y_fit, len(class_names))
        probe[f'pca{k}_mahalanobis'] = {
            'auroc_clean_vs_ood': auroc(mk.score(proj(emb_test)), mk.score(proj(emb_ood))),
            'auroc_field_vs_ood': auroc(mk.score(proj(emb_shift)), mk.score(proj(emb_ood))),
            'params': int(k * EMB_DIM + EMB_DIM + len(class_names) * k + k * k),
        }
    probe['mahalanobis_full'] = {
        'auroc_clean_vs_ood': results['mahalanobis']['clean_vs_ood']['auroc'],
        'auroc_field_vs_ood': results['mahalanobis']['field_vs_ood']['auroc'],
        'params': int(len(class_names) * EMB_DIM + EMB_DIM * EMB_DIM),
    }
    probe['cosine'] = {
        'auroc_clean_vs_ood': results['cosine']['clean_vs_ood']['auroc'],
        'auroc_field_vs_ood': results['cosine']['field_vs_ood']['auroc'],
        'params': int(len(class_names) * EMB_DIM),
    }

    # What the gate is worth in UI terms: negatives that currently reach a tier that
    # shows a diagnosis, before and after gating.
    conf_ood = softmax(log_ood, TEMPERATURE).max(axis=1)
    passed = scores[winner]['ood'] >= ship_thr
    tier_impact = {
        'calibrated_conf_mean': float(conf_ood.mean()),
        'calibrated_conf_max': float(conf_ood.max()),
        'high_tier_before': float(np.mean(conf_ood >= TIER_HIGH)),
        'high_tier_after_gate': float(np.mean((conf_ood >= TIER_HIGH) & passed)),
        'diagnosis_shown_before': float(np.mean(conf_ood >= TIER_MODERATE)),
        'diagnosis_shown_after_gate': float(np.mean((conf_ood >= TIER_MODERATE) & passed)),
    }

    shift_arr = np.array(leaf_srcs)
    shifted_by_source = {
        det: {src: float(np.mean(scores[det]['shifted'][shift_arr == src]
                                 >= thresholds[det]['field']))
              for src in sorted(set(leaf_srcs))} for det in DETECTORS
    }

    plot_scores(scores, thresholds)
    plot_roc(scores)

    cat_arr = np.array(ood_cats)
    report = {
        'model': KERAS_PATH,
        'note': ('scores computed on the Keras float model; the shipped .tflite '
                 'exposes no logits or embeddings — see src/export_ood_model.py'),
        'head_reconstruction_max_abs_err': head_err,
        'populations': {
            'fit': {'n': len(f_paths), 'per_class': args.fit_per_class,
                    'split': 'train', 'seed': SEED},
            'id_val': {'n': len(val_paths)}, 'id_test': {'n': len(test_paths)},
            'ood': {'n': len(ood_paths), 'categories':
                    {c: int((cat_arr == c).sum()) for c in sorted(set(ood_cats))}},
            'shifted': {'n': len(leaf_paths),
                        'sources': {s: int((shift_arr == s).sum())
                                    for s in sorted(set(leaf_srcs))}},
        },
        'mahalanobis_shrinkage': maha.shrinkage,
        'detectors': results,
        'winner': winner,
        'winner_selected_on': 'field_vs_ood AUROC',
        'shipped_threshold': ship_thr,
        'shipped_field_tpr': args.field_tpr,
        'embedding_probe': probe,
        'shifted_accept_rate_at_field_threshold': shifted_by_source,
        'tier_impact_on_ood': tier_impact,
        'plots': {'scores': SCORES_PLOT, 'roc': ROC_PLOT},
    }

    if not args.no_write_params and winner == 'cosine':
        path = write_gate_params(cos, ship_thr, class_names, {
            'fit': {'split': 'train', 'per_class': args.fit_per_class, 'seed': SEED},
            'field_tpr': args.field_tpr,
            'field_n': len(leaf_paths),
        })
        report['gate_params'] = path
        print(f'gate params -> {path}')
    elif not args.no_write_params:
        print(f'winner is {winner}, not cosine — no browser params written '
              f'(only the cosine gate has a shippable parameter set)')

    os.makedirs('outputs', exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print('\n--- AUROC vs the negative set ---')
    print(f"{'detector':13s} {'clean':>8s} {'field':>8s}   "
          f"{'ood accept':>10s} {'field keep':>10s}  (at field@"
          f"{args.field_tpr:g} threshold)")
    for det in DETECTORS:
        r = results[det]
        op = r['operating_points'][f'field@{args.field_tpr:g}']
        print(f"  {det:11s} {r['clean_vs_ood']['auroc']:8.4f} "
              f"{r['field_vs_ood']['auroc']:8.4f}   "
              f"{op['accept_rate']['ood']:10.3f} {op['accept_rate']['shifted']:10.3f}")
    print(f'  winner: {winner} (threshold {ship_thr:.4f})')
    print('\n--- the textbook threshold, for comparison ---')
    for det in DETECTORS:
        op = results[det]['operating_points'][f'clean_val@{args.clean_tpr:g}']
        print(f"  {det:11s} keeps {op['accept_rate']['shifted']:.3f} of field leaves, "
              f"accepts {op['accept_rate']['ood']:.3f} of OOD")
    print(f"\nOOD reaching HIGH tier: {tier_impact['high_tier_before']:.3f}"
          f" -> {tier_impact['high_tier_after_gate']:.3f} with the gate")
    print(f'\nreport -> {REPORT_PATH}\nplots  -> {SCORES_PLOT}, {ROC_PLOT}')


if __name__ == '__main__':
    main()
