"""
CropGuard AI — calibration analysis and data-derived confidence thresholds.

The tier system in web/src/lib/confidenceTier.ts gates the UI on top-1 softmax
probability, with thresholds (0.85 / 0.50) originally picked by intuition. But a
softmax score is not a probability: a network can be 95% "confident" and right
only 80% of the time. This script measures that gap and closes it.

On the **validation** split (never used for gradient updates, and the only split
the thresholds are allowed to see):

  1. Calibration error — ECE and MCE over 15 equal-width confidence bins,
     plus Brier score and NLL.
  2. Reliability diagram -> outputs/reliability_diagram.png
  3. Temperature scaling — a single scalar T fit by minimizing NLL, reported
     as ECE-before vs ECE-after. T is then applied unchanged to the test split,
     which is the honest generalization check.
  4. Re-derived tier thresholds: the smallest confidence at which empirical
     selective accuracy meets a target (default 0.99 for HIGH, 0.50 for
     MODERATE — i.e. "more likely right than wrong"), plus a sweep showing
     what each candidate target costs in coverage.
  5. Risk-coverage curve -> outputs/risk_coverage.png, with AURC.

Everything runs on models/cropguard_v1_production.tflite — the exact artifact
the browser loads, not the Keras model — so the numbers describe what ships.

Logits note: the production graph ends in softmax, so true logits are not
exposed. log(p) recovers them up to an additive constant, and softmax is
invariant to that constant, so temperature scaling on log(p) is exact.
Probabilities are floored at 1e-30 before the log to avoid -inf.

Run:  python -m src.calibration
      python -m src.calibration --high-target 0.98 --no-cache
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from src.config import LEGACY_TIER_HIGH, LEGACY_TIER_MODERATE
from src.eval_tiers import TFLITE_PATH, _load_split, run_inference

REPORT_PATH = os.path.join('outputs', 'calibration_report.json')
RELIABILITY_PATH = os.path.join('outputs', 'reliability_diagram.png')
RISK_COVERAGE_PATH = os.path.join('outputs', 'risk_coverage.png')
CACHE_DIR = os.path.join('outputs', 'cache')  # *.npy is gitignored

N_BINS = 15
PROB_FLOOR = 1e-30
COVERAGE_GRID = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# MCE over all bins is dominated by bins holding a handful of images; this is the
# minimum bin population for the reported "robust" variant.
MCE_MIN_N = 10
# Targets swept for the HIGH tier. Anything at or below the model's unconditional
# accuracy (0.94) is satisfied by answering almost everything, so the useful range
# starts above it.
TARGET_SWEEP = [0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
# Window width for the local (MODERATE) threshold criterion.
MODERATE_WINDOW = 0.05


# ---------------------------------------------------------------- inference

def get_probs(split: str, use_cache: bool = True):
    """(probs [N,17], y_true [N], class_names) for a split, cached on disk."""
    p_path = os.path.join(CACHE_DIR, f'probs_{split}.npy')
    y_path = os.path.join(CACHE_DIR, f'labels_{split}.npy')
    split_df, label_map = _load_split(split)
    class_names = list(label_map.keys())

    if use_cache and os.path.exists(p_path) and os.path.exists(y_path):
        probs, y_true = np.load(p_path), np.load(y_path)
        if len(probs) == len(split_df):
            print(f'  {split}: {len(probs)} cached predictions')
            return probs, y_true, class_names
        print(f'  {split}: cache stale ({len(probs)} != {len(split_df)}), re-running')

    print(f'  {split}: running {len(split_df)} images through {TFLITE_PATH}')
    probs, y_true = run_inference(split_df, label_map)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(p_path, probs)
    np.save(y_path, y_true)
    return probs, y_true, class_names


# ------------------------------------------------------- temperature scaling

def to_logits(probs: np.ndarray) -> np.ndarray:
    """log(p) — a valid logit set, since softmax ignores additive constants."""
    return np.log(np.clip(probs.astype(np.float64), PROB_FLOOR, 1.0))


def softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true], PROB_FLOOR, 1.0)
    return float(-np.mean(np.log(p)))


def fit_temperature(probs: np.ndarray, y_true: np.ndarray):
    """Single scalar T minimizing NLL. T > 1 softens, T < 1 sharpens."""
    logits = to_logits(probs)

    def obj(log_t):
        return nll(softmax_T(logits, float(np.exp(log_t))), y_true)

    res = minimize_scalar(obj, bounds=(np.log(0.05), np.log(20.0)),
                          method='bounded', options={'xatol': 1e-5})
    return float(np.exp(res.x)), float(res.fun)


# --------------------------------------------------------------- calibration

def bin_stats(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """Equal-width confidence bins: lo, hi, n, mean confidence, accuracy."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        out.append({
            'lo': float(lo), 'hi': float(hi), 'n': n,
            'mean_confidence': float(conf[m].mean()) if n else None,
            'accuracy': float(correct[m].mean()) if n else None,
        })
    return out


def calibration_metrics(probs: np.ndarray, y_true: np.ndarray) -> dict:
    top1 = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (top1 == y_true).astype(np.float64)
    bins = bin_stats(conf, correct)

    n = len(y_true)
    ece = sum(b['n'] / n * abs(b['accuracy'] - b['mean_confidence'])
              for b in bins if b['n'])

    def worst_bin(min_n):
        cand = [b for b in bins if b['n'] >= min_n]
        if not cand:
            return 0.0, None
        b = max(cand, key=lambda b: abs(b['accuracy'] - b['mean_confidence']))
        return abs(b['accuracy'] - b['mean_confidence']), b

    mce, mce_bin = worst_bin(1)
    mce_robust, mce_robust_bin = worst_bin(MCE_MIN_N)

    onehot = np.zeros(probs.shape, dtype=np.float64)
    onehot[np.arange(n), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    return {
        'n': n,
        'accuracy': float(correct.mean()),
        'mean_confidence': float(conf.mean()),
        # Positive = overconfident: the model claims more than it delivers.
        'confidence_minus_accuracy': float(conf.mean() - correct.mean()),
        'ece': float(ece),
        'mce': float(mce),
        'mce_bin': mce_bin,
        # MCE ignoring bins with < MCE_MIN_N images — the all-bins figure is
        # routinely set by a single sparse bin and reads as far worse than the
        # model behaves.
        'mce_robust': float(mce_robust),
        'mce_robust_bin': mce_robust_bin,
        'brier': brier,
        'nll': nll(probs, y_true),
        'bins': bins,
    }


# ----------------------------------------------------- threshold derivation

def selective_accuracy(conf, correct, tau):
    m = conf >= tau
    return (float(correct[m].mean()), int(m.sum())) if m.any() else (None, 0)


def derive_threshold(conf, correct, target, min_n=30, window=None):
    """Smallest tau where accuracy meets `target` and keeps meeting it for every
    larger tau — monotone-safe, so one lucky bin cannot pick the threshold.

    Two criteria, and the difference matters:

    - cumulative (`window=None`): accuracy over everything at or above tau.
      Correct for HIGH, where the question is "of the answers we give, how many
      are right?"
    - local (`window=w`): accuracy over the slice [tau, tau+w) only. Correct for
      the MODERATE floor, where the question is "at *this* confidence, is the
      top-1 still more likely right than wrong?" The cumulative version is
      useless here — it is dragged above any sane target by the high-confidence
      mass above it, and so slides all the way to zero.
    """
    grid = np.round(np.arange(0.01, 1.00, 0.005), 4)
    ok = []
    for tau in grid:
        m = conf >= tau
        if window is not None:
            m = m & (conf < tau + window)
        n = int(m.sum())
        acc = float(correct[m].mean()) if n else None
        ok.append(n >= min_n and acc is not None and acc >= target)
    # Walk down from the top and keep the last contiguous run of successes.
    best = None
    for i in range(len(grid) - 1, -1, -1):
        if ok[i]:
            best = float(grid[i])
        elif best is not None:
            break
    return best


def sweep_targets(conf, correct, targets=TARGET_SWEEP):
    """What each candidate HIGH accuracy target costs in coverage.

    The point of this table: a target below the model's unconditional accuracy
    is met by answering nearly everything, which *raises* the absolute rate of
    confident errors even though the conditional accuracy target is satisfied.
    Choosing the target is a product decision, and this makes the price visible.
    """
    rows = []
    for target in targets:
        tau = derive_threshold(conf, correct, target)
        if tau is None:
            rows.append({'target': target, 'threshold': None})
            continue
        m = conf >= tau
        rows.append({
            'target': target,
            'threshold': tau,
            'coverage': float(m.mean()),
            'accuracy': float(correct[m].mean()),
            # Share of ALL images that get a confident wrong diagnosis — the
            # metric the tier system exists to hold down.
            'confidently_wrong_rate': float((m & (correct == 0)).mean()),
        })
    return rows


def tier_table(conf, correct, in_top3, high, moderate):
    tiers = np.where(conf >= high, 'HIGH',
                     np.where(conf >= moderate, 'MODERATE', 'LOW'))
    out = {}
    for name in ('HIGH', 'MODERATE', 'LOW'):
        m = tiers == name
        n = int(m.sum())
        out[name] = {
            'n': n,
            'coverage': float(m.mean()),
            'accuracy': float(correct[m].mean()) if n else None,
            'top3_accuracy': float(in_top3[m].mean()) if n else None,
        }
    out['confidently_wrong_rate'] = float(((tiers == 'HIGH') & (correct == 0)).mean())
    return out


# ------------------------------------------------------------ risk-coverage

def risk_coverage(conf, correct):
    """Sort by confidence descending; risk = error rate over the retained head."""
    order = np.argsort(-conf)
    c = correct[order]
    n = len(c)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = np.cumsum(1.0 - c) / k
    aurc = float(np.trapezoid(risk, coverage))
    points = []
    for target in COVERAGE_GRID:
        idx = max(0, int(round(target * n)) - 1)
        points.append({
            'coverage': float(coverage[idx]),
            'risk': float(risk[idx]),
            'accuracy': float(1.0 - risk[idx]),
            'confidence_threshold': float(conf[order][idx]),
        })
    return coverage, risk, aurc, points


# ------------------------------------------------------------------- plots

def plot_reliability(before, after, T, path=RELIABILITY_PATH):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw={'height_ratios': [3, 1]})
    panels = ((f"Uncalibrated  (ECE {before['ece']:.4f})", before),
              (f"Temperature scaled, T={T:.3f}  (ECE {after['ece']:.4f})", after))
    for col, (title, m) in enumerate(panels):
        ax = axes[0][col]
        bins = m['bins']
        width = bins[0]['hi'] - bins[0]['lo']
        # Empty bins are omitted entirely: drawn as accuracy 0 they read as
        # catastrophic miscalibration where there is simply no data. Sparse bins
        # (< MCE_MIN_N images) are drawn faded — they are what inflates MCE.
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
        for i, b in enumerate(bins):
            if not b['n']:
                continue
            c = (b['lo'] + b['hi']) / 2
            acc, mc = b['accuracy'], b['mean_confidence']
            sparse = b['n'] < MCE_MIN_N
            ax.bar(c, acc, width=width * 0.95, color='#2b7bba',
                   alpha=0.35 if sparse else 1.0, edgecolor='white',
                   label='empirical accuracy' if i == len(bins) - 1 else None)
            ax.bar(c, mc - acc, bottom=acc, width=width * 0.95, color='#d1495b',
                   alpha=0.2 if sparse else 0.45, edgecolor='#d1495b',
                   hatch='//',
                   label='gap (confidence - accuracy)'
                         if i == len(bins) - 1 else None)
            if sparse:
                ax.text(c, max(min(acc, mc) - 0.03, 0.02), f"n={b['n']}",
                        ha='center',
                        va='top', fontsize=6, color='0.35', rotation=90)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('accuracy')
        ax.legend(loc='upper left', fontsize=8)
        rb = m['mce_robust_bin']
        rb_txt = (f" [{rb['lo']:.2f}-{rb['hi']:.2f}), n={rb['n']}" if rb else '')
        ax.text(0.98, 0.05,
                f"MCE {m['mce']:.4f} (all bins)\n"
                f"MCE {m['mce_robust']:.4f} (n>={MCE_MIN_N}){rb_txt}\n"
                f"Brier {m['brier']:.4f}\nNLL {m['nll']:.4f}",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='0.7'))

        ax2 = axes[1][col]
        ax2.bar([(b['lo'] + b['hi']) / 2 for b in bins],
                [max(b['n'], 0.1) for b in bins], width=width * 0.95,
                color='0.5')
        ax2.axhline(MCE_MIN_N, color='#d1495b', ls=':', lw=1)
        ax2.set_yscale('log')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('top-1 confidence')
        ax2.set_ylabel('images (log)')

    fig.suptitle('CropGuard AI — reliability diagram (validation split, '
                 f"{before['n']} images, {N_BINS} bins)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(cov, risk, aurc, high, moderate, conf,
                       path=RISK_COVERAGE_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(cov, risk, color='#d1495b', lw=2)
    axes[0].fill_between(cov, 0, risk, color='#d1495b', alpha=0.12)
    axes[0].set_xlabel('coverage (fraction of images answered)')
    axes[0].set_ylabel('risk (error rate on answered images)')
    axes[0].set_title(f'Risk-coverage curve   AURC = {aurc:.4f}')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(alpha=0.3)

    axes[1].plot(cov, 1 - risk, color='#2b7bba', lw=2, label='selective accuracy')
    for tau, color, label in ((high, '#1b7837', 'HIGH'),
                              (moderate, '#e08214', 'MODERATE')):
        c = float((conf >= tau).mean())
        if c > 0:
            axes[1].axvline(c, color=color, ls='--', lw=1.2,
                            label=f'{label} tau={tau:.3f} (coverage {c:.2f})')
    axes[1].set_xlabel('coverage')
    axes[1].set_ylabel('accuracy on answered images')
    axes[1].set_title('Selective prediction — accuracy bought by abstaining')
    axes[1].set_xlim(0, 1)
    axes[1].legend(loc='lower left', fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle('CropGuard AI — selective prediction '
                 '(validation split, temperature scaled)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main

def _fmt(v, nd=4):
    return f'{v:.{nd}f}' if isinstance(v, (int, float)) else '-'


def print_metrics(tag, m):
    print(f"  {tag:<22} acc {m['accuracy']:.4f}  conf {m['mean_confidence']:.4f}  "
          f"ECE {m['ece']:.4f}  MCE {m['mce']:.4f}  "
          f"MCE>={MCE_MIN_N} {m['mce_robust']:.4f}  "
          f"Brier {m['brier']:.4f}  NLL {m['nll']:.4f}")


def print_tiers(tiers):
    print('\nTier behaviour:')
    print(f"  {'configuration':<40}{'tier':<10}{'cov':>8}{'acc':>9}{'top3':>9}")
    for key, table in tiers.items():
        for name in ('HIGH', 'MODERATE', 'LOW'):
            t = table[name]
            print(f"  {key if name == 'HIGH' else '':<40}{name:<10}"
                  f"{t['coverage']:>8.3f}{_fmt(t['accuracy']):>9}"
                  f"{_fmt(t['top3_accuracy']):>9}")
        print(f"  {'':<40}confidently wrong: "
              f"{table['confidently_wrong_rate'] * 100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--high-target', type=float, default=0.99,
                    help='target selective accuracy for the HIGH tier. Must be '
                         'well above the unconditional accuracy (~0.94) to buy '
                         'anything — see the target sweep in the output.')
    ap.add_argument('--moderate-target', type=float, default=0.50,
                    help='target in-band accuracy for the MODERATE tier')
    ap.add_argument('--no-cache', action='store_true',
                    help='ignore cached predictions and re-run inference')
    args = ap.parse_args()

    print('=' * 74)
    print('CropGuard AI — calibration (fit on val, verified on test)')
    print('=' * 74)
    print(f'Model: {TFLITE_PATH} ({os.path.getsize(TFLITE_PATH) / 1e6:.2f} MB)')

    val_probs, val_y, class_names = get_probs('val', not args.no_cache)
    test_probs, test_y, _ = get_probs('test', not args.no_cache)

    # --- 1. calibration before / after temperature scaling -----------------
    T, fit_nll = fit_temperature(val_probs, val_y)
    val_cal = softmax_T(to_logits(val_probs), T)
    test_cal = softmax_T(to_logits(test_probs), T)

    metrics = {
        'val_raw': calibration_metrics(val_probs, val_y),
        'val_calibrated': calibration_metrics(val_cal, val_y),
        'test_raw': calibration_metrics(test_probs, test_y),
        'test_calibrated': calibration_metrics(test_cal, test_y),
    }

    print(f'\nTemperature fit on val by NLL minimization: T = {T:.4f} '
          f'({"softens" if T > 1 else "sharpens"} the distribution)')
    print('\nCalibration:')
    for tag in ('val_raw', 'val_calibrated', 'test_raw', 'test_calibrated'):
        print_metrics(tag, metrics[tag])
    d_val = metrics['val_raw']['ece'] - metrics['val_calibrated']['ece']
    d_test = metrics['test_raw']['ece'] - metrics['test_calibrated']['ece']
    print(f'\n  ECE reduction — val {d_val:+.4f}  test {d_test:+.4f} '
          '(test is the honest number: T was not fit on it)')

    # --- 2. derived thresholds --------------------------------------------
    vconf = val_cal.max(axis=1)
    vcorrect = (val_cal.argmax(axis=1) == val_y).astype(np.float64)
    v_top3 = np.any(val_cal.argsort(axis=1)[:, -3:] == val_y[:, None], axis=1)

    sweep = sweep_targets(vconf, vcorrect)
    print('\nHIGH-threshold target sweep (calibrated val):')
    print(f"  {'target':>8}{'tau':>8}{'coverage':>10}{'accuracy':>10}"
          f"{'confidently wrong':>20}")
    for r in sweep:
        if r['threshold'] is None:
            print(f"  {r['target']:>8.3f}{'  unreachable':>8}")
            continue
        print(f"  {r['target']:>8.3f}{r['threshold']:>8.3f}"
              f"{r['coverage']:>10.3f}{r['accuracy']:>10.4f}"
              f"{r['confidently_wrong_rate'] * 100:>19.2f}%")

    high = derive_threshold(vconf, vcorrect, args.high_target)
    if high is None:
        print(f'\nNo threshold reaches {args.high_target:.2f} selective accuracy '
              f'— keeping HIGH={LEGACY_TIER_HIGH}')
        high = LEGACY_TIER_HIGH
    # Local criterion with a 0.05-wide window and >=50 images per window: the
    # point where the reliability curve stops promising a better-than-coin-flip
    # top-1, which is where the UI must stop showing a single diagnosis.
    moderate = derive_threshold(vconf, vcorrect, args.moderate_target,
                                window=MODERATE_WINDOW, min_n=50)
    if moderate is None:
        print(f'\nNo band reaches {args.moderate_target:.2f} in-band accuracy '
              f'— keeping MODERATE={LEGACY_TIER_MODERATE}')
        moderate = LEGACY_TIER_MODERATE

    hacc, hn = selective_accuracy(vconf, vcorrect, high)
    print('\nDerived thresholds (calibrated val confidences):')
    print(f'  HIGH     >= {high:.3f}  -> selective accuracy {hacc:.4f} '
          f'on {hn}/{len(vconf)} images (target {args.high_target:.2f})')
    win = (vconf >= moderate) & (vconf < moderate + MODERATE_WINDOW)
    band = (vconf >= moderate) & (vconf < high)
    print(f'  MODERATE >= {moderate:.3f}  -> local accuracy in '
          f'[{moderate:.3f}, {moderate + MODERATE_WINDOW:.3f}) '
          f"{vcorrect[win].mean():.4f} on {int(win.sum())} images "
          f'(target {args.moderate_target:.2f}: more likely right than wrong)')
    print(f"           full MODERATE band accuracy {vcorrect[band].mean():.4f} "
          f'on {int(band.sum())} images')

    # --- 3. tier tables: legacy vs derived thresholds, val and test --------------
    tconf = test_cal.max(axis=1)
    tcorrect = (test_cal.argmax(axis=1) == test_y).astype(np.float64)
    t_top3 = np.any(test_cal.argsort(axis=1)[:, -3:] == test_y[:, None], axis=1)
    raw_tconf = test_probs.max(axis=1)
    raw_tcorrect = (test_probs.argmax(axis=1) == test_y).astype(np.float64)
    raw_t_top3 = np.any(test_probs.argsort(axis=1)[:, -3:] == test_y[:, None],
                        axis=1)

    tiers = {
        'test_raw_legacy_thresholds':
            tier_table(raw_tconf, raw_tcorrect, raw_t_top3,
                       LEGACY_TIER_HIGH, LEGACY_TIER_MODERATE),
        'test_calibrated_legacy_thresholds':
            tier_table(tconf, tcorrect, t_top3,
                       LEGACY_TIER_HIGH, LEGACY_TIER_MODERATE),
        'test_calibrated_new_thresholds':
            tier_table(tconf, tcorrect, t_top3, high, moderate),
        'val_calibrated_new_thresholds':
            tier_table(vconf, vcorrect, v_top3, high, moderate),
    }
    print_tiers(tiers)

    # --- 4. risk-coverage -------------------------------------------------
    vcov, vrisk, v_aurc, v_points = risk_coverage(vconf, vcorrect)
    _, _, t_aurc, t_points = risk_coverage(tconf, tcorrect)
    print(f'\nRisk-coverage: AURC val {v_aurc:.4f}  test {t_aurc:.4f} '
          '(lower is better; 0 = errors ranked last perfectly)')
    print(f"  {'coverage':>10}{'accuracy':>10}{'risk':>9}{'tau':>9}   (test)")
    for p in t_points:
        print(f"  {p['coverage']:>10.2f}{p['accuracy']:>10.4f}"
              f"{p['risk']:>9.4f}{p['confidence_threshold']:>9.4f}")

    plot_reliability(metrics['val_raw'], metrics['val_calibrated'], T)
    plot_risk_coverage(vcov, vrisk, v_aurc, high, moderate, vconf)
    print(f'\nSaved {RELIABILITY_PATH}')
    print(f'Saved {RISK_COVERAGE_PATH}')

    report = {
        'model': TFLITE_PATH,
        'n_bins': N_BINS,
        'temperature': T,
        'temperature_fit_val_nll': fit_nll,
        'targets': {'high': args.high_target, 'moderate': args.moderate_target},
        'high_target_sweep': sweep,
        'derived_thresholds': {'HIGH': high, 'MODERATE': moderate},
        'legacy_thresholds': {'HIGH': LEGACY_TIER_HIGH,
                              'MODERATE': LEGACY_TIER_MODERATE},
        'metrics': metrics,
        'tiers': tiers,
        'risk_coverage': {'val_aurc': v_aurc, 'test_aurc': t_aurc,
                          'val_points': v_points, 'test_points': t_points},
        'class_names': class_names,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Saved {os.path.abspath(REPORT_PATH)}')
    print('=' * 74)


if __name__ == '__main__':
    main()
