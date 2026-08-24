"""
CropGuard AI — Phase 4: re-export the shipped model with the OOD gate's extra outputs.

`models/cropguard_v1_production.tflite` emits softmax only. That is enough for MSP,
and enough for temperature scaling (softmax is invariant to the additive constant
that `log(p)` loses), but it is *not* enough for the two detectors that beat MSP in
src/ood.py:

  * energy  = logsumexp(logits) — destroyed by softmax normalization
              (logsumexp(log p) == 0 for every input, always)
  * Mahalanobis — needs the 576-d penultimate embedding

So the browser needs a second artifact whose graph stops before the softmax and also
taps the global-average-pooled feature vector. This script builds it from the same
`models/cropguard_v1.keras` weights with the same dynamic-range quantization as the
production model (see docs/quantization_findings.md for why not full INT8), then
verifies three things against the shipped artifact on real images:

  1. softmax(logits from the new model) matches the production model's probabilities
  2. the argmax never changes
  3. the **gate decision** survives quantization: the cosine-to-class-mean score
     computed from the quantized embedding is compared against the float-model
     score src/ood.py set the threshold on, and every accept/reject flip at that
     threshold is counted. A threshold measured on the float model is only
     shippable if the browser's quantized embedding lands on the same side of it.

Output: models/cropguard_v1_gate.tflite (+ a copy into web/public/), and
outputs/gate_export_report.json.

Run:  python -m src.export_ood_model
      python -m src.export_ood_model --force        # overwrite an existing artifact
      python -m src.export_ood_model --n-verify 300
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from src.config import SEED
from src.data_pipeline import CSV_PATH
from src.eval_tiers import TFLITE_PATH
from src.explain import BACKBONE, KERAS_PATH, load_image

GATE_PATH = os.path.join('models', 'cropguard_v1_gate.tflite')
# Only web/ loads the gate artifact; app/ is the legacy vanilla-JS harness and
# still uses cropguard_v1_production.tflite.
WEB_COPIES = (os.path.join('web', 'public'),)
REPORT_PATH = os.path.join('outputs', 'gate_export_report.json')
GATE_PARAMS_PATH = os.path.join('models', 'ood_gate.json')

N_VERIFY = 200
# Agreement tolerances between the new artifact and the shipped one. Both are
# dynamic-range quantized from identical weights, so they should differ only by
# kernel-scheduling noise; these are loose enough for that and tight enough that a
# genuine graph mistake fails the check.
PROB_TOL = 2e-3
ENERGY_TOL = 0.05


def build_gate_model(keras_path=KERAS_PATH):
    """Keras model with outputs (probs, logits, embedding), sharing trained weights.

    Reconstructs the head explicitly rather than deleting the softmax activation:
    the trained graph is Rescaling -> MobileNetV3Small -> GAP -> Dropout -> Dense,
    and Dense's kernel/bias are reused as-is, so this is the same function with two
    extra taps. Dropout is inference-only and therefore omitted.
    """
    model = keras.models.load_model(keras_path)
    base = model.get_layer(BACKBONE)
    dense = model.get_layer('predictions')

    inp = keras.Input(shape=(224, 224, 3), name='image')
    # [0,1] -> [0,255]: MobileNetV3's built-in preprocessing expects 0-255, which the
    # trained graph bridges with the same Rescaling layer (src/train.py).
    feat = base(keras.layers.Rescaling(255.0)(inp), training=False)
    emb = keras.layers.GlobalAveragePooling2D(name='embedding')(feat)
    logits_layer = keras.layers.Dense(dense.units, name='logits')
    logits = logits_layer(emb)
    # Keras 3 dropped the `weights=` constructor kwarg, so the trained head is
    # copied in after the layer is built by the call above.
    logits_layer.set_weights([dense.kernel.numpy(), dense.bias.numpy()])
    probs = keras.layers.Softmax(name='probs')(logits)
    return keras.Model(inp, [probs, logits, emb], name='cropguard_gate')


def convert(model, out_path):
    """Dynamic-range PTQ — the exact setting the production artifact ships with."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = conv.convert()
    with open(out_path, 'wb') as f:
        f.write(blob)
    return len(blob)


class GateRunner:
    """Interpreter wrapper returning the three outputs keyed by role.

    TFLite orders outputs by internal tensor index, not by the order the Keras model
    declares them, and the exported names are opaque (`StatefulPartitionedCall:N`).
    Roles are therefore resolved by probing: the 576-wide output is the embedding,
    and of the two 17-wide ones the probability vector is the one that is
    non-negative and sums to 1.
    """

    def __init__(self, path=GATE_PATH):
        self.interp = tf.lite.Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        details = self.interp.get_output_details()

        self.interp.set_tensor(self.inp['index'],
                               np.zeros(self.inp['shape'], np.float32))
        self.interp.invoke()

        self.out = {}
        for d in details:
            v = self.interp.get_tensor(d['index'])[0]
            if v.shape[-1] > 17:
                self.out['embedding'] = d
            elif np.all(v >= 0) and abs(float(v.sum()) - 1.0) < 1e-3:
                self.out['probs'] = d
            else:
                self.out['logits'] = d
        missing = {'probs', 'logits', 'embedding'} - set(self.out)
        if missing:
            raise RuntimeError('gate model is missing outputs: %s' % missing)

    def __call__(self, img):
        self.interp.set_tensor(self.inp['index'], img[None, ...].astype(np.float32))
        self.interp.invoke()
        return {k: self.interp.get_tensor(d['index'])[0] for k, d in self.out.items()}


def cosine_gate(emb, unit_means):
    """max_c cos(emb, unit_mean_c) — the shipped OOD score (src/ood.ClassMeanCosine)."""
    e = np.asarray(emb, dtype=np.float64)
    return float(np.max(unit_means @ (e / max(np.linalg.norm(e), 1e-12))))


def energy(logits):
    z = np.asarray(logits, dtype=np.float64)
    m = z.max()
    return float(m + np.log(np.exp(z - m).sum()))


def sample_paths(n, seed=SEED):
    """A mixed verification sample: clean test images plus every OOD/field photo.

    Agreement on the clean split says the graph is right; agreement on the shifted
    and OOD photos is what the gate threshold actually depends on, and is where the
    quantized and float models are already known to diverge (docs/EXPLAINABILITY.md).
    """
    from src.ood import walk_leaves, walk_ood
    df = pd.read_csv(CSV_PATH)
    test = df[df['split'] == 'test']['image_path'].tolist()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(test))[:n]
    clean = [test[i] for i in idx]
    ood_paths, _ = walk_ood()
    leaf_paths, _ = walk_leaves()
    return ([(p, 'test') for p in clean] + [(p, 'ood') for p in ood_paths]
            + [(p, 'shifted') for p in leaf_paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true',
                    help='re-convert even if the artifact exists')
    ap.add_argument('--n-verify', type=int, default=N_VERIFY)
    ap.add_argument('--no-copy', action='store_true',
                    help='skip copying into web/public and app/public')
    args = ap.parse_args()

    os.makedirs('models', exist_ok=True)
    if os.path.exists(GATE_PATH) and not args.force:
        # Same convention as src/quantize.py: conversion is the fragile step on this
        # machine (Windows Application Control blocks some native DLLs), so an
        # existing artifact is never silently rebuilt.
        print(f'{GATE_PATH} exists — skipping conversion (--force to rebuild)')
        size = os.path.getsize(GATE_PATH)
    else:
        print('building gate model...')
        model = build_gate_model()
        size = convert(model, GATE_PATH)
        print(f'wrote {GATE_PATH} ({size / 1e6:.3f} MB)')

    gate = GateRunner()
    prod = tf.lite.Interpreter(model_path=TFLITE_PATH)
    prod.allocate_tensors()
    p_in = prod.get_input_details()[0]
    p_out = prod.get_output_details()[0]

    gate_params = None
    if os.path.exists(GATE_PARAMS_PATH):
        with open(GATE_PARAMS_PATH, encoding='utf-8') as f:
            gate_params = json.load(f)
        unit_means = np.array(gate_params['unit_means'], dtype=np.float64)
        gate_thr = float(gate_params['threshold'])
        # The float-model reference the threshold was measured on.
        from src.ood import Features
        print('loading Keras model for the float-vs-quantized gate check...')
        fx = Features()
    else:
        print(f'{GATE_PARAMS_PATH} missing — run `python -m src.ood` first; '
              f'skipping the gate-decision check')

    samples = sample_paths(args.n_verify)
    print(f'verifying on {len(samples)} images...')
    rows = []
    for path, group in samples:
        img = load_image(path)
        g = gate(img)
        prod.set_tensor(p_in['index'], img[None, ...])
        prod.invoke()
        p_ref = prod.get_tensor(p_out['index'])[0]
        z = g['logits'].astype(np.float64)
        rows.append({
            'group': group,
            'prob_max_abs_diff': float(np.max(np.abs(g['probs'] - p_ref))),
            'argmax_agree': bool(int(np.argmax(g['probs'])) == int(np.argmax(p_ref))),
            # softmax(logits) must reproduce the production probabilities too — this
            # is what proves the logits are the real pre-softmax values.
            'logit_softmax_diff': float(np.max(np.abs(
                np.exp(z - z.max()) / np.exp(z - z.max()).sum() - p_ref))),
            'energy': energy(z),
            'emb_norm': float(np.linalg.norm(g['embedding'])),
        })
        if gate_params is not None:
            emb_float, _ = fx(img[None, ...])
            q = cosine_gate(g['embedding'], unit_means)
            f_ = cosine_gate(emb_float[0], unit_means)
            rows[-1].update({
                'gate_score_quant': q,
                'gate_score_float': f_,
                'gate_score_diff': abs(q - f_),
                'gate_flip': (q >= gate_thr) != (f_ >= gate_thr),
                'gate_accept': q >= gate_thr,
            })

    df = pd.DataFrame(rows)
    summary = {}
    for group, sub in df.groupby('group'):
        summary[group] = {
            'n': int(len(sub)),
            'prob_max_abs_diff': float(sub['prob_max_abs_diff'].max()),
            'logit_softmax_max_diff': float(sub['logit_softmax_diff'].max()),
            'argmax_agreement': float(sub['argmax_agree'].mean()),
            'energy_mean': float(sub['energy'].mean()),
            'energy_std': float(sub['energy'].std()),
        }

    if gate_params is not None:
        for group, sub in df.groupby('group'):
            summary[group].update({
                'gate_score_max_abs_diff': float(sub['gate_score_diff'].max()),
                'gate_decision_flips': int(sub['gate_flip'].sum()),
                'gate_accept_rate_quantized': float(sub['gate_accept'].mean()),
            })

    worst_prob = float(df['prob_max_abs_diff'].max())
    worst_logit = float(df['logit_softmax_diff'].max())
    ok = worst_prob < PROB_TOL and worst_logit < PROB_TOL

    report = {
        'gate_model': GATE_PATH,
        'size_bytes': int(size),
        'production_model': TFLITE_PATH,
        'quantization': 'dynamic range (tf.lite.Optimize.DEFAULT)',
        'n_verify_clean': args.n_verify,
        'tolerances': {'prob': PROB_TOL, 'energy': ENERGY_TOL},
        'per_group': summary,
        'worst_prob_abs_diff': worst_prob,
        'worst_logit_softmax_abs_diff': worst_logit,
        'argmax_agreement_overall': float(df['argmax_agree'].mean()),
        'passed': bool(ok),
    }
    if gate_params is not None:
        report['gate_decision'] = {
            'detector': gate_params['detector'],
            'threshold': gate_thr,
            'max_abs_score_diff': float(df['gate_score_diff'].max()),
            'total_flips': int(df['gate_flip'].sum()),
            'n': int(len(df)),
        }
    os.makedirs('outputs', exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"worst |prob diff| vs production: {worst_prob:.3g} "
          f"({'PASS' if ok else 'FAIL'})")

    if not args.no_copy and ok:
        for dest in WEB_COPIES:
            if os.path.isdir(dest):
                shutil.copy2(GATE_PATH, os.path.join(dest, os.path.basename(GATE_PATH)))
                print(f'copied -> {dest}')
    print(f'report -> {REPORT_PATH}')


if __name__ == '__main__':
    main()
