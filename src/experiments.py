"""
CropGuard AI — Phase 5: baselines and ablations.

Every architecture and data decision in this project currently rests on an
assertion ("MobileNetV3-Small because it's small", "BackgroundReplace because
lab photos aren't farm photos", "class weights because Potato_healthy has 152
images"). This module turns each of those into a number by training the
variants under one identical, fixed budget and reporting them in a single
table.

WHY A REDUCED BUDGET
--------------------
The shipped model was trained on Colab (T4) with early stopping over 50 + 30
epochs. This machine is CPU-only (TF >= 2.11 has no native Windows GPU), so
reproducing that budget ~11 times is not on. Instead every run here gets the
SAME reduced budget:

  * a fixed stratified subsample of the train split (BUDGET['train_fraction']),
  * a fixed number of epochs, no early stopping (equal compute per run),
  * the full, untouched val and test splits for evaluation.

That makes rows comparable to *each other*, which is the point of an ablation.
It does NOT make them comparable to the shipped 0.9465 — that number came from
a longer run on all of the data, and is quoted here only as context. Any row
read as "the model scores X" instead of "variant A beats variant B by X" is a
misreading.

WHAT EACH RUN REPORTS
---------------------
clean test accuracy / top-3, ECE on val (raw and temperature-scaled, T fit on
val), AURC on test, accuracy on the hand-taken field photos in
real_world_test/ (the domain-shift number Phase 1 showed matters most),
per-class recall, parameter counts, artifact sizes and wall-clock train time.

Results accumulate in outputs/experiments.json (keyed by run id, so an
interrupted sweep resumes) and render to outputs/experiments_table.md.

Run:  python -m src.experiments --list
      python -m src.experiments                     # all pending runs
      python -m src.experiments --only mnv3s_ref,aug_none
      python -m src.experiments --quantize          # the INT8 capstone
      python -m src.experiments --table-only
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

from src import augmentation as aug
from src.augmentation import AUG_CONFIG
from src.calibration import (calibration_metrics, fit_temperature,
                             risk_coverage, softmax_T, to_logits)
from src.data_pipeline import CSV_PATH, IMG_SIZE, build_dataset
from src.eval_real_world import _read_image as read_image
from src.eval_real_world import discover as discover_field

IMG_SHAPE = (*IMG_SIZE, 3)

RUN_DIR = os.path.join('outputs', 'experiments')
MODEL_DIR = os.path.join(RUN_DIR, 'models')          # gitignored — ~6 MB per run
REPORT_PATH = os.path.join('outputs', 'experiments.json')
TABLE_PATH = os.path.join('outputs', 'experiments_table.md')
FIELD_DIR = 'real_world_test'

# The shipped model's numbers, for context only — a different (much longer)
# budget on all of the train split. Not comparable to the rows below.
SHIPPED = {'keras_test_accuracy': 0.9465, 'tflite_test_accuracy': 0.9401,
           'field_accuracy': 0.5294}

BUDGET = {
    'train_fraction': 0.40,   # stratified subsample of the train split, seed 42
    'head_epochs': 8,         # phase 1: frozen backbone, head only
    'ft_epochs': 4,           # phase 2: unfrozen tail, fine-tune
    'batch_size': 32,
    'head_lr': 1e-3,
    'ft_lr': 1e-5,
    'freeze_fraction': 0.75,  # phase 2 unfreezes the last 25% of backbone layers
    'dropout': 0.3,
    'early_stopping': False,  # deliberately off: equal compute for every row
    'seed': 42,
}


# ==========================================================================
# RUN DEFINITIONS
# ==========================================================================
# Augmentation variants are implemented by zeroing probabilities in AUG_CONFIG,
# so each variant is literally the production pipeline with transforms switched
# off — no second copy of the augmentation code to drift out of sync:
#   none      — no augmentation at all
#   standard  — flips + affine + brightness/contrast (the textbook set)
#   realism   — standard + motion blur, JPEG, colour jitter, cutout
#   full      — realism + BackgroundReplace (what ships)
AUG_OFF = {
    'standard': ('p_motion_blur', 'p_jpeg_compression', 'p_color_temp',
                 'p_cutout', 'p_background_paste'),
    'realism': ('p_background_paste',),
    'full': (),
}

RUNS = [
    # ---------------- trivial baselines (no training) ----------------
    dict(id='majority', kind='trivial', rule='global',
         label='Majority class',
         note='The floor. Anything not beating this has learned nothing.'),
    dict(id='majority_per_crop', kind='trivial', rule='per_crop',
         label='Majority class per crop (crop given)',
         note='Oracle floor: assumes the crop is known, which the model is '
              'never told. Beating this is the real bar.'),

    # ---------------- architecture baselines ----------------
    dict(id='scratch_cnn', kind='train', arch='scratch_cnn', aug='full',
         class_weights=True, two_phase=False, scratch=True,
         label='Small CNN from scratch',
         note='What transfer learning buys, in one row.'),
    dict(id='mnv2', kind='train', arch='mnv2', aug='full',
         class_weights=True, two_phase=True,
         label='MobileNetV2 (transfer)',
         note='Same budget, the older mobile backbone.'),
    dict(id='effb0', kind='train', arch='effb0', aug='full',
         class_weights=True, two_phase=True,
         label='EfficientNetB0 (transfer)',
         note='Same budget, ~6x the FLOPs. Stands in for EfficientNet-Lite0, '
              'which keras.applications does not ship.'),

    # ---------------- the reference configuration ----------------
    dict(id='mnv3s_ref', kind='train', arch='mnv3s', aug='full',
         class_weights=True, two_phase=True,
         label='MobileNetV3-Small (reference = the shipped recipe)',
         note='Every ablation below changes exactly one thing from this row.'),

    # ---------------- ablations ----------------
    dict(id='aug_none', kind='train', arch='mnv3s', aug='none',
         class_weights=True, two_phase=True,
         label='...no augmentation',
         note='Isolates the whole augmentation stack.'),
    dict(id='aug_standard', kind='train', arch='mnv3s', aug='standard',
         class_weights=True, two_phase=True,
         label='...standard augmentation only',
         note='Flips/affine/brightness — none of the realism transforms.'),
    dict(id='aug_realism', kind='train', arch='mnv3s', aug='realism',
         class_weights=True, two_phase=True,
         label='...realism augmentation, no BackgroundReplace',
         note='Against the reference row this isolates BackgroundReplace alone '
              '— the transform Phase 3 and Phase 4 both lean on.'),
    dict(id='no_class_weights', kind='train', arch='mnv3s', aug='full',
         class_weights=False, two_phase=True,
         label='...without class weights',
         note='Read this one in the per-class recall table: Potato_healthy has '
              '152 train images against Tomato_YLCV\'s 5357.'),
    dict(id='frozen_only', kind='train', arch='mnv3s', aug='full',
         class_weights=True, two_phase=False,
         label='...frozen backbone only (no fine-tuning)',
         note='Same epoch count, all of it head-only: what phase 2 buys.'),

    # ---------------- quantization capstone ----------------
    dict(id='mnv3s_minimalistic', kind='train', arch='mnv3s_min', aug='full',
         class_weights=True, two_phase=True,
         label='MobileNetV3-Small "minimalistic" (no SE, ReLU)',
         note='docs/quantization_findings.md predicted this variant survives '
              'full INT8 where the standard one collapsed. --quantize tests it.'),
]

RUNS_BY_ID = {r['id']: r for r in RUNS}
# The two rows the INT8 capstone compares: same budget, same data, architecture
# is the only difference — which is what makes the comparison mean anything.
QUANT_PAIR = ('mnv3s_ref', 'mnv3s_minimalistic')


# ==========================================================================
# DATA
# ==========================================================================
def load_splits():
    """(df, label_map) with the train split stratified-subsampled to budget."""
    df = pd.read_csv(CSV_PATH)
    label_map = {lbl: i for i, lbl in enumerate(sorted(df['label'].unique()))}

    frac = BUDGET['train_fraction']
    train = df[df['split'] == 'train']
    if frac < 1.0:
        train = pd.concat([
            g.sample(frac=frac, random_state=BUDGET['seed'])
            for _, g in train.groupby('label')
        ])
    rest = df[df['split'] != 'train']
    return pd.concat([train, rest], ignore_index=True), label_map


def build_train_ds(df, label_map, aug_variant):
    """Train pipeline for one augmentation variant.

    'none' skips the albumentations map entirely; the others rebind the
    module-level transform with the excluded transforms' probabilities at 0.
    """
    base = build_dataset(df, 'train', label_map, batch_size=BUDGET['batch_size'])
    if aug_variant == 'none':
        return base

    cfg = dict(AUG_CONFIG)
    for key in AUG_OFF[aug_variant]:
        cfg[key] = 0.0
    aug.set_training_transform(cfg)

    ds = base.unbatch()
    ds = aug.apply_augmentations_to_dataset(ds)
    # unbatch -> map -> batch leaves cardinality unknown, so Keras prints "input
    # ran out of data" at each epoch boundary. Verified harmless in src/train.py.
    return ds.batch(BUDGET['batch_size']).prefetch(tf.data.AUTOTUNE)


def class_weights_for(df, label_map):
    y = [label_map[l] for l in df[df['split'] == 'train']['label']]
    classes = np.arange(len(label_map))
    w = compute_class_weight('balanced', classes=classes, y=y)
    return {int(i): float(v) for i, v in zip(classes, w)}


# ==========================================================================
# MODELS
# ==========================================================================
def _head(x, num_classes, dropout):
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(num_classes, activation='softmax', name='predictions')(x)


def build_arch(arch, num_classes, dropout=None):
    """(model, base_model or None). Inputs are always [0,1] float32, matching
    src/data_pipeline; each backbone's own expected range is bridged here."""
    dropout = BUDGET['dropout'] if dropout is None else dropout
    inputs = keras.Input(shape=IMG_SHAPE)

    if arch == 'scratch_cnn':
        # Deliberately small (~0.3M params) — a plausible "just train a CNN"
        # first attempt, not a crippled strawman.
        x = inputs
        for filters in (32, 64, 128, 128):
            x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D()(x)
        outputs = _head(x, num_classes, dropout)
        return keras.Model(inputs, outputs, name='scratch_cnn'), None

    if arch in ('mnv3s', 'mnv3s_min'):
        base = keras.applications.MobileNetV3Small(
            input_shape=IMG_SHAPE, include_top=False, weights='imagenet',
            include_preprocessing=True,                  # expects [0,255]
            minimalistic=(arch == 'mnv3s_min'),
        )
        x = layers.Rescaling(255.0)(inputs)
    elif arch == 'mnv2':
        base = keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        x = layers.Rescaling(2.0, offset=-1.0)(inputs)   # [0,1] -> [-1,1]
    elif arch == 'effb0':
        base = keras.applications.EfficientNetB0(
            input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
        x = layers.Rescaling(255.0)(inputs)             # B0 normalizes internally
    else:
        raise ValueError(f'unknown arch: {arch}')

    base.trainable = False
    x = base(x, training=False)                         # BN in inference mode
    outputs = _head(x, num_classes, dropout)
    return keras.Model(inputs, outputs, name=arch), base


def unfreeze_tail(base):
    """Unfreeze the last (1 - freeze_fraction) of backbone layers, BN excepted."""
    base.trainable = True
    cut = int(len(base.layers) * BUDGET['freeze_fraction'])
    for layer in base.layers[:cut]:
        layer.trainable = False
    for layer in base.layers[cut:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)
    return sum(l.trainable for l in base.layers)


# ==========================================================================
# EVALUATION
# ==========================================================================
def _labels(df, split, label_map):
    return np.array([label_map[l] for l in df[df['split'] == split]['label']])


def field_metrics(model, label_map):
    """Accuracy on the hand-taken field photos — the domain-shift number."""
    if not os.path.isdir(FIELD_DIR):
        return None
    samples, _ = discover_field(FIELD_DIR, label_map)
    if not samples:
        return None
    x = np.stack([read_image(s['path']) for s in samples])
    y = np.array([s['y_true'] for s in samples])
    p = model.predict(x, batch_size=BUDGET['batch_size'], verbose=0)
    order = p.argsort(axis=1)[:, ::-1]
    return {
        'n': int(len(y)),
        'accuracy': float((p.argmax(axis=1) == y).mean()),
        'top3_accuracy': float(np.any(order[:, :3] == y[:, None], axis=1).mean()),
        'mean_confidence': float(p.max(axis=1).mean()),
    }


def per_class_recall(probs, y_true, class_names):
    top1 = probs.argmax(axis=1)
    out = {}
    for i, name in enumerate(class_names):
        m = y_true == i
        out[name] = {'n': int(m.sum()),
                     'recall': float((top1[m] == i).mean()) if m.any() else None}
    return out


def evaluate_model(model, df, label_map):
    """The metric block every trained row reports."""
    class_names = list(label_map.keys())
    val_ds = build_dataset(df, 'val', label_map, batch_size=BUDGET['batch_size'])
    test_ds = build_dataset(df, 'test', label_map, batch_size=BUDGET['batch_size'])

    p_val = model.predict(val_ds, verbose=0)
    p_test = model.predict(test_ds, verbose=0)
    y_val = _labels(df, 'val', label_map)
    y_test = _labels(df, 'test', label_map)

    val_m = calibration_metrics(p_val, y_val)
    test_m = calibration_metrics(p_test, y_test)

    # Temperature is fit on val only (test never sees a fitted parameter), then
    # applied unchanged to test — the same protocol as src/calibration.py.
    temperature, _ = fit_temperature(p_val, y_val)
    val_cal = calibration_metrics(softmax_T(to_logits(p_val), temperature), y_val)
    test_cal = calibration_metrics(softmax_T(to_logits(p_test), temperature),
                                   y_test)

    conf = p_test.max(axis=1)
    correct = (p_test.argmax(axis=1) == y_test).astype(np.float64)
    _, _, aurc, _ = risk_coverage(conf, correct)

    order = p_test.argsort(axis=1)[:, ::-1]
    top3 = float(np.any(order[:, :3] == y_test[:, None], axis=1).mean())

    return {
        'val_accuracy': val_m['accuracy'],
        'test_accuracy': test_m['accuracy'],
        'test_top3_accuracy': top3,
        'test_mean_confidence': test_m['mean_confidence'],
        'val_ece': val_m['ece'],
        'val_ece_temp_scaled': val_cal['ece'],
        'test_ece': test_m['ece'],
        'test_ece_temp_scaled': test_cal['ece'],
        'temperature': temperature,
        'test_aurc': aurc,
        'test_brier': test_m['brier'],
        'test_nll': test_m['nll'],
        'field': field_metrics(model, label_map),
        'per_class_recall': per_class_recall(p_test, y_test, class_names),
    }


# ==========================================================================
# TRIVIAL BASELINES
# ==========================================================================
def _crop_of(label):
    return label.split('___')[0]


def run_trivial(cfg, df, label_map):
    """No-training floors, predicted from the TRAIN distribution only."""
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    if cfg['rule'] == 'global':
        pred = train['label'].value_counts().idxmax()
        preds = np.array([pred] * len(test))
        detail = {'prediction': pred}
    else:
        by_crop = (train.groupby(train['label'].map(_crop_of))['label']
                        .agg(lambda s: s.value_counts().idxmax()).to_dict())
        preds = test['label'].map(_crop_of).map(by_crop).values
        detail = {'prediction_per_crop': by_crop}

    return {
        'test_accuracy': float((preds == test['label'].values).mean()),
        'test_top3_accuracy': None,
        'detail': detail,
        'n_test': int(len(test)),
    }


# ==========================================================================
# TRAINING
# ==========================================================================
def train_one(cfg, df, label_map):
    keras.utils.set_random_seed(BUDGET['seed'])
    num_classes = len(label_map)

    train_ds = build_train_ds(df, label_map, cfg['aug'])
    val_ds = build_dataset(df, 'val', label_map, batch_size=BUDGET['batch_size'])
    cw = class_weights_for(df, label_map) if cfg['class_weights'] else None

    model, base = build_arch(cfg['arch'], num_classes)
    total_epochs = BUDGET['head_epochs'] + BUDGET['ft_epochs']
    history = {}
    t0 = time.perf_counter()

    if cfg.get('scratch'):
        # No backbone to freeze: spend the whole budget training everything.
        model.compile(optimizer=keras.optimizers.Adam(BUDGET['head_lr']),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        h = model.fit(train_ds, validation_data=val_ds, epochs=total_epochs,
                      class_weight=cw, verbose=2)
        history['scratch'] = h.history
    else:
        # Phase 1 — head only. The no-fine-tuning ablation gets the FULL epoch
        # budget here, so it costs the same compute as a two-phase run.
        p1_epochs = BUDGET['head_epochs'] if cfg['two_phase'] else total_epochs
        model.compile(optimizer=keras.optimizers.Adam(BUDGET['head_lr']),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=p1_epochs,
                       class_weight=cw, verbose=2)
        history['phase1'] = h1.history

        if cfg['two_phase']:
            n_trainable = unfreeze_tail(base)
            print(f'  phase 2: unfroze {n_trainable}/{len(base.layers)} '
                  f'backbone layers (BN kept frozen)')
            model.compile(optimizer=keras.optimizers.Adam(BUDGET['ft_lr']),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            h2 = model.fit(train_ds, validation_data=val_ds,
                           epochs=BUDGET['ft_epochs'], class_weight=cw,
                           verbose=2)
            history['phase2'] = h2.history

    train_seconds = time.perf_counter() - t0

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{cfg['id']}.keras")
    model.save(model_path)

    metrics = evaluate_model(model, df, label_map)
    metrics.update({
        'train_seconds': train_seconds,
        'epochs': total_epochs,
        'params_total': int(model.count_params()),
        'params_trainable': int(sum(int(np.prod(v.shape))
                                    for v in model.trainable_variables)),
        'keras_size_mb': os.path.getsize(model_path) / 1e6,
        'dynrange_tflite_size_mb': dynrange_size_mb(model),
        'model_path': model_path.replace('\\', '/'),
        'history': history,
    })
    return metrics


def dynrange_size_mb(model):
    """Size of the dynamic-range .tflite — the deployment artifact's size."""
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        return len(conv.convert()) / 1e6
    except Exception as exc:                 # a size column is not worth a crash
        print(f'  dynamic-range conversion failed: {exc}')
        return None


# ==========================================================================
# QUANTIZATION CAPSTONE
# ==========================================================================
def _tflite_accuracy(tflite_bytes, images, y_true, make_interp,
                     quantize_input=None, dequantize_output=None):
    interp = make_interp(tflite_bytes)
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    correct = 0
    for i in range(len(images)):
        img = images[i]
        x = (quantize_input(img, inp_d) if quantize_input
             else img[np.newaxis].astype(np.float32))
        interp.set_tensor(inp_d['index'], x)
        interp.invoke()
        raw = interp.get_tensor(out_d['index'])[0]
        scores = dequantize_output(raw, out_d) if dequantize_output else raw
        correct += int(int(scores.argmax()) == int(y_true[i]))
    return correct / len(images)


def quantize_capstone(df, label_map, results):
    """Full INT8 PTQ on the standard vs 'minimalistic' MobileNetV3-Small.

    docs/quantization_findings.md traced the production model's INT8 collapse
    (0.9401 -> 0.7630) to hard-swish and squeeze-excite. 'minimalistic' removes
    both. Both models here were trained under the same budget on the same data,
    so architecture is the only difference between their two INT8 deltas — which
    the original investigation could not say, having only one model.
    """
    from src.quantize import (_dequantize_output, _make_interpreter,
                              _quantize_input, representative_dataset)

    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    y_test = np.array([label_map[l] for l in test_df['label']])
    images = None                                   # decoded once, then reused

    out = {}
    for run_id in QUANT_PAIR:
        entry = results.get(run_id)
        if not entry or not os.path.exists(entry['metrics'].get('model_path', '')):
            print(f'  {run_id}: no trained model on disk — run it first, skipping')
            continue
        model = keras.models.load_model(entry['metrics']['model_path'])

        if images is None:
            print(f'  decoding {len(test_df)} test images once')
            images = np.stack([read_image(p) for p in test_df['image_path']])

        float_acc = float((model.predict(images, batch_size=32, verbose=0)
                           .argmax(axis=1) == y_test).mean())
        row = {'float32_test_accuracy': float_acc}

        # --- dynamic range (what ships): int8 weights, float32 activations
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        dyn = conv.convert()
        dyn_acc = _tflite_accuracy(dyn, images, y_test, _make_interpreter)
        row['dynrange'] = {'size_mb': len(dyn) / 1e6, 'test_accuracy': dyn_acc,
                           'delta_vs_float32': dyn_acc - float_acc}

        # --- full INT8: weights AND activations int8, calibrated on train
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda: representative_dataset(train_df)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
            i8 = conv.convert()
            acc = _tflite_accuracy(i8, images, y_test, _make_interpreter,
                                   quantize_input=_quantize_input,
                                   dequantize_output=_dequantize_output)
            path = os.path.join(MODEL_DIR, f'{run_id}_int8.tflite')
            with open(path, 'wb') as f:
                f.write(i8)
            row['int8'] = {'size_mb': len(i8) / 1e6, 'test_accuracy': acc,
                           'delta_vs_float32': acc - float_acc,
                           'path': path.replace('\\', '/')}
        except Exception as exc:
            row['int8'] = {'error': str(exc)}
            print(f'  {run_id}: INT8 conversion failed: {exc}')

        out[run_id] = row
        print(f"  {run_id}: float32 {float_acc:.4f} | dynrange {dyn_acc:.4f} | "
              f"int8 {row['int8'].get('test_accuracy')}")
        keras.backend.clear_session()

    return out


# ==========================================================================
# PERSISTENCE + TABLE
# ==========================================================================
def config_hash(cfg):
    payload = {'run': {k: v for k, v in cfg.items() if k not in ('note', 'label')},
               'budget': BUDGET}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def load_results():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            return json.load(f)
    return {'budget': BUDGET, 'shipped_for_context': SHIPPED, 'runs': {}}


def save_results(state):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    state['budget'] = BUDGET
    state['shipped_for_context'] = SHIPPED
    with open(REPORT_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def _f(value, nd=4):
    return '—' if value is None else f'{value:.{nd}f}'


def render_table(state):
    runs = state.get('runs', {})
    quant = state.get('quantization', {})
    lines = []
    add = lines.append

    add('# Phase 5 — baselines and ablations\n')
    add('Generated by `python -m src.experiments`. '
        'See `docs/EXPERIMENTS.md` for what each row means.\n')
    add('**Shared budget** — every trained row below got exactly this, with no '
        'early stopping:\n')
    add(f"- {int(BUDGET['train_fraction'] * 100)}% stratified subsample of the "
        f"train split (seed {BUDGET['seed']}); the full val/test splits for "
        'evaluation')
    add(f"- {BUDGET['head_epochs']} head epochs (lr {BUDGET['head_lr']:g}) + "
        f"{BUDGET['ft_epochs']} fine-tune epochs (lr {BUDGET['ft_lr']:g}), "
        f"batch {BUDGET['batch_size']}")
    add('- CPU-only (no TF GPU on native Windows), float32 Keras models\n')
    add('The shipped model scored **0.9465** test / **0.5294** field on a much '
        'longer run over all of the train split. Rows here are comparable to '
        '*each other*, not to that.\n')

    add('| run | test acc | top-3 | field acc | val ECE (raw → T) | AURC | '
        'params | .tflite MB | train s |')
    add('|---|---|---|---|---|---|---|---|---|')
    for cfg in RUNS:
        entry = runs.get(cfg['id'])
        if not entry:
            add(f"| {cfg['label']} | _not run_ | | | | | | | |")
            continue
        m = entry['metrics']
        if cfg['kind'] == 'trivial':
            add(f"| {cfg['label']} | {_f(m['test_accuracy'])} | — | — | — | — | "
                '0 | — | 0 |')
            continue
        field = m.get('field') or {}
        add(f"| {cfg['label']} | {_f(m['test_accuracy'])} | "
            f"{_f(m['test_top3_accuracy'])} | {_f(field.get('accuracy'))} | "
            f"{_f(m['val_ece'])} → {_f(m['val_ece_temp_scaled'])} | "
            f"{_f(m['test_aurc'])} | {m['params_total'] / 1e6:.2f}M | "
            f"{_f(m.get('dynrange_tflite_size_mb'), 2)} | "
            f"{m['train_seconds']:.0f} |")

    field_n = next((r['metrics']['field']['n'] for r in runs.values()
                    if (r['metrics'].get('field') or {}).get('n')), None)
    if field_n:
        add(f'\nField accuracy is on the {field_n} hand-taken photos in '
            '`real_world_test/` — small n, so treat differences under ~3 photos '
            '(0.08) as noise.\n')

    add('\n## What each row is for\n')
    for cfg in RUNS:
        add(f"- **{cfg['label']}** — {cfg['note']}")

    if quant:
        add('\n## Quantization capstone — full INT8 by architecture\n')
        add('| model | float32 | dynamic-range | Δ | full INT8 | Δ |')
        add('|---|---|---|---|---|---|')
        for run_id, row in quant.items():
            dyn = row.get('dynrange', {})
            i8 = row.get('int8', {})
            i8_acc = ('conversion failed' if 'error' in i8
                      else _f(i8.get('test_accuracy')))
            add(f"| {RUNS_BY_ID[run_id]['label']} | "
                f"{_f(row.get('float32_test_accuracy'))} | "
                f"{_f(dyn.get('test_accuracy'))} | "
                f"{_f(dyn.get('delta_vs_float32'))} | {i8_acc} | "
                f"{_f(i8.get('delta_vs_float32'))} |")

    trained = [r['id'] for r in RUNS if r['kind'] == 'train' and r['id'] in runs]
    if trained:
        add('\n## Per-class recall (test split)\n')
        add('The class-weight ablation is decided here, not in the accuracy '
            'column.\n')
        classes = list(runs[trained[0]]['metrics']['per_class_recall'].keys())
        add('| class | n | ' + ' | '.join(trained) + ' |')
        add('|---|---|' + '---|' * len(trained))
        for cls in classes:
            n = runs[trained[0]]['metrics']['per_class_recall'][cls]['n']
            cells = [_f(runs[i]['metrics']['per_class_recall'][cls]['recall'], 3)
                     for i in trained]
            add(f'| {cls} | {n} | ' + ' | '.join(cells) + ' |')

    text = '\n'.join(lines) + '\n'
    with open(TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Wrote {TABLE_PATH}')
    return text


# ==========================================================================
# CLI
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description='Phase 5 baselines and ablations.')
    ap.add_argument('--only', help='comma-separated run ids')
    ap.add_argument('--force', action='store_true',
                    help='re-run even if results already exist')
    ap.add_argument('--table-only', action='store_true',
                    help='re-render the markdown table from experiments.json')
    ap.add_argument('--quantize', action='store_true',
                    help='run the INT8 capstone on already-trained models')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    state = load_results()

    if args.list:
        for cfg in RUNS:
            mark = '[done]' if cfg['id'] in state.get('runs', {}) else '[    ]'
            print(f"{mark} {cfg['id']:<20s} {cfg['label']}")
        return

    if args.table_only:
        render_table(state)
        return

    os.makedirs(RUN_DIR, exist_ok=True)
    df, label_map = load_splits()
    n_train = int((df['split'] == 'train').sum())
    print(f"Budget: {n_train} train images "
          f"({int(BUDGET['train_fraction'] * 100)}% subsample), "
          f"{BUDGET['head_epochs']}+{BUDGET['ft_epochs']} epochs")

    if args.quantize:
        state['quantization'] = quantize_capstone(df, label_map,
                                                 state.get('runs', {}))
        save_results(state)
        render_table(state)
        return

    wanted = [RUNS_BY_ID[i] for i in args.only.split(',')] if args.only else RUNS
    for cfg in wanted:
        if cfg['id'] in state['runs'] and not args.force:
            print(f"[skip] {cfg['id']} — already in {REPORT_PATH}")
            continue
        print('\n' + '=' * 70)
        print(f"[run] {cfg['id']} — {cfg['label']}")
        print('=' * 70)
        if cfg['kind'] == 'trivial':
            metrics = run_trivial(cfg, df, label_map)
        else:
            metrics = train_one(cfg, df, label_map)
            keras.backend.clear_session()
        state['runs'][cfg['id']] = {'config': cfg,
                                    'config_hash': config_hash(cfg),
                                    'metrics': metrics}
        save_results(state)                        # crash-safe: written per run
        field = (metrics.get('field') or {}).get('accuracy')
        print(f"[done] {cfg['id']}: test {_f(metrics.get('test_accuracy'))} "
              f"field {_f(field)}")

    render_table(state)


if __name__ == '__main__':
    main()
