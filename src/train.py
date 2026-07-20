"""
CropGuard AI — training script.

Two-phase transfer learning on MobileNetV3-Small (ImageNet) for PlantVillage
Tomato/Potato/Corn disease classification (17 classes):

  Phase 1  — freeze the backbone, train only the classification head (lr 1e-3).
  Phase 2  — unfreeze the last ~25% of the backbone, fine-tune (lr 1e-5).

Class imbalance is handled with 'balanced' class weights (not oversampling).
The best model (by val_accuracy, across BOTH phases) is saved to
models/cropguard_v1.keras. Training history + loss/accuracy curves are written to
outputs/ for overfitting inspection.

NOTE ON SAVE FORMAT: the requested legacy `.h5` format cannot round-trip a
MobileNetV3 backbone under Keras 3 (loading fails inside the built-in hard-swish
activation). We therefore use the native `.keras` format — the Keras 3 successor
to `.h5` — which saves/loads reliably at the same size and converts to TFLite via
`tf.lite.TFLiteConverter.from_keras_model(...)` for the later <10MB quantization.

Run:  python -m src.train
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from sklearn.utils.class_weight import compute_class_weight

# Reuse the existing data + augmentation pipelines.
from src.data_pipeline import create_dataset_csv, build_dataset, BATCH_SIZE
from src.augmentation import apply_augmentations_to_dataset

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SHAPE = (224, 224, 3)

MODEL_PATH = os.path.join('models', 'cropguard_v1.keras')
HISTORY_PATH = os.path.join('outputs', 'training_history.json')
CURVES_PATH = os.path.join('outputs', 'training_curves.png')

# Head
DROPOUT_RATE = 0.3

# Phase 1 (head only)
PHASE1_LR = 1e-3
PHASE1_MAX_EPOCHS = 50

# Phase 2 (fine-tuning)
PHASE2_LR = 1e-5
PHASE2_MAX_EPOCHS = 30
# Fraction of backbone layers (counted from the INPUT side) to keep frozen in
# phase 2. 0.75 => unfreeze the last ~25% of layers.
FINE_TUNE_FREEZE_FRACTION = 0.75

# Early stopping / LR schedule
ES_PATIENCE = 6
RLR_PATIENCE = 3


# ==========================================
# DATASETS
# ==========================================
def get_datasets():
    """
    Build train/val/test tf.data pipelines.

    - Train: base pipeline -> unbatch -> realism augmentation (per-image) -> re-batch.
      (augmentation.apply_augmentation operates on single images, so we must
       augment BEFORE batching.)
    - Val/Test: no augmentation.

    Returns (train_ds, val_ds, test_ds, df, label_map, num_classes).
    """
    df, label_map = create_dataset_csv()
    if df is None:
        raise RuntimeError(
            "Dataset CSV/scan failed — check DATA_DIR in src/data_pipeline.py."
        )
    num_classes = len(label_map)

    # Base (already shuffled + batched + prefetched) datasets.
    train_base = build_dataset(df, 'train', label_map)
    val_ds = build_dataset(df, 'val', label_map)
    test_ds = build_dataset(df, 'test', label_map)

    # Augmentation runs per-image, so unbatch -> augment -> re-batch -> prefetch.
    train_ds = train_base.unbatch()
    train_ds = apply_augmentations_to_dataset(train_ds)
    # NOTE: unbatch->map->batch makes the dataset cardinality UNKNOWN, so Keras
    # prints "Your input ran out of data; interrupting training" when it hits the
    # natural end of each fit's first epoch — verified harmless: all 529 batches
    # (16914 images) are consumed per epoch, nothing is dropped, and val/test
    # (known cardinality, 114 batches) are unaffected. No .repeat() needed.
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, df, label_map, num_classes


# ==========================================
# CLASS WEIGHTS
# ==========================================
def compute_class_weights(df, label_map):
    """
    'balanced' class weights from the TRAIN split distribution, to counter the
    heavy imbalance (e.g. Potato_healthy ~152 vs Tomato_YLCV ~5357 images).

    Returns a {class_index: weight} dict for model.fit(class_weight=...).
    """
    train_labels = [label_map[lbl] for lbl in df[df['split'] == 'train']['label']]
    classes = np.arange(len(label_map))
    weights = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weight = {int(i): float(w) for i, w in zip(classes, weights)}

    print("\nClass weights (balanced):")
    index_to_label = {v: k for k, v in label_map.items()}
    for i in classes:
        print(f"  [{i:2d}] {index_to_label[i]:<35s} weight={class_weight[i]:.3f}")
    return class_weight


# ==========================================
# MODEL
# ==========================================
def build_model(num_classes, dropout_rate=DROPOUT_RATE):
    """
    MobileNetV3-Small backbone (frozen initially) + a small classification head.

    Input images arrive in [0, 1] (from data_pipeline), but MobileNetV3's built-in
    preprocessing expects [0, 255]; a Rescaling(255.) layer bridges that.

    The head is intentionally tiny (GAP -> Dropout -> Dense) to keep the model
    small for later <10MB post-training quantization.

    Returns (model, base_model) so the caller can toggle backbone trainability.
    """
    base_model = MobileNetV3Small(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet',
        include_preprocessing=True,  # expects [0,255]; handled by Rescaling below
    )
    base_model.trainable = False  # Phase 1: fully frozen backbone

    inputs = keras.Input(shape=IMG_SHAPE)
    x = layers.Rescaling(255.0)(inputs)          # [0,1] -> [0,255] for MobileNetV3
    # training=False keeps BatchNorm in inference mode (recommended for transfer
    # learning); it stays that way in phase 2 so BN running stats aren't disturbed.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='cropguard_mobilenetv3s')
    return model, base_model


def unfreeze_backbone_tail(base_model, freeze_fraction=FINE_TUNE_FREEZE_FRACTION):
    """
    Phase 2: unfreeze the last (1 - freeze_fraction) of backbone layers.
    BatchNorm layers in the unfrozen tail are kept frozen for fine-tuning stability.
    Returns the number of trainable layers unfrozen.
    """
    base_model.trainable = True
    n_layers = len(base_model.layers)
    freeze_until = int(n_layers * freeze_fraction)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        # Keep BN frozen even in the trainable tail.
        layer.trainable = not isinstance(layer, layers.BatchNormalization)

    n_trainable = sum(l.trainable for l in base_model.layers)
    print(f"\nPhase 2: backbone has {n_layers} layers; froze first {freeze_until}, "
          f"unfroze {n_trainable} (BN kept frozen) in the tail.")
    return n_trainable


# ==========================================
# CALLBACKS
# ==========================================
class PhaseLogger(keras.callbacks.Callback):
    """Prints a clear one-line summary per epoch, tagged with the phase name."""

    def __init__(self, phase_name):
        super().__init__()
        self.phase_name = phase_name

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        lr = float(lr.numpy()) if hasattr(lr, 'numpy') else float(lr)
        print(
            f"[{self.phase_name}] epoch {epoch + 1:03d} | "
            f"loss {logs.get('loss', 0):.4f}  acc {logs.get('accuracy', 0):.4f} | "
            f"val_loss {logs.get('val_loss', 0):.4f}  "
            f"val_acc {logs.get('val_accuracy', 0):.4f} | lr {lr:.2e}"
        )


def make_callbacks(phase_name, checkpoint_threshold=None, use_reduce_lr=True):
    """Standard callback set: checkpoint (best val_acc), early stop, LR schedule, logger."""
    cbs = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            # In phase 2, only overwrite the file if we beat phase 1's best.
            initial_value_threshold=checkpoint_threshold,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        PhaseLogger(phase_name),
    ]
    if use_reduce_lr:
        cbs.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                mode='max',
                factor=0.5,
                patience=RLR_PATIENCE,
                min_lr=1e-6,
                verbose=1,
            )
        )
    return cbs


# ==========================================
# HISTORY + PLOTS
# ==========================================
def save_history(history_phase1, history_phase2):
    """Persist both phases' raw history to JSON."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    payload = {
        'phase1': history_phase1.history,
        'phase2': history_phase2.history,
    }
    with open(HISTORY_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved training history to {os.path.abspath(HISTORY_PATH)}")


def plot_curves(history_phase1, history_phase2):
    """Plot accuracy + loss across both phases, with a Phase 1|2 boundary marker."""
    h1, h2 = history_phase1.history, history_phase2.history

    acc = h1['accuracy'] + h2['accuracy']
    val_acc = h1['val_accuracy'] + h2['val_accuracy']
    loss = h1['loss'] + h2['loss']
    val_loss = h1['val_loss'] + h2['val_loss']

    epochs = range(1, len(acc) + 1)
    boundary = len(h1['accuracy']) + 0.5  # between last phase-1 epoch and first phase-2

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    ax_acc.plot(epochs, acc, label='train acc')
    ax_acc.plot(epochs, val_acc, label='val acc')
    ax_acc.axvline(boundary, ls='--', color='gray', label='Phase 1 → 2')
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(epochs, loss, label='train loss')
    ax_loss.plot(epochs, val_loss, label='val loss')
    ax_loss.axvline(boundary, ls='--', color='gray', label='Phase 1 → 2')
    ax_loss.set_title('Loss')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    fig.suptitle('CropGuard AI — Training Curves (Phase 1: head, Phase 2: fine-tune)',
                 fontsize=14)
    fig.tight_layout()

    os.makedirs(os.path.dirname(CURVES_PATH), exist_ok=True)
    fig.savefig(CURVES_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {os.path.abspath(CURVES_PATH)}")


# ==========================================
# TRAINING ORCHESTRATION
# ==========================================
def _best(history, key='val_accuracy'):
    return max(history.history[key]) if history.history.get(key) else None


def train():
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    print("=" * 70)
    print("CropGuard AI — building datasets")
    print("=" * 70)
    train_ds, val_ds, test_ds, df, label_map, num_classes = get_datasets()
    class_weight = compute_class_weights(df, label_map)

    model, base_model = build_model(num_classes)
    model.summary()

    # ---------- PHASE 1: train the head ----------
    print("\n" + "=" * 70)
    print(f"PHASE 1 — frozen backbone, head only (lr={PHASE1_LR})")
    print("=" * 70)
    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_MAX_EPOCHS,
        class_weight=class_weight,
        callbacks=make_callbacks('Phase 1'),
        verbose=2,
    )
    phase1_best = _best(history1)
    print(f"\nPhase 1 best val_accuracy: {phase1_best:.4f}")

    # ---------- PHASE 2: fine-tune the backbone tail ----------
    print("\n" + "=" * 70)
    print(f"PHASE 2 — fine-tune last ~{int((1 - FINE_TUNE_FREEZE_FRACTION) * 100)}% "
          f"of backbone (lr={PHASE2_LR})")
    print("=" * 70)
    unfreeze_backbone_tail(base_model)

    # Must recompile after changing trainability, with a much smaller LR.
    model.compile(
        optimizer=keras.optimizers.Adam(PHASE2_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_MAX_EPOCHS,
        class_weight=class_weight,
        # Only checkpoint if phase 2 beats phase 1's best val_accuracy.
        callbacks=make_callbacks('Phase 2', checkpoint_threshold=phase1_best),
        verbose=2,
    )
    phase2_best = _best(history2)
    print(f"\nPhase 2 best val_accuracy: {phase2_best:.4f}")

    # ---------- Persist history + curves ----------
    save_history(history1, history2)
    plot_curves(history1, history2)

    # ---------- Final test-set evaluation (best model on disk) ----------
    overall_best = max(v for v in [phase1_best, phase2_best] if v is not None)
    print("\n" + "=" * 70)
    print(f"Best val_accuracy across both phases: {overall_best:.4f}")
    print(f"Best model saved to: {os.path.abspath(MODEL_PATH)}")
    print("Evaluating the best saved model on the held-out test set...")
    best_model = keras.models.load_model(MODEL_PATH)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    train()
