"""
CropGuard AI — resume Phase 2 fine-tuning from the saved checkpoint.

The original Phase 2 hit its 30-epoch cap while val_accuracy was still climbing
(best 0.9454, val_loss at its minimum on the final epoch), so this continues
fine-tuning models/cropguard_v1.keras for up to 15 more epochs:

- Same unfreeze config as train.py Phase 2 (last ~25% of backbone, BN frozen),
  applied via train.py's own unfreeze_backbone_tail().
- Adam at 5e-6 — where ReduceLROnPlateau had already brought the LR by the end
  of the original Phase 2 (not the original 1e-5 starting point).
- Same callback pattern via train.py's make_callbacks(); the checkpoint only
  overwrites models/cropguard_v1.keras if val_accuracy beats the previous best
  (initial_value_threshold), so a worse run never clobbers the saved model.
- Appends the new epochs to outputs/training_history.json (phase2 lists;
  phase1 untouched) and regenerates outputs/training_curves.png with markers
  for both the Phase 1 -> 2 boundary and the resume point.
- Finally re-evaluates whichever model is on disk against the held-out test set.

Run:  python -m src.resume_phase2
"""

import json
import os

import matplotlib.pyplot as plt
from tensorflow import keras

from src.train import (
    CURVES_PATH,
    HISTORY_PATH,
    MODEL_PATH,
    compute_class_weights,
    get_datasets,
    make_callbacks,
    unfreeze_backbone_tail,
)

RESUME_LR = 5e-6
RESUME_MAX_EPOCHS = 15


def find_backbone(model):
    """The MobileNetV3 backbone is the only nested Model inside the saved model."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise RuntimeError(f"No nested backbone Model found in {MODEL_PATH}")


def plot_curves_with_resume(hist, n_phase1, n_phase2_original):
    """Like train.plot_curves, with an extra marker where the resumed segment starts."""
    acc = hist['phase1']['accuracy'] + hist['phase2']['accuracy']
    val_acc = hist['phase1']['val_accuracy'] + hist['phase2']['val_accuracy']
    loss = hist['phase1']['loss'] + hist['phase2']['loss']
    val_loss = hist['phase1']['val_loss'] + hist['phase2']['val_loss']

    epochs = range(1, len(acc) + 1)
    boundary_p2 = n_phase1 + 0.5
    boundary_resume = n_phase1 + n_phase2_original + 0.5

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, train_vals, val_vals, title, ylabel in (
        (ax_acc, acc, val_acc, 'Accuracy', 'accuracy'),
        (ax_loss, loss, val_loss, 'Loss', 'loss'),
    ):
        ax.plot(epochs, train_vals, label=f'train {ylabel}')
        ax.plot(epochs, val_vals, label=f'val {ylabel}')
        ax.axvline(boundary_p2, ls='--', color='gray', label='Phase 1 → 2')
        ax.axvline(boundary_resume, ls=':', color='red', label='Phase 2 resumed')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle('CropGuard AI — Training Curves (Phase 1: head, Phase 2: fine-tune + resume)',
                 fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(CURVES_PATH), exist_ok=True)
    fig.savefig(CURVES_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved training curves to {os.path.abspath(CURVES_PATH)}")


def resume():
    # ---------- Previous best from the recorded history ----------
    with open(HISTORY_PATH) as f:
        hist = json.load(f)
    prev_best = max(hist['phase1']['val_accuracy'] + hist['phase2']['val_accuracy'])
    n_phase2_original = len(hist['phase2']['val_accuracy'])
    print("=" * 70)
    print(f"RESUME PHASE 2 — up to {RESUME_MAX_EPOCHS} more epochs (lr={RESUME_LR})")
    print(f"Previous best val_accuracy: {prev_best:.4f} "
          f"(checkpoint threshold — model file only overwritten beyond this)")
    print("=" * 70)

    # ---------- Data + class weights (identical to train.py) ----------
    train_ds, val_ds, test_ds, df, label_map, _ = get_datasets()
    class_weight = compute_class_weights(df, label_map)

    # ---------- Load checkpoint, re-apply Phase 2 unfreeze, recompile ----------
    model = keras.models.load_model(MODEL_PATH)
    unfreeze_backbone_tail(find_backbone(model))
    model.compile(
        optimizer=keras.optimizers.Adam(RESUME_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # ---------- Continue training ----------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=RESUME_MAX_EPOCHS,
        class_weight=class_weight,
        callbacks=make_callbacks('Phase 2 resumed', checkpoint_threshold=prev_best),
        verbose=2,
    )
    new_best = max(history.history['val_accuracy'])

    # ---------- Append to history (phase1 untouched) + redraw curves ----------
    for key, values in history.history.items():
        hist['phase2'].setdefault(key, []).extend(float(v) for v in values)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(hist, f, indent=2)
    print(f"Appended {len(history.history['val_accuracy'])} epochs to "
          f"{os.path.abspath(HISTORY_PATH)}")
    plot_curves_with_resume(hist, len(hist['phase1']['val_accuracy']), n_phase2_original)

    # ---------- Old vs new + test evaluation of whatever is on disk ----------
    print("\n" + "=" * 70)
    print(f"Previous best val_accuracy: {prev_best:.4f}")
    print(f"Resumed-run best val_accuracy: {new_best:.4f}")
    if new_best > prev_best:
        print(f"IMPROVED by {new_best - prev_best:+.4f} — checkpoint updated on disk.")
    else:
        print("No improvement — plateaued; the original checkpoint remains on disk.")
    print("Evaluating the saved model on the held-out test set...")
    best_model = keras.models.load_model(MODEL_PATH)
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    resume()
