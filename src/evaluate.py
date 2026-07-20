"""
CropGuard AI — test-set evaluation of the trained model.

Single forward pass over the (unaugmented) test split from data/dataset_split.csv:

- Overall accuracy and top-3 accuracy
- Per-class precision/recall/F1 (printed table + outputs/classification_report.json)
- Labeled confusion-matrix heatmap -> outputs/confusion_matrix.png
- Failure-pattern summary: 3 lowest-F1 classes and most-confused class pairs
- Spotlight on the two smallest training classes (Potato___healthy,
  Tomato___Tomato_mosaic_virus) to check whether class weighting helped them.

Run:  python -m src.evaluate
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from src.data_pipeline import CSV_PATH, build_dataset

MODEL_PATH = os.path.join('models', 'cropguard_v1.keras')
REPORT_PATH = os.path.join('outputs', 'classification_report.json')
CM_PATH = os.path.join('outputs', 'confusion_matrix.png')

# The two smallest training classes — called out to verify class weighting worked.
SPOTLIGHT_CLASSES = ('Potato___healthy', 'Tomato___Tomato_mosaic_virus')


def load_test_data():
    """Test split + the same label map training used (sorted unique labels)."""
    df = pd.read_csv(CSV_PATH)
    label_map = {label: i for i, label in enumerate(sorted(df['label'].unique()))}
    # build_dataset only shuffles the train split, so test predictions stay
    # aligned with the CSV row order of the test split.
    test_ds = build_dataset(df, 'test', label_map)
    y_true = np.array([label_map[lbl] for lbl in df[df['split'] == 'test']['label']])
    return test_ds, y_true, label_map


def plot_confusion_matrix(cm, class_names, path=CM_PATH):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'CropGuard AI — Test Confusion Matrix ({len(class_names)} classes)')

    # Annotate non-zero cells; white text on dark diagonal cells for contrast.
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                        color='white' if cm[i, j] > threshold else 'black')

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrix heatmap to {os.path.abspath(path)}")


def evaluate():
    print("=" * 70)
    print("CropGuard AI — test-set evaluation")
    print("=" * 70)

    test_ds, y_true, label_map = load_test_data()
    class_names = list(label_map.keys())  # already in index order
    print(f"Test images: {len(y_true)} | classes: {len(class_names)}")

    model = keras.models.load_model(MODEL_PATH)
    probs = model.predict(test_ds, verbose=1)
    y_pred = probs.argmax(axis=1)

    # ---------- Overall accuracy / top-3 ----------
    accuracy = float((y_pred == y_true).mean())
    top3 = probs.argsort(axis=1)[:, -3:]
    top3_acc = float(np.any(top3 == y_true[:, None], axis=1).mean())

    # ---------- Per-class report ----------
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report['top3_accuracy'] = top3_acc
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    # ---------- Confusion matrix ----------
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)

    # ---------- Console summary ----------
    print("\nPer-class results:")
    print(report_str)
    print(f"Saved classification report to {os.path.abspath(REPORT_PATH)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Top-3 accuracy:   {top3_acc:.4f}")

    per_class_f1 = sorted(
        ((name, report[name]['f1-score']) for name in class_names),
        key=lambda x: x[1],
    )
    print("\nLowest-F1 classes (worst 3):")
    for name, f1 in per_class_f1[:3]:
        r = report[name]
        print(f"  {name:<45s} F1 {f1:.4f}  "
              f"(precision {r['precision']:.4f}, recall {r['recall']:.4f}, "
              f"support {int(r['support'])})")

    # Most-confused pairs: largest off-diagonal counts (true -> predicted).
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    flat_order = np.argsort(off_diag, axis=None)[::-1]
    print("\nMost-confused class pairs (true -> predicted, top 5 off-diagonal):")
    for idx in flat_order[:5]:
        i, j = divmod(int(idx), cm.shape[1])
        if off_diag[i, j] == 0:
            break
        pct = off_diag[i, j] / cm[i].sum() * 100
        print(f"  {class_names[i]} -> {class_names[j]}: "
              f"{off_diag[i, j]} images ({pct:.1f}% of that class's test set)")

    print("\nSmallest-training-class spotlight (did class weighting help?):")
    for name in SPOTLIGHT_CLASSES:
        r = report[name]
        print(f"  {name:<45s} precision {r['precision']:.4f}  "
              f"recall {r['recall']:.4f}  F1 {r['f1-score']:.4f}  "
              f"support {int(r['support'])}")
    print("=" * 70)


if __name__ == '__main__':
    evaluate()
