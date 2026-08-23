DATASET_DIR = "data/plantvillage_dataset/color"
SPLIT_CSV = "data/dataset_split.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# --- Calibration (derived by src/calibration.py on the val split; full numbers
# in outputs/calibration_report.json). Mirrored in web/src/lib/calibration.ts
# and web/src/lib/confidenceTier.ts — changing one without the other makes the
# shipped UI disagree with the measurement that justifies it.
TEMPERATURE = 0.8878          # < 1: the model was mildly UNDER-confident
TIER_HIGH = 0.945             # target: >= 0.99 selective accuracy
TIER_MODERATE = 0.595         # target: local accuracy >= 0.50 (better than a coin flip)

# The pre-calibration thresholds, chosen by intuition. Kept as the explicit
# before-picture that src/eval_tiers.py measures and src/calibration.py compares
# against — not for use in new code.
LEGACY_TIER_HIGH = 0.85
LEGACY_TIER_MODERATE = 0.50
