"""
CropGuard AI — input pipeline benchmark.

Times the training input pipeline (data loading + augmentation, NO model) over
~50 batches in three configurations to isolate where the time goes:

  (a) no augmentation at all             -> baseline I/O + decode + resize cost
  (b) augmentation, p_background_paste=0 -> adds albumentations EXCEPT GrabCut
  (c) full augmentation, normal settings -> adds the GrabCut BackgroundReplace

If (c) is much slower than (b), GrabCut is the bottleneck. If (b) is already
much slower than (a), the rest of the albumentations/numpy_function path is.

Run:  python -m src.benchmark_pipeline
"""

import time

import tensorflow as tf

from src import augmentation
from src.data_pipeline import create_dataset_csv, build_dataset, BATCH_SIZE

N_BATCHES = 50
WARMUP_BATCHES = 5


def _build_train_pipeline(df, label_map, augment, config_overrides=None):
    """
    Replicates train.get_datasets()'s training pipeline exactly:
    base (shuffled/batched/prefetched) -> unbatch -> augment -> re-batch -> prefetch.
    """
    train_base = build_dataset(df, 'train', label_map)
    if not augment:
        return train_base

    # Rebuild the module-level transform with the requested config so the
    # tf.numpy_function wrapper (which reads augmentation.train_transform)
    # picks it up.
    config = dict(augmentation.AUG_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    augmentation.train_transform = augmentation.get_training_augmentation(config)

    ds = train_base.unbatch()
    ds = augmentation.apply_augmentations_to_dataset(ds)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def benchmark(ds, n_batches=N_BATCHES, warmup=WARMUP_BATCHES):
    """Iterate warmup + n_batches batches; return (batches/sec, images/sec)."""
    it = iter(ds)
    for _ in range(warmup):
        next(it)

    start = time.perf_counter()
    n_images = 0
    for _ in range(n_batches):
        images, _ = next(it)
        n_images += int(images.shape[0])
    elapsed = time.perf_counter() - start

    return n_batches / elapsed, n_images / elapsed, elapsed


def main():
    print("Building dataset split / label map...")
    df, label_map = create_dataset_csv()
    if df is None:
        raise RuntimeError("Dataset scan failed — check DATA_DIR in src/data_pipeline.py.")

    configs = [
        ("(a) no augmentation", dict(augment=False)),
        ("(b) augmentation, p_background_paste=0",
         dict(augment=True, config_overrides={'p_background_paste': 0.0})),
        ("(c) full augmentation (normal settings)",
         dict(augment=True, config_overrides=None)),
    ]

    print(f"\nTiming {N_BATCHES} batches (batch_size={BATCH_SIZE}, "
          f"warmup={WARMUP_BATCHES}) per configuration...\n")

    results = []
    for name, kwargs in configs:
        ds = _build_train_pipeline(df, label_map, **kwargs)
        bps, ips, elapsed = benchmark(ds)
        results.append((name, bps, ips, elapsed))
        print(f"{name:<45s} {bps:6.2f} batches/s  ({ips:7.1f} img/s, "
              f"{elapsed:6.1f}s total)")

    print("\nSummary:")
    base_bps = results[0][1]
    for name, bps, _, _ in results:
        print(f"  {name:<45s} {bps:6.2f} batches/s  "
              f"({base_bps / bps:4.1f}x slower than baseline)")


if __name__ == '__main__':
    main()
