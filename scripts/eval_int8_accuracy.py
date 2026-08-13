"""Accuracy-only re-evaluation of the existing INT8 TFLite model.

Skips conversion and latency; prints running accuracy every 250 images so
progress is visible and partial results survive an interrupted run.

Run:  python -m scripts.eval_int8_accuracy
"""

import numpy as np
import tensorflow as tf

from src.quantize import (
    TFLITE_PATH, FLOAT32_BASELINE_ACC,
    _load_split, _read_image, _print_io_details,
    _quantize_input, _dequantize_output,
)

PROGRESS_EVERY = 250


def main():
    with open(TFLITE_PATH, 'rb') as f:
        tflite_bytes = f.read()

    # Same interpreter config as src.quantize._make_interpreter (XNNPACK
    # disabled — it can't prepare some ops in this graph), plus num_threads
    # since this is a one-off accuracy run, not a latency benchmark.
    interp = tf.lite.Interpreter(
        model_content=tflite_bytes,
        num_threads=8,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interp.allocate_tensors()
    _print_io_details(interp)
    input_detail = interp.get_input_details()[0]
    output_detail = interp.get_output_details()[0]

    test_df, label_map = _load_split('test')
    n = len(test_df)
    print(f"\nEvaluating {n} test images ...", flush=True)

    correct = 0
    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        img = _read_image(row['image_path'])
        interp.set_tensor(input_detail['index'],
                          _quantize_input(img, input_detail))
        interp.invoke()
        raw = interp.get_tensor(output_detail['index'])[0]
        scores = _dequantize_output(raw, output_detail)
        if int(scores.argmax()) == label_map[row['label']]:
            correct += 1
        if i % PROGRESS_EVERY == 0:
            print(f"  {i}/{n}  running acc = {correct / i:.4f}", flush=True)

    acc = correct / n
    print(f"\nRESULT: INT8 accuracy = {acc:.4f}  "
          f"(float32 baseline {FLOAT32_BASELINE_ACC:.4f}, "
          f"delta {acc - FLOAT32_BASELINE_ACC:+.4f})", flush=True)


if __name__ == '__main__':
    main()
