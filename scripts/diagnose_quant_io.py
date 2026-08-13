"""Diagnostic: print full input/output quantization details of the INT8 model
and check whether _quantize_input can overflow int8 (wraparound, not clip)."""

import numpy as np
import pandas as pd

from src.quantize import TFLITE_PATH, _make_interpreter, _load_split, _read_image

with open(TFLITE_PATH, 'rb') as f:
    tflite_bytes = f.read()

interp = _make_interpreter(tflite_bytes)
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

print("INPUT DETAILS:")
for k, v in inp.items():
    print(f"  {k}: {v}")
print("\nOUTPUT DETAILS:")
for k, v in out.items():
    print(f"  {k}: {v}")

in_scale, in_zp = inp['quantization']
out_scale, out_zp = out['quantization']
print(f"\ninput  scale={in_scale!r} zero_point={in_zp!r}")
print(f"output scale={out_scale!r} zero_point={out_zp!r}")

# Overflow check: on a sample of test images, how many pixels does
# round(img/scale + zp) push outside [-128, 127] (which .astype(np.int8) WRAPS)?
test_df, _ = _load_split('test')
rng = np.random.default_rng(0)
sample = test_df.iloc[rng.choice(len(test_df), size=25, replace=False)]

total_pix = 0
overflow_pix = 0
for _, row in sample.iterrows():
    img = _read_image(row['image_path'])
    q = np.round(img / in_scale + in_zp)
    total_pix += q.size
    overflow_pix += int(((q < -128) | (q > 127)).sum())
    mn, mx = q.min(), q.max()
    if mx > 127 or mn < -128:
        print(f"  OVERFLOW in {row['image_path']}: q range [{mn}, {mx}]")

print(f"\nSampled {len(sample)} images, {total_pix} pixels total")
print(f"Pixels outside [-128, 127] before int8 cast: {overflow_pix} "
      f"({overflow_pix / total_pix * 100:.4f}%)")
