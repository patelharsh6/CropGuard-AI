# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CropGuard AI — offline-first plant disease classification. A MobileNetV3-Small model is trained in
Python (TensorFlow/Keras 3), quantized to a 1.15 MB dynamic-range `.tflite`, and run entirely
in the browser via LiteRT.js (WebAssembly/XNNPACK). Scope: 17 classes across Tomato, Potato, Corn
(PlantVillage). No inference server exists — the model runs on-device.

## Repository layout (two halves + one legacy dir)

- `src/` — Python training/eval/quantization pipeline. Modules are run as packages from the repo
  root (`python -m src.train`), not as file paths.
- `web/` — the **current** frontend: Next.js 16 + React 19 + Tailwind 4 (TypeScript, App Router).
- `app/` — **legacy** vanilla-HTML/JS test harness for LiteRT (`npx serve`). The README's
  "cd app && npm run dev" instructions are stale; new frontend work goes in `web/`.
- `docs/ARCHITECTURE.md`, `docs/quantization_findings.md`, `docs/DOMAIN_SHIFT.md`,
  `docs/CALIBRATION.md`, `docs/EXPLAINABILITY.md`, `docs/OOD.md` — design rationale,
  the INT8 failure investigation, the robustness/domain-shift results, the calibration
  analysis that produced the shipped confidence thresholds, the Grad-CAM /
  background-leakage analysis, and the OOD gate. Read `quantization_findings.md`
  before touching `src/quantize.py`, `CALIBRATION.md` before touching any threshold or
  `web/src/lib/calibration.ts`, `EXPLAINABILITY.md` before touching `src/explain.py`
  (in particular: at the final conv layer Grad-CAM here is *provably* plain CAM, which
  is why a second 14×14 map exists — don't delete it as redundant), and `OOD.md`
  before touching `src/ood.py` or the gate threshold (in particular: the threshold is
  set on the 37 *field* photos, not on the clean val split — the textbook choice
  rejects ~95% of real leaf photos).
- `real_world_test/` — hand-taken, hand-labelled field photos for `src.eval_real_world`.
  The **class folders** must never be populated with stock or internet images (unverified
  labels, and some are PlantVillage images, which would leak the training distribution).
  Its `_ood/` subfolder is the exception and holds 97 curated Commons *negatives* for
  Phase 4 — "not a single crop leaf" is a label anyone can verify by looking, and a
  negative cannot leak the training distribution. `_ood/provenance.json` carries the
  per-image licence, attribution and review decision. See its README.
- `web_sourced_test/` — 20 hand-vetted Wikimedia Commons photos used as a Phase 1 stopgap (0.40
  top-1). Built by `scripts/harvest_commons.py`; `provenance.json` carries the per-image licence
  and attribution and must not be deleted. Keep it separate from `real_world_test/` — its labels
  are uploader captions, not verified ground truth.
- `scripts/investigations/` — one-off diagnostic scripts from the quantization debugging effort;
  not part of the pipeline.
- `data/`, `venv/` are gitignored; `models/` and `outputs/` artifacts are committed.

## Commands

Python (repo root, venv activated; there is no test suite):
```bash
python -m src.data_pipeline   # scan dataset, write data/dataset_split.csv (70/15/15 stratified)
python -m src.train           # two-phase transfer learning -> models/cropguard_v1.keras
python -m src.evaluate        # test/top-3 accuracy, confusion matrix -> outputs/
python -m src.quantize        # TFLite conversion + accuracy/latency report
python -m src.eval_tiers      # UNCALIBRATED tier baseline (0.85/0.50, raw softmax)
python -m src.calibration     # ECE/MCE, temperature scaling, derived thresholds,
                              # reliability + risk-coverage plots -> outputs/
python -m src.explain         # Grad-CAM panels -> outputs/gradcam/, leakage metrics ->
                              # outputs/gradcam_report.json (~9 min; needs the
                              # outputs/cache/*.npy written by src.calibration)
python -m src.ood             # MSP/energy/Mahalanobis/cosine OOD comparison ->
                              # outputs/ood_report.json + plots; writes models/ood_gate.json
python -m src.export_ood_model   # 3-output .tflite (probs/logits/embedding) for the gate,
                              # verified against the production artifact
python scripts/harvest_ood.py --out D --per-category 9   # harvest OOD candidates (Commons)
python scripts/curate_ood.py --staging D                 # vet -> real_world_test/_ood/
python -m src.eval_domain_shift  # synthetic corruption stress test -> outputs/domain_shift_report.json
python -m src.eval_real_world    # scores real_world_test/
python -m src.eval_real_world --calibrated   # ...the way the shipped UI now gates it
python -m src.eval_real_world --dir web_sourced_test --json outputs/web_sourced_report.json
python scripts/check_test_photos.py [--dir D] [--fix]   # validate a photo set (no TF import, fast)
python -m src.augmentation    # writes augmentation previews to outputs/
python -m src.benchmark_pipeline
```
`src/quantize.py` **skips conversion if the target `.tflite` already exists** — delete the file to
force re-conversion.

Web (`cd web`):
```bash
npm run dev     # next dev
npm run build
npm run lint    # eslint (flat config)
```

## Model/web contract (easy to break silently)

Four things must stay in sync between Python and `web/src/lib/`:

1. **Class index order.** Python derives labels as `sorted(df['label'].unique())` over PlantVillage
   folder names (`Corn___Cercospora_leaf_spot...`). `web/src/lib/constants.ts` `CLASS_NAMES` is the
   same order but with *display-sanitized* names. Editing either list without the other silently
   mislabels every prediction.
2. **Preprocessing.** `web/src/lib/preprocess.ts` reproduces `tf.image.resize(bilinear)` + `/255.0`
   using a 224×224 canvas with `imageSmoothingQuality = 'medium'`. This was verified numerically
   against the Python pipeline — do not "optimize" it.
3. **Model artifacts.** `models/cropguard_v1_production.tflite` is copied to
   `web/public/` (and `app/public/`); regenerating the model means recopying all copies.
   The web app actually **loads `cropguard_v1_gate.tflite`** — same weights and
   quantization, re-exported with logits and the 576-d embedding as extra outputs
   (verified bit-identical on probabilities). Retraining means re-running
   `src.quantize`, then `src.ood`, then `src.export_ood_model`, in that order: the
   gate's class means and threshold are functions of the weights.
4. **Accelerator.** The model is dynamic-range quantized (float32 activations) and is loaded with
   `{ accelerator: 'wasm' }` (XNNPACK). Full INT8 was abandoned — see the docs — so don't switch
   the converter to full-integer quantization expecting it to work.
5. **OOD gate parameters.** `models/ood_gate.json` (copied to `web/public/`) holds 17
   unit class-mean embeddings in `CLASS_NAMES` order plus the threshold, all derived by
   `src/ood.py`. Reordering the classes without re-running it silently scores every
   image against the wrong means. The browser resolves the model's three outputs by
   shape at runtime (`resolveSlots` in `useCropGuardModel.ts`), so TFLite output
   ordering is not part of the contract.
6. **Calibration constants.** `TEMPERATURE`, `TIER_HIGH`, `TIER_MODERATE` live in three places:
   `src/config.py`, `web/src/lib/calibration.ts`, `web/src/lib/confidenceTier.ts`. They are
   *derived* by `src/calibration.py`, so editing a value by hand makes the shipped UI disagree
   with the measurement documented in `docs/CALIBRATION.md` — re-run the script instead. The
   thresholds apply to calibrated probabilities only.

## Web frontend notes

- `web/src/lib/useCropGuardModel.ts` owns the whole LiteRT lifecycle: a **module-level init
  promise** guards against React StrictMode double-mount, the compiled model lives in a ref (not
  state), and input/output `Tensor`s are explicitly `.delete()`d to avoid WASM heap leaks.
- LiteRT WASM binaries are vendored in `web/public/litert_wasm/` and served from `/litert_wasm/`;
  they are not fetched from a CDN (offline-first).
- `web/next.config.ts` sets `logging.browserConsole: false` because the Emscripten runtime writes
  via low-level `_fd_write`, which Turbopack otherwise floods the terminal with. `allowedDevOrigins`
  holds a hardcoded LAN IP for phone testing — update it for a different network.
- `web/src/lib/oodGate.ts` runs **before** the tier system: it scores the embedding
  against the class means and, below threshold, `page.tsx` shows a "not a leaf" panel
  instead of any diagnosis. A failed `/ood_gate.json` fetch yields `ood: null`, which
  means *unknown* — the UI warns and falls back to tiers rather than assuming a leaf.
- `web/src/lib/confidenceTier.ts` gates the UI: HIGH ≥ 0.945, MODERATE ≥ 0.595, else LOW (show
  top-3 instead of a single diagnosis). `diseaseInfo.ts` maps each class to advice text. Those
  thresholds apply to **calibrated** probabilities — `web/src/lib/calibration.ts` applies
  temperature scaling (T = 0.8878) inside `useCropGuardModel.infer()` before anything reads a
  confidence. Both values are derived by `src/calibration.py` and mirrored in `src/config.py`
  (`TEMPERATURE`, `TIER_HIGH`, `TIER_MODERATE`); changing one copy without the others makes the
  UI disagree with the measurement that justifies it. See `docs/CALIBRATION.md`.
- `web/AGENTS.md` (auto-generated and re-added by `next dev`) requires reading
  `web/node_modules/next/dist/docs/` before writing Next.js code — this is Next 16, which differs
  from older conventions.

## Environment quirks

- TensorFlow ≥ 2.11 has no native Windows GPU support; training was done on Colab (T4). Local
  Windows Application Control policy also blocks some native DLLs used during TFLite calibration,
  so conversion may fail locally while inference/eval works.
- `tensorflow-model-optimization` 0.8.1 requires legacy Keras 2 and crashes on this Keras 3
  project — QAT is not available.
- Models are saved as `.keras`, not `.h5`: Keras 3 cannot round-trip a MobileNetV3 backbone from
  `.h5` (hard-swish activation fails on load).
- Albumentations API params are version-sensitive; keep to `requirements.txt` pins.
