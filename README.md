# 🌿 CropGuard AI

**Plant disease diagnosis that runs entirely in your browser.**

Photograph or upload a leaf; a 1.15 MB MobileNetV3-Small classifier runs on-device
via WebAssembly and returns a diagnosis plus general treatment guidance in ~22 ms.
There is no inference server — the image never leaves the device.

Scope: **17 classes across Tomato, Potato and Corn** (PlantVillage). A deliberate MVP
narrowing, not a limitation discovered late: prove the pipeline end-to-end on three
crops before widening to all 14.

> **Scope note.** This is a plain website, **not an installable/offline PWA**. No
> service worker, no offline caching, no "Add to Home Screen." Camera capture, upload
> and inference are all still fully client-side and private; the only thing dropped is
> working at *zero* connectivity.

> **Not a substitute for an agronomist.** Treatment text is general, category-level
> guidance from university extension services and contains no chemical dosing. See
> [`MODEL_CARD.md`](MODEL_CARD.md) for intended use and failure modes.

---

## 📊 Key results

| Metric | Value | Notes |
|---|---|---|
| **Test accuracy** | **0.9465** | Float32 model, 15% stratified test split (3,625 images) |
| **Top-3 accuracy** | **0.9961** | Float32 |
| **Production accuracy** | **0.9401** | Dynamic-range quantized TFLite — the artifact the browser loads |
| **Model size** | **1.15 MB** | From 3.62 MB float32 Keras |
| **Median latency** | **22 ms** | Single image, XNNPACK on laptop CPU |

Confidence gating, measured on the test split (`python -m src.eval_tiers`):

| Tier | Rule | Coverage | Accuracy in tier |
|---|---|---|---|
| HIGH — full diagnosis + treatment | p ≥ 0.85 | 82.6% | **0.9860** |
| MODERATE — caution banner + top-3 | 0.50 ≤ p < 0.85 | 14.9% | 0.7681 |
| LOW — no diagnosis, no treatment panel | p < 0.50 | 2.5% | 0.4505 |

Accuracy falls monotonically across the tiers, which is exactly what the gate is for:
the app declines to diagnose in the band where it is barely better than a coin flip.
The residue it does *not* catch is the 1.16% of images that are confident **and**
wrong.

---

## 🏗️ Architecture

Two halves, no server between them:

1. **Python training pipeline** (`src/`) — dataset scoping and stratified splitting,
   realism-focused augmentation, two-phase transfer learning on MobileNetV3-Small,
   evaluation, and TFLite quantization.
2. **Web frontend** (`web/`) — Next.js 16 / React 19 / TypeScript / Tailwind 4, running
   the `.tflite` model through **LiteRT.js** (WebAssembly + XNNPACK) directly in the
   browser. The WASM binaries are vendored in `web/public/litert_wasm/`, never fetched
   from a CDN.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed data flow and
[`MODEL_CARD.md`](MODEL_CARD.md) for the model itself.

**Repository layout**

| Path | What it is |
|---|---|
| `src/` | Python pipeline. Run as packages from the repo root (`python -m src.train`). |
| `web/` | **Current** frontend (Next.js/TypeScript). |
| `app/` | **Legacy** vanilla HTML/JS LiteRT harness, kept as reference. New work goes in `web/`. |
| `docs/` | Architecture and the quantization investigation. |
| `scripts/investigations/` | One-off diagnostics from the quantization hunt; not part of the pipeline. |
| `models/`, `outputs/` | Committed artifacts. `data/` and `venv/` are gitignored. |

---

## 🔍 Engineering highlight: the quantization debugging story

Full INT8 post-training quantization collapsed accuracy from 0.9465 to **0.7630**.

* **Hypothesis 1 — class imbalance in calibration.** Uniform random sampling over a
  35×-imbalanced training set gave `Potato_healthy` about *one* calibration image.
  Rewrote calibration to sample 30 images per class. Result: **0.6836** — *worse*,
  decisively disproving the hypothesis.
* **Bug hunt.** Channel order (RGB confirmed via `decode_jpeg`), normalization,
  `.iloc`/`.loc` indexing, pixel ranges — all correct. No bug.
* **Hypothesis 2 — architectural sensitivity.** Weights-only (dynamic-range)
  quantization scored **0.9401**, isolating the collapse to *activation* quantization.
  MobileNetV3's hard-swish and Squeeze-and-Excitation blocks produce activation
  distributions that per-tensor INT8 clips severely — which is why the MobileNetV3
  paper ships a "minimalistic" variant with plain ReLU and no SE.
* **QAT blocked.** `tensorflow-model-optimization` 0.8.1 is tied to legacy Keras 2;
  this project is Keras 3.
* **Decision.** Shipped the dynamic-range model: 0.64 accuracy points for a
  web-friendly 1.15 MB, with float32 activations that let XNNPACK run it in 22 ms.

Full write-up: [`docs/quantization_findings.md`](docs/quantization_findings.md).

---

## 💻 Tech stack

- **Model:** TensorFlow 2.21 / Keras 3.15, MobileNetV3-Small (ImageNet-pretrained)
- **Data & augmentation:** pandas, Albumentations 2.0.8, opencv-python-headless 4.11
- **Web inference:** LiteRT.js (`@litertjs/core`) — WebAssembly / XNNPACK
- **Frontend:** Next.js 16 (App Router), React 19, TypeScript, Tailwind 4

---

## 🚀 Setup and run

### Python pipeline

```bash
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m src.data_pipeline   # scan dataset -> data/dataset_split.csv (70/15/15 stratified)
python -m src.train           # two-phase transfer learning -> models/cropguard_v1.keras
python -m src.evaluate        # test/top-3 accuracy, confusion matrix -> outputs/
python -m src.quantize        # TFLite conversion + accuracy/latency report
python -m src.eval_tiers      # confidence-tier behaviour -> outputs/tier_report.json
```

`src/quantize.py` **skips conversion if the target `.tflite` already exists** — delete
the file to force re-conversion.

> **Windows note.** TF ≥ 2.11 has no native Windows GPU support, and a local
> Application Control policy blocks some native DLLs used during TFLite *calibration* —
> so full-INT8 conversion can fail locally while training, inference and evaluation
> work fine.

### Web app

```bash
cd web
npm install
npm run dev     # http://localhost:3000
npm run build
npm run lint
```

`web/next.config.ts` holds a hardcoded LAN IP in `allowedDevOrigins` for phone testing
— update it for your network.

---

## 💊 Treatment & actionable advice

Every prediction resolves to a typed entry in `web/src/lib/diseaseInfo.ts` — 17
entries, one per class — containing the pathogen/cause, prevention bullets, a
**category-level** treatment description (deliberately no specific chemical or
dosing), and a link to a university extension source. All 17 reference links are
verified live.

The panel is gated by confidence: shown in full at HIGH, shown with a caveat at
MODERATE, and **withheld entirely at LOW**, where displaying treatment for a diagnosis
the app has just called unreliable would contradict itself.

---

## 📱 Real-world testing

**Not yet measured, and this is the project's biggest open gap.** Every training image
is a single detached leaf on a uniform studio background under controlled lighting;
field photos are not. The custom `BackgroundReplace` augmentation (GrabCut leaf
segmentation composited onto procedural soil/grass/wood/hand textures) attacks that gap
synthetically, but synthetic backgrounds are not field data.

Two anecdotal tests exist and are *not* data — one on a stock photo with no verified
ground truth (93.65%, above the HIGH gate), one on a whole-plant photo the model has no
category for (79.74% on a wrong class). Until a self-labeled phone-photo set is
collected and scored, **treat 0.9401 as a benchmark number, not a deployment number.**

---

## ⚠️ Known limitations

1. **OOD inputs can be confidently wrong.** The classifier is closed-set with no
   "not a leaf" rejection; a whole-plant photo returned 79.74% on `Corn_healthy`.
2. **Confidence tiers help but do not fully solve false confidence.** 1.16% of test
   images are HIGH-tier and wrong. Tiering catches genuine uncertainty, not confident
   errors on inputs that merely resemble the training distribution.
3. **Confidence is uncalibrated.** The 0.85 / 0.50 thresholds are validated (table
   above) but not derived — no ECE, reliability diagram or temperature scaling yet.
4. **Only 3 of 14 PlantVillage crops** — a data-filter change to widen, not an
   architecture change.
5. **Real-world accuracy is unmeasured** (see above).
6. **No baselines or ablations**, so the architecture and augmentation choices rest on
   argument rather than measurement.
7. **No explainability evidence** that the model attends to lesions rather than to
   background or leaf shape.

Roadmap for 1–7 lives in `plan.md`.

---

## 🖼️ Demo

Run `cd web && npm run dev` and drop a leaf photo onto the page. Screenshots and a
recorded walkthrough are not yet in the repo.
