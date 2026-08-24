# 🌿 CropGuard AI

**Plant disease diagnosis that runs entirely in your browser.**

Photograph or upload a leaf; a 1.15 MB MobileNetV3-Small classifier runs on-device via
WebAssembly and returns a diagnosis plus general treatment guidance in ~22 ms. There is
no inference server — the image never leaves the device.

Scope: **17 classes across Tomato, Potato and Corn** (PlantVillage). A deliberate MVP
narrowing, not a limitation discovered late: prove the pipeline end-to-end on three
crops before widening to all 14.

> **Scope note.** This is a plain website, **not an installable/offline PWA**. No service
> worker, no offline caching, no "Add to Home Screen." Camera capture, upload and
> inference are all still fully client-side and private; the only thing dropped is
> working at *zero* connectivity.

> **Not a substitute for an agronomist.** Treatment text is general, category-level
> guidance from university extension services and contains no chemical dosing. See
> [`MODEL_CARD.md`](MODEL_CARD.md) for intended use and failure modes.

---

## Contents

- [Headline numbers](#-headline-numbers)
- [How the model was built, and what each step bought](#-how-the-model-was-built-and-what-each-step-bought)
- [Engineering highlight: the quantization debugging story](#-engineering-highlight-the-quantization-debugging-story)
- [Confidence tiers and calibration: turning a guess into a measurement](#-confidence-tiers-and-calibration-turning-a-guess-into-a-measurement)
- [Domain shift: what happens outside the lab](#-domain-shift-what-happens-outside-the-lab)
- [Explainability: what the model actually looks at](#-explainability-what-the-model-actually-looks-at)
- [Architecture](#-architecture)
- [Setup and run](#-setup-and-run)
- [Treatment & actionable advice](#-treatment--actionable-advice)
- [Known limitations](#-known-limitations)
- [Project status by phase](#-project-status-by-phase)

---

## 📊 Headline numbers

| Metric | Value | Notes |
|---|---|---|
| **Test accuracy** | **0.9465** | Float32 model, 15% stratified test split (3,625 images) |
| **Top-3 accuracy** | **0.9961** | Float32 |
| **Production accuracy** | **0.9401** | Dynamic-range quantized TFLite — the artifact the browser loads |
| **Model size** | **1.15 MB** | From 3.62 MB float32 Keras |
| **Median latency** | **22 ms** | Single image, XNNPACK on laptop CPU |
| **Out-of-distribution accuracy** | **0.5294** | 17-image all-classes test set, CI [0.31, 0.74] |
| **Confidently wrong, OOD** | **5.9%** | was 23.5% before calibration; 0.52% on the benchmark |
| **Calibration (ECE)** | **0.0075** | Test split, after temperature scaling (T = 0.8878); 0.0187 raw |

Those last three rows are the important ones, and they are deliberately not hidden at the
bottom of the page. **A 94% benchmark model scores 53% on images from outside its training
distribution.** One image in four used to come back confidently wrong there; measured
calibration and data-derived thresholds cut that to roughly one in seventeen — on a
17-image set, so treat it as a direction, not a guarantee. Most of this README is about how
each number was obtained and how much weight it can carry.

---

## 🏗 How the model was built, and what each step bought

### Data

**PlantVillage**, `color` (unsegmented) variant — deliberately *not* the
background-removed variant, since real photographs have natural backgrounds.

- **Split:** stratified 70/15/15 → **16,914 / 3,625 / 3,625**, written once to
  `data/dataset_split.csv` so it is fixed across every run and every script.
- **Imbalance:** ~35×, from `Potato___healthy` (152 train images) to
  `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (5,357).

### Augmentation — realism, not just regularisation

`src/augmentation.py` applies affine rotate/scale, flips, brightness/contrast and
colour-temperature jitter, plus three "realism" transforms aimed at the lab→field gap:
motion blur, JPEG artifacts and `CoarseDropout`. On top of those sits a custom
**`BackgroundReplace`**: GrabCut leaf segmentation at 128 px (~3× faster than at full
resolution), a feathered alpha blend, and compositing onto procedurally generated
soil / grass / wood / hand textures, at p=0.4. Textures are generated rather than
downloaded so the pipeline stays self-contained.

Applied through `unbatch()` → `tf.numpy_function` → `batch()`, because Albumentations is
NumPy-based and cannot run inside a batched `tf.data` graph.

### Training — two-phase transfer learning

MobileNetV3-Small (`include_top=False`, ImageNet weights) → `GlobalAveragePooling2D` →
`Dropout(0.3)` → `Dense(17, softmax)`. 948,929 parameters.

| Phase | What | LR | Val accuracy |
|---|---|---|---|
| 1 | Frozen backbone, head only | Adam 1e-3 | 0.8908 |
| 2 | Unfroze last ~25% (first 117/157 layers frozen, BatchNorm frozen throughout) | Adam 1e-5 | 0.9454 |
| 2b | Resumed 15 more epochs at a lower rate | Adam 5e-6 | **0.9481** |

EarlyStopping and ReduceLROnPlateau both monitored `val_accuracy`. BatchNorm stays frozen
through fine-tuning — unfreezing it with a small batch and a tiny learning rate is the
classic way to quietly destroy a transferred backbone.

**Class imbalance** was handled with `sklearn.compute_class_weight('balanced', ...)`
rather than resampling. It worked, and the evidence is per-class recall rather than
overall accuracy:

- `Potato___healthy` — 152 training images → precision 0.88, **recall 0.96**
- `Tomato___mosaic_virus` — 373 training images → **recall 1.00**

Neither rare class sits at the bottom of the per-class F1 ranking, and macro-averaged F1
(0.9402) sits close to weighted (0.9467) — which is what tells you the model is not
simply riding the majority classes.

### Where the model is weakest, and why it makes sense

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| Tomato Target Spot | 0.847 | 0.895 | 0.870 | 210 |
| Tomato Early blight | 0.872 | 0.867 | 0.870 | 150 |
| Tomato Septoria leaf spot | 0.925 | 0.883 | 0.904 | 266 |
| Potato healthy | 0.880 | 0.957 | 0.917 | 23 |
| Corn Gray leaf spot | 0.872 | 0.974 | 0.920 | 77 |

The confusion is **structured, not noise**:

- A tomato brown-lesion cluster: Early blight ↔ Target Spot ↔ Septoria ↔ Spider mites.
  Extension guides separate these partly by lesion halo and by how symptoms distribute
  across the whole plant — information a cropped single-leaf image simply does not
  contain.
- One corn pair: Northern Leaf Blight → Gray leaf spot.

Full metrics: `outputs/classification_report.json`; matrix: `outputs/confusion_matrix.png`.

---

## 🔍 Engineering highlight: the quantization debugging story

Full INT8 post-training quantization collapsed accuracy from 0.9465 to **0.7630**. The
hunt for why is the most transferable part of this project.

| Step | Hypothesis / action | Result |
|---|---|---|
| 1 | Full INT8 PTQ, default uniform calibration | **0.7630** |
| 2 | *Rare classes are starved during calibration* — `Potato_healthy` got ~1 of 400 calibration images. Rewrote to stratified sampling, 30/class | **0.6836** — *worse*, hypothesis disproved |
| 3 | Bug hunt: channel order (RGB confirmed via `decode_jpeg`), normalization, `.iloc`/`.loc` indexing, pixel ranges | No bug found |
| 4 | *Isolate weights from activations* — dynamic-range quantization (INT8 weights, float32 activations, no calibration) | **0.9401** |
| 5 | Conclusion | Weights quantize fine. **Activation** quantization is the whole problem |
| 6 | Root cause | MobileNetV3's hard-swish and Squeeze-and-Excitation blocks produce activation distributions that per-tensor INT8 clips severely — the documented reason the MobileNetV3 paper ships a "minimalistic" variant with plain ReLU and no SE |
| 7 | QAT as a fix? | Blocked: `tensorflow-model-optimization` 0.8.1 requires legacy Keras 2; this is Keras 3 |
| 8 | **Decision** | Ship dynamic-range: 0.64 accuracy points for a web-friendly 1.15 MB, with float32 activations that let XNNPACK run it in 22 ms |

Step 2 is the one to dwell on: the intuitive hypothesis was not just wrong, it made
things *worse*, and chasing that surprise is what led to the real answer. Step 4 is the
technique — when two variables are coupled, find the configuration that changes one and
not the other.

Full write-up: [`docs/quantization_findings.md`](docs/quantization_findings.md).

**Production artifact details.** `models/cropguard_v1_production.tflite` is
dynamic-range, **float32 in/out** (so no scale/zero-point arithmetic downstream) and
**static batch size 1** — LiteRT.js's JS wrapper does not handle TFLite's dynamic `-1`
batch dimension, which was fixed by re-exporting the Keras model with a fixed
`batch_size=1` Input before conversion.

---

## 🎚 Confidence tiers and calibration: turning a guess into a measurement

The app never shows a raw prediction. `web/src/lib/confidenceTier.ts` gates the UI on the
**calibrated** top-1 probability:

- **HIGH (≥94.5%)** — full diagnosis plus the treatment/prevention panel.
- **MODERATE (59.5–94.5%)** — caution banner, top-3 alternatives, treatment with a caveat.
- **LOW (<59.5%)** — **no headline diagnosis and no treatment panel**; top-3 offered as
  "possible matches", with a prompt to retake the photo or consult an expert.

Design principle: **a false-confident wrong diagnosis is worse than no diagnosis.**

Those thresholds started life as intuited magic numbers (0.85 / 0.50). They are now
*derived*: `python -m src.calibration` calibrates the production `.tflite` on the 3,625-image
**validation** split and picks the boundaries from the empirical accuracy curve, and every
figure below is then read off the held-out test split. Full write-up:
[`docs/CALIBRATION.md`](docs/CALIBRATION.md).

### Step 1 — the model was under-confident, not over-confident

Temperature scaling fits a single scalar T by minimizing validation NLL. Modern CNNs are
famously *over*confident and need T > 1. This one came out at **T = 0.8878** — mean
confidence 0.9223 against 0.9401 accuracy on test, so it needed *sharpening*. Plausible
cause: fine-tuning at 1e-5 / 5e-6 under heavy realism augmentation, both of which suppress
logit scale.

| | ECE | Brier | NLL |
|---|---|---|---|
| test, raw softmax | 0.0187 | 0.0934 | 0.1847 |
| test, temperature scaled | **0.0075** | 0.0925 | 0.1819 |

The transform is monotone, so **no prediction ever changes** — only the confidence value,
and therefore the tier. In the browser it is one line: the graph already ends in softmax, so
`p ** (1/T)` renormalized *is* temperature scaling on the hidden logits (verified against the
Python path to 5.6e-16).

### Step 2 — a 0.95 accuracy target turned out to be worthless

The obvious rule — "pick the smallest threshold whose accuracy reaches 0.95" — is a trap
when unconditional accuracy is already 0.940. It yields τ = 0.51, answers 98% of everything,
and *quadruples* the confident-error rate. So the script prints the whole sweep and makes
the choice of target an explicit product decision:

| Accuracy target | Threshold | HIGH coverage | Confidently wrong |
|---|---|---|---|
| 0.95 | 0.510 | 98.0% | 4.88% |
| 0.97 | 0.680 | 92.8% | 2.76% |
| 0.98 | 0.815 | 87.4% | 1.74% |
| **0.99 (shipped)** | **0.945** | **75.4%** | **0.69%** |
| 0.995 | 0.975 | 66.7% | 0.28% |

### Step 3 — what shipped, on the held-out test split

| | Tier | Coverage | Top-1 acc | Top-3 acc |
|---|---|---|---|---|
| old: raw, 0.85 / 0.50 | HIGH | 82.6% | 0.9860 | 0.9990 |
| | MODERATE | 14.9% | 0.7681 | 0.9852 |
| | LOW | 2.5% | 0.4505 | 0.9341 |
| **new: calibrated, 0.945 / 0.595** | HIGH | 74.4% | **0.9930** | 0.9993 |
| | MODERATE | 20.9% | 0.8393 | 0.9895 |
| | LOW | 4.7% | 0.5529 | 0.9588 |

**Confidently wrong — HIGH tier and incorrect — fell from 1.16% of all images to 0.52%,
for 8 points of HIGH coverage.** Nothing is hidden by that trade: images leaving HIGH land
in MODERATE, which still shows a diagnosis behind a caution banner. And the LOW tier's
0.9588 top-3 accuracy is what justifies offering three candidates there instead of nothing.

Selective prediction, the same idea as a curve (`outputs/risk_coverage.png`, AURC 0.0067):
abstaining on the least-confident 30% of images lifts accuracy on the rest to **0.9972**,
and at 50% coverage there are **zero** errors in 1,812 images. The shipped 0.945 sits at
the knee.

**What this still is not.** One scalar T cannot fix *per-class* miscalibration, and with
35× class imbalance that probably exists. The thresholds are derived on clean PlantVillage
data; they transfer usefully to field photos (below) on samples far too small to prove it.
And calibration measures confidence *given a leaf* — it says nothing about a photo of a
chair, which is the Phase 4 OOD gate.

---

## 🌍 Domain shift: what happens outside the lab

Every training image is a **single detached leaf on a uniform studio background under
controlled lighting.** Field photos are not like this. Measuring that gap is Phase 1, and
it comes in three parts — full discussion in
[`docs/DOMAIN_SHIFT.md`](docs/DOMAIN_SHIFT.md).

### Part 1 — developer-assembled set, all 17 classes: **0.5294**

17 images, one per class, covering every class in the taxonomy
(`real_world_test/`, scored by `python -m src.eval_real_world`).

| | This set (n=17) | Clean test | Drop |
|---|---|---|---|
| Top-1 | **0.5294** (CI [0.31, 0.74]) | 0.9401 | −0.4107 |
| Top-3 | 0.7059 | 0.9953 | −0.2894 |
| Crop identified correctly | 0.882 | — | — |
| **Confidently wrong** (pre-calibration) | **23.5%** | 1.16% | — |
| **Confidently wrong** (calibrated, shipped) | **5.9%** | 0.52% | — |

All 17 files were hashed against all 162,916 PlantVillage images first: **zero
duplicates**, so no training-set leakage. The four confident errors were each inspected
by eye and are model errors, not label errors.

**Accuracy collapses along taxonomic granularity:**

| Crop | Classes | n | Accuracy |
|---|---|---|---|
| Corn | 4 | 4 | **1.000** |
| Potato | 3 | 3 | 0.667 |
| Tomato | **10** | 10 | **0.300** |

The crop is identified correctly 88% of the time while the full class is right only 59% of
the time. The model is not confused about what it is looking at — it cannot separate
similar lesions *within* a crop, which is exactly where the clean-test confusion matrix was
already weakest.

**And the confidence gate did not hold.** HIGH-tier coverage only fell to 59% while HIGH
accuracy fell to 0.60, so **23.5% of images were confident and wrong** — twenty times the
benchmark rate. On the two other test sets the model went *unconfident* under shift and the
tier system converted that into abstention; here it stayed confident and was wrong. That
makes abstention a useful mitigation, **not a safety guarantee**.

All four confident errors are plausible confusions between visually adjacent classes —
Potato Late blight → *Tomato* Late blight (right disease, wrong crop), Bacterial spot →
Septoria, Septoria → Early blight, mosaic virus → Leaf Mold — and every one would have been
shown to a user as a full diagnosis with a treatment panel.

One correction to a previous hypothesis: **all 3 healthy leaves were correctly identified**,
and no diseased leaf was ever called healthy. Part 2 saw the opposite on its 3 healthy
images, so neither sample supports a claim about healthy-class bias.

To extend the set: [`real_world_test/HOW_TO_ADD_PHOTOS.md`](real_world_test/HOW_TO_ADD_PHOTOS.md),
validated by `python scripts/check_test_photos.py`.

### Part 2 — web-sourced photos (n=20): **0.40 top-1**

20 leaf photos harvested from Wikimedia Commons via `scripts/harvest_commons.py`, which
records title, page URL, licence, author and caption for every file — then **every
candidate was viewed and vetted individually.**

| | Web-sourced | Clean test | Drop |
|---|---|---|---|
| Top-1 | **0.40** (CI [0.22, 0.61]) | 0.9401 | −0.54 |
| Top-3 | 0.75 | 0.9953 | −0.25 |
| Crop identified correctly | 0.80 | — | — |
| HIGH-tier coverage | **20%** | 82.6% | — |
| Confidently wrong | 1 of 20 | 1.16% | — |

Three readings:

1. **The tier system held.** HIGH coverage collapsed from 83% to 20%, and only one image
   out of 20 was confidently wrong. Faced with input it has never seen, the model mostly
   *declines to be confident* rather than being confidently wrong — the app would have
   shown a caution banner or refused to diagnose on 80% of these. This is the strongest
   real-data justification for the whole tier design.
2. **Crop right, disease wrong.** 80% crop accuracy against 40% class accuracy, with
   top-3 at 0.75. The failure is not "no idea what it is looking at" — fine-grained
   lesion discrimination breaks under shift, which is exactly the axis on which studio
   conditions are least representative.
3. **Every healthy leaf was called diseased** (3/3), and no diseased leaf was called
   healthy. On n=3 that is a hypothesis rather than a finding, but the direction matters:
   errors here are **false alarms, not missed disease.**

**The vetting is itself a result.** 49 candidates were harvested; 29 were rejected:

| Rejected | Count | Examples |
|---|---|---|
| Not a photograph | 6 | an 1882 book scan, a disease-cycle diagram, a histological line drawing, a phylogenetic tree |
| Wrong species | 2 | *Ipomoea batatas* (sweet potato) for `Potato___healthy`; a pepper leaf for `Tomato___Bacterial_spot` |
| **Actively wrong label** | 2 | two images captioned *Septoria leaf spot* returned into the `Tomato___healthy` folder |
| Wrong plant part | 2 | a lesion on the maize stalk sheath; a macro of leaf trichomes |
| Caption/symptom mismatch | 1 | a sunken concentric lesion on an entire-margined leaf captioned *Passalora fulva* |
| Whole plant → routed to `_ood/` | 4 | corn canopy, young plant with weeds, tomato plant with fruit |
| Duplicates across search passes | 12 | |

**More than half the candidates were unusable and two arrived mislabelled.** An unvetted
scrape would have produced a confident, meaningless number — which is the concrete
argument for why Part 1 still matters. Details, licensing and per-image provenance:
[`web_sourced_test/README.md`](web_sourced_test/README.md).

### Part 3 — synthetic corruption stress test (n=602)

`src/eval_domain_shift.py` corrupts clean test images one factor at a time and scores
each corruption separately (`outputs/domain_shift_report.json`):

| Corruption | Top-1 | Δ vs clean | Top-3 | HIGH coverage | HIGH acc | Confidently wrong |
|---|---|---|---|---|---|---|
| sensor_noise_severe | 0.5399 | **−0.3937** | 0.7990 | 47.3% | 0.761 | **11.3%** |
| field_composite | 0.7641 | −0.1694 | 0.9352 | 46.0% | **0.971** | 1.3% |
| underexposed | 0.7674 | −0.1661 | 0.9086 | 55.8% | 0.961 | 2.2% |
| background_replace | 0.8721 | −0.0615 | 0.9850 | 61.5% | 0.989 | 0.7% |
| defocus_blur | 0.8837 | −0.0498 | 0.9850 | 72.1% | 0.988 | 0.8% |
| jpeg_artifacts | 0.8887 | −0.0449 | 0.9850 | 72.9% | 0.979 | 1.5% |
| off_angle | 0.9053 | −0.0282 | 0.9967 | 74.4% | 0.991 | 0.7% |
| motion_blur | 0.9203 | −0.0133 | 0.9850 | 72.9% | 0.991 | 0.7% |
| white_balance | 0.9269 | −0.0066 | 0.9934 | 78.9% | 0.989 | 0.8% |
| **clean (subsample)** | **0.9336** | — | 0.9967 | 80.1% | 0.994 | 0.5% |
| overexposed | 0.9435 | +0.0100 | 0.9967 | 79.7% | 0.988 | 1.0% |

Clean reads 0.9336 rather than 0.9401 because this is a 602-image stratified subsample
that over-weights rare classes; every Δ is against this subsample's own clean row.

**The key finding: robustness tracks the augmentation recipe and nothing else.** Rank the
corruptions by damage and you recover the contents of `src/augmentation.py` — motion
blur, JPEG, white balance, off-angle and background replacement each cost ≤ 0.06.
Gaussian sensor noise, the one family *absent* from the training augmentation, costs
**0.39** and takes the confidence gate down with it (11.3% confidently wrong versus 0.5%
clean). So the measured robustness is evidence of *augmentation coverage*, not of general
robustness — and every number in this table is a **lower bound**. Adding a noise
transform and re-measuring is now a concrete Phase 5 ablation row rather than a guess.

Also worth noting: `underexposed` costs 0.17 and is completely ordinary user behaviour (a
leaf in shade), whereas `overexposed` costs nothing — consistent with a dataset of
brightly and evenly lit studio images.

---

## 🔬 Explainability: what the model actually looks at

Accuracy says how often the model is right. It does not say *why*. That matters here
because the whole training set is a detached leaf on a plain background, so a network can
score 0.94 by reading the background — the reason `BackgroundReplace` exists in the
augmentation pipeline. `python -m src.explain` runs Grad-CAM over the Keras model and
measures it. Full write-up: [`docs/EXPLAINABILITY.md`](docs/EXPLAINABILITY.md).

The metric: fraction of CAM mass inside a leaf mask, divided by the mask's own area
fraction — because a heatmap that ignores the image entirely already scores
`mass == area`. So **lift > 1 means attention concentrates on the leaf**.

| 200 random test images | leaf area | CAM mass in leaf | lift | hottest pixel on leaf |
|---|---|---|---|---|
| all | 0.387 | 0.625 | **1.66** | **86%** |
| correct (n=190) | 0.390 | 0.629 | 1.66 | 86% |
| wrong (n=10) | 0.335 | 0.551 | 1.56 | 80% |

**No background shortcut.** Attention is 1.66× denser on the leaf than chance, and the
corruption sweep already showed background replacement costing only 0.06 accuracy. The
accuracy collapse under domain shift is therefore *not* explained by background reliance.

**And attention does not flag its own errors** — the correct-vs-wrong gap is real in
direction but tiny, on 10 wrong images. Not an abstention signal.

Three failures that were hypotheses before now have explanations:

- **The tomato brown-lesion cluster** (its accuracy falls to 0.30 on field photos) is a
  discrimination failure, not leakage: on cluster errors the hottest pixel is on the leaf
  87.5% of the time. Rendering the predicted *and* true class side by side shows two
  plausible maps on different lesion regions. Brown spots versus brown spots, at a 7×7
  feature grid.
- **The surviving confident errors** put near-identical heatmaps for both classes on the
  correct lesion. The Potato → *Tomato* Late blight error at 0.957 finds the right
  pathology and picks the wrong host.
- **Whole-plant photos** are the worst case: the map latches onto one leaf-shaped blob and
  commits — including a tomato field scene at **0.983 calibrated, above the shipped HIGH
  gate**. A closed-set softmax cannot say "that is not one leaf", which is exactly what
  Phase 4's OOD gate is for.

Two side findings worth their own lines: at the final layer, Grad-CAM on this architecture
is provably identical to plain CAM (`d logit/d A = W[k,c]/49`, verified to 4e-9), which is
why a second 14×14 map is computed; and the shipped quantized model disagrees with the
float model on **1 shifted prediction in 5** while agreeing on 98% of clean ones — so
every heatmap explains the float model, not always the browser.

Honest caveat: the leaf mask is the same rough GrabCut segmentation, and on field photos it
often segments the *lesion* instead of the leaf. The test-split sweep above is the
measurement; the field-photo attention numbers are descriptive.

---

## 🏛 Architecture

Two halves, no server between them:

1. **Python training pipeline** (`src/`) — dataset scoping and stratified splitting,
   realism-focused augmentation, two-phase transfer learning, evaluation, TFLite
   quantization, and the evaluation scripts described above.
2. **Web frontend** (`web/`) — Next.js 16 / React 19 / TypeScript / Tailwind 4, running
   the `.tflite` model through **LiteRT.js** (WebAssembly + XNNPACK) directly in the
   browser. WASM binaries are vendored in `web/public/litert_wasm/`, never fetched from a
   CDN.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed data flow and
[`MODEL_CARD.md`](MODEL_CARD.md) for the model itself.

**Repository layout**

| Path | What it is |
|---|---|
| `src/` | Python pipeline. Run as packages from the repo root (`python -m src.train`). |
| `web/` | **Current** frontend (Next.js/TypeScript). |
| `app/` | **Legacy** vanilla HTML/JS LiteRT harness, kept as reference. New work goes in `web/`. |
| `docs/` | Architecture, the quantization investigation, and the domain-shift, calibration and explainability write-ups. |
| `real_world_test/` | Hand-taken field photos. **Empty on purpose** — see its README. |
| `web_sourced_test/` | 20 hand-vetted Commons photos + `provenance.json` (licence/attribution). |
| `scripts/investigations/` | One-off diagnostics from the quantization hunt; not part of the pipeline. |
| `models/`, `outputs/` | Committed artifacts. `data/` and `venv/` are gitignored. |

### Frontend implementation notes

- **LiteRT.js lifecycle** lives entirely in `web/src/lib/useCropGuardModel.ts`: a
  module-level init promise guards against React StrictMode double-mount, the compiled
  model lives in a ref (not state), and input/output `Tensor`s are explicitly `.delete()`d
  to avoid WASM heap leaks. Client-only — SSR cannot touch WASM or browser APIs.
- **Preprocessing parity (do not "optimize").** `web/src/lib/preprocess.ts` reproduces
  `tf.image.resize(bilinear)` + `/255.0` using a 224×224 canvas with
  `imageSmoothingQuality = 'medium'` — bilinear, matching TF's default, *not*
  `'high'`/bicubic. Verified numerically against the Python pipeline to 4+ decimals.
- **Input methods:** file upload, camera (`<input type="file" capture="environment">`,
  the native camera app rather than a live `getUserMedia()` preview), and drag-and-drop —
  all funnelling into one preprocess/infer path.
- **Regression check** — re-run after *any* change to preprocessing, model loading or the
  inference path. This has caught real bugs:

  | Image | Expected (raw softmax) | Expected (calibrated, what the UI shows) |
  |---|---|---|
  | `Corn_(maize)___Common_rust_/RS_Rust 2370.JPG` | 99.9954% Corn_Common_rust | ~99.998% |
  | `Potato___healthy/...RS_HL 1951.JPG` | 99.8074% Potato_healthy | ~99.91% |
  | `Tomato___Leaf_Mold/...Crnl_L.Mold 7082.JPG` | 99.7541% Tomato_Leaf_Mold | ~99.90% |

  Since Phase 2 the UI displays the **calibrated** value (T = 0.8878), so the third column
  is the one to compare against on screen; the raw column is still visible in the debug
  panel. The calibrated figures come from the Python path — expect agreement to a few
  decimals, not to the last digit.

### The model/web contract (breaks silently)

Four things must stay in sync between Python and `web/src/lib/`:

1. **Class index order.** Python uses `sorted(df['label'].unique())` over PlantVillage
   folder names; `constants.ts` `CLASS_NAMES` is the same order with display-sanitized
   names. Editing one without the other silently mislabels every prediction.
2. **Preprocessing.** Bilinear resize plus `/255.0`, as above.
3. **Model artifact.** `models/cropguard_v1_production.tflite` is copied to
   `web/public/` **and** `app/public/` — regenerating means recopying all copies.
4. **Accelerator.** Dynamic-range quantized, loaded with `{ accelerator: 'wasm' }`
   (XNNPACK). Do not switch the converter to full-integer quantization expecting it to
   work.

Contract tests for these are Phase 6. In the meantime, `_sanitize()` in
`src/eval_real_world.py` reproduces `CLASS_NAMES` from the Python label order exactly,
which is a working draft of test #1.

---

## 💻 Tech stack

- **Model:** TensorFlow 2.21 / Keras 3.15, MobileNetV3-Small (ImageNet-pretrained)
- **Data & augmentation:** pandas, Albumentations 2.0.8, opencv-python-headless 4.11
  (pinned — 5.0 is blocked by a Windows Application Control policy)
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
python -m src.augmentation    # augmentation previews -> outputs/
```

Evaluation and robustness:

```bash
python -m src.eval_tiers          # UNCALIBRATED tier baseline -> outputs/tier_report.json
python -m src.calibration         # ECE/temperature/thresholds -> outputs/calibration_report.json
python -m src.eval_domain_shift   # corruption stress test   -> outputs/domain_shift_report.json
python scripts/check_test_photos.py   # validate a photo set before scoring it
python -m src.eval_real_world     # hand-taken photos        -> outputs/real_world_report.json
python -m src.eval_real_world --dir web_sourced_test --json outputs/web_sourced_report.json
python -m src.eval_real_world --calibrated   # score photos the way the shipped UI gates them
python scripts/harvest_commons.py --out <staging> --per-class 4   # rebuild web candidates
```

`src/quantize.py` **skips conversion if the target `.tflite` already exists** — delete the
file to force re-conversion.

> **Windows notes.** TF ≥ 2.11 has no native Windows GPU support, and a local Application
> Control policy blocks some native DLLs used during TFLite *calibration* — so full-INT8
> conversion can fail locally while training, inference and evaluation work fine. If a
> DLL-load error appears, check `requirements.txt` for an already-known pinned-version
> workaround before debugging from scratch. Models are saved as `.keras`, not `.h5`:
> Keras 3 cannot round-trip a MobileNetV3 backbone from `.h5` (hard-swish fails on load).

### Web app

```bash
cd web
npm install
npm run dev     # http://localhost:3000
npm run build
npm run lint
```

`web/next.config.ts` sets `logging.browserConsole: false` because the Emscripten runtime
writes via low-level `_fd_write`, which otherwise floods the Turbopack terminal. It also
holds a hardcoded LAN IP in `allowedDevOrigins` for phone testing — update it for your
network.

---

## 💊 Treatment & actionable advice

Every prediction resolves to a typed entry in `web/src/lib/diseaseInfo.ts` — 17 entries,
one per class — containing the pathogen/cause, prevention bullets, a **category-level**
treatment description (deliberately no specific chemical or dosing), and a link to a
university extension source.

All 17 reference links were checked and **7 were dead 404s**; they have been replaced and
re-verified, and all 17 now return 200. Worth knowing for the next check:
`extension.umn.edu` returns 403 to a bare `curl` — that is bot-blocking, not a dead link.
Re-check with a browser User-Agent before replacing anything.

The panel is gated by confidence: shown in full at HIGH, shown with a caveat at MODERATE,
and **withheld entirely at LOW**, where displaying treatment for a diagnosis the app has
just called unreliable would contradict itself.

---

## ⚠️ Known limitations

1. **OOD inputs can be confidently wrong — now gated, not solved.** The classifier is
   closed-set: 9.9% of 101 non-leaf photos reached the HIGH tier and 62.4% got some
   diagnosis shown. A cosine-to-class-mean gate on the embedding
   (`web/src/lib/oodGate.ts`) now rejects them before any diagnosis appears, cutting
   those to 2.0% / 5.9% while keeping 90% of real field leaf photos. What still gets
   through is mostly *other species'* leaves — the gate knows "leaf", not "my crop".
   Its threshold is a percentile of only 37 field photos. See `docs/OOD.md`.
2. **Confidence tiers help but do not fully solve false confidence.** 0.52% of test
   images are still HIGH-tier and wrong (was 1.16% before calibration). Tiering catches
   genuine uncertainty, not confident errors on inputs that merely resemble the training
   distribution.
3. **Calibration is a single scalar.** T = 0.8878 fixes the *average* confidence gap
   (test ECE 0.0075) but cannot fix **per-class** miscalibration, which with 35× class
   imbalance probably exists and is unmeasured. ECE with 15 equal-width bins is also a
   biased estimator, and 92% of validation images fall in the top two bins.
4. **Out-of-distribution accuracy is far below the benchmark** — 0.5294 on the 17-class
   set, 0.40 on the web-sourced set, against 0.9401 on the benchmark.
5. **The confidence gate is not a safety guarantee.** Calibration cut the 17-class set's
   confidently-wrong rate from 23.5% to 5.9%, but one confident error survives — a
   `Bacterial_spot → Septoria_leaf_spot` confusion at 0.971, inside the tomato
   brown-lesion cluster. That is a discrimination failure no threshold can reach. And the
   sample is 17 images.
6. **Robust only to what it was augmented with.** Gaussian sensor noise costs 0.39
   accuracy versus ≤ 0.06 for every augmented corruption family.
7. **Only 3 of 14 PlantVillage crops** — a data-filter change to widen, not an
   architecture change.
8. **No baselines or ablations**, so architecture and augmentation choices rest on
   argument rather than controlled measurement.
9. **Attention is measured, but the ruler is rough.** Grad-CAM mass sits 1.66× more
   densely on the leaf than area alone would give (86% of hottest pixels on-leaf), so the
   background shortcut `BackgroundReplace` was written against is not present. But the
   leaf mask is the same rough GrabCut segmentation, which on field photos often segments
   the lesion rather than the leaf, and only 10 wrong predictions fell in the sweep — too
   few to tell whether off-leaf attention predicts error. See `docs/EXPLAINABILITY.md`.
10. **No test suite and no experiment tracking.** Results live in JSON reports and prose,
   not a machine-readable registry.

---

## 📍 Project status by phase

Roadmap and full detail in `plan.md`.

| Phase | Scope | Status |
|---|---|---|
| **0 — Housekeeping & honesty** | README rewrite, link-check, model card, tier confirmation | ✅ **Complete** |
| **1 — Real-world test set** | Field evaluation tooling, domain-shift measurement | ✅ **Complete** |
| **2 — Calibration** | ECE/MCE, reliability diagram, temperature scaling (T = 0.8878), derived thresholds (0.945 / 0.595), risk–coverage (AURC 0.0067) | ✅ **Complete** |
| **3 — Explainability** | Grad-CAM (deep 7×7 + mid 14×14), CAM-mass-in-leaf leakage metric, 200-image sweep, per-cohort attention analysis | ✅ **Complete** |
| 4 — OOD gate | MSP vs energy vs Mahalanobis, AUROC/FPR@95, ship the winner | ⬜ Not started |
| 5 — Baselines & ablations | Sanity floors, architecture comparison, augmentation / class-weight / fine-tuning ablations, INT8 "minimalistic" capstone | ⬜ Not started |
| 6 — Engineering rigor | pytest suite, cross-language preprocessing parity test, CI | ⬜ Not started |
| 7 — Optional stretch | Scan history, kNN explanations, all 38 classes, PWA | ⬜ Not started |

**Phase 0 — complete.** This README was rewritten against reality: it previously claimed a
vanilla HTML/JS PWA frontend that was "IN PROGRESS", a treatment table that was "NOT
STARTED", and pointed setup instructions at the legacy `app/` directory. All 17 extension
links were checked and 7 replaced. `MODEL_CARD.md` was added. Tier behaviour was measured
empirically for the first time via a new `src/eval_tiers.py`.

**Phase 1 — complete.** Built: `src/eval_real_world.py`, `src/eval_domain_shift.py`,
`scripts/harvest_commons.py`, `scripts/check_test_photos.py`, the 17-image all-classes
`real_world_test/` set, the 20-image `web_sourced_test/` set with full provenance, and
`docs/DOMAIN_SHIFT.md`. Measured across three independent test sets: **0.5294** (17
classes), **0.40** (web-sourced), **0.7641** (stacked synthetic corruption) — plus the
corruption ranking and the tier system's behaviour under shift.

The most consequential result is a negative one. Two of the three sets showed the
confidence gate degrading gracefully; the third showed it failing outright at 23.5%
confidently-wrong. **That made Phase 2 (calibration) and Phase 4 (the OOD gate) load-
bearing rather than nice-to-have** — they are what turn abstention from something that
usually works into something that can be relied on. A larger set with several images per
class would tighten the intervals, but the direction is no longer in question.

**Phase 2 — complete.** Built: `src/calibration.py`, `web/src/lib/calibration.ts`,
`docs/CALIBRATION.md`, `outputs/reliability_diagram.png`, `outputs/risk_coverage.png`,
and a `--calibrated` mode on `src/eval_real_world.py`. The model turned out mildly
*under*-confident (T = 0.8878, where the literature expects T > 1), test ECE halved to
0.0075, and the tier thresholds moved from intuited 0.85 / 0.50 to derived 0.945 / 0.595.
Confidently-wrong fell **1.16% → 0.52%** on the benchmark and **23.5% → 5.9%** on the
17-image field set, for 8 points of HIGH coverage. The methodological finding worth as
much as the numbers: the plan's own suggested 0.95 accuracy target was worthless against a
0.940 unconditional accuracy, and taking it literally would have made the app *less* safe.
Still no model change — the weights remain byte-identical.

**Phase 3 — complete.** Built: `src/explain.py`, `docs/EXPLAINABILITY.md`, and 57
attention panels in `outputs/gradcam/`. The background-leakage question is answered — CAM
mass concentrates on the leaf at **1.66× area** on held-out test data, and the corruption
sweep already showed the model nearly indifferent to background replacement — so the
accuracy collapse under shift is *not* a background shortcut. Three open failures now have
explanations rather than hypotheses: the brown-lesion cluster is a discrimination failure
with attention correctly on-leaf; the two surviving confident errors put identical
heatmaps on the right lesion and pick the wrong class (one wrong *host*); and on
whole-plant photos the map latches onto a single leaf-shaped blob and commits — one at
0.983 calibrated, above the HIGH gate, which is the clearest argument yet for Phase 4.
Two side findings: at the final layer Grad-CAM here is provably identical to CAM (verified
numerically), and the shipped quantized model disagrees with the float model on ~1 shifted
prediction in 5 while agreeing on 98% of clean ones.

**Nothing in Phases 0–3 changed the model.** No retraining, no accuracy improvement — the
weights are byte-identical to where they started. What changed is how much is *known*: the
tier thresholds went from unjustified magic numbers to measured operating points, the
out-of-distribution gap went from unmeasured to quantified across three test sets, the
robustness profile went from assumed to ranked, and the safety story went from "the tier
system protects the user" to "the tier system protects the user on two of three
distributions tested" to "the tier system is measured, and its worst case is 5.9% rather
than 23.5%", and the attention maps now say the remaining errors are discrimination
failures rather than data artifacts. Phases 4–5 are where numbers would move next — an OOD gate would cut the
remaining confidently-wrong rate, and a
noise augmentation would close the 0.39 gap identified in Part 3.

---

## 🖼️ Demo

Run `cd web && npm run dev` and drop a leaf photo onto the page. Screenshots and a
recorded walkthrough are not yet in the repo.
