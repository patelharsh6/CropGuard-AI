# Model Card — CropGuard AI v1

**Model:** `models/cropguard_v1_production.tflite` (dynamic-range quantized)
**Source model:** `models/cropguard_v1.keras` (float32)
**Version:** v1 — 2026-08
**Owner:** CropGuard AI project (personal/portfolio)
**Use:** research and demonstration only

---

## 1. Intended use

**Intended:** an on-device, in-browser aid for identifying common foliar diseases
from a **close-up photo of a single leaf** of **tomato, potato, or corn**, filling
the frame. The model returns a probability distribution over 17 classes; the app
gates what it shows on that confidence (see §6).

**Out of scope — the model will still return a confident answer, and it will be wrong:**

- Any crop other than tomato, potato, or corn (11 of PlantVillage's 14 crops are excluded).
- Whole-plant, field, or multi-leaf photos. Measured failure: a whole-plant garden
  photo returned **79.7% confidence on `Corn_healthy`** — a wrong class, above the
  MODERATE gate.
- Non-plant images (faces, soil, sky, text). The classifier itself is closed-set and
  will name a disease for any of them; a **pre-classification OOD gate** now rejects
  them in the UI before a diagnosis is shown (§8.1, `docs/OOD.md`). The gate is not
  part of the classifier — anyone calling the model directly gets the closed-set
  behaviour.
- Fruit, stem, root, or tuber symptoms — training data is leaves only.
- Nutrient deficiency, herbicide damage, or abiotic stress — no such class exists,
  so these are forced into a disease class.
- Diseases outside the 17 classes, including co-infections.

**Not a substitute for an agronomist.** This model does not diagnose, and its output
must not be the sole basis for a pesticide, fungicide, or crop-destruction decision.
Treatment text in the app is *general, category-level* guidance sourced from
university extension services — it deliberately contains no chemical dosing — and
every prediction should be confirmed by a local extension office or plant clinic
before money or chemicals are spent.

---

## 2. Model details

| | |
|---|---|
| Architecture | MobileNetV3-Small (`tf.keras.applications`, ImageNet-pretrained) → GlobalAveragePooling2D → Dropout(0.3) → Dense(17, softmax) |
| Parameters | 948,929 |
| Input | 224×224×3 RGB, float32, scaled to [0,1] (bilinear resize) |
| Output | 17-way softmax, float32 |
| Training | Two-phase transfer learning. Phase 1: frozen backbone, Adam(1e-3) → val 0.8908. Phase 2: last ~25% unfrozen (first 117/157 layers frozen, BatchNorm frozen throughout), Adam(1e-5) → 0.9454, then 15 more epochs at Adam(5e-6) → **val 0.9481** |
| Class imbalance | `sklearn.compute_class_weight('balanced', ...)` |
| Production quantization | Dynamic-range PTQ (INT8 weights, float32 activations), static batch 1, float32 in/out |
| Size | 3.62 MB float32 Keras → **1.15 MB** TFLite |
| Runtime | LiteRT.js (WebAssembly + XNNPACK), fully client-side |
| Latency | **22 ms** median, single image, laptop CPU |

Full-integer INT8 was evaluated and **rejected** — it collapsed accuracy to 0.7630.
See §7 and `docs/quantization_findings.md`.

---

## 3. Training data

**PlantVillage**, `color` (unsegmented) variant — deliberately not the
background-removed variant, since real photographs have natural backgrounds.

- **17 classes**, 3 crops: Corn (Cercospora/Gray leaf spot, Common rust, Northern
  Leaf Blight, healthy); Potato (Early blight, Late blight, healthy); Tomato
  (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider
  mites, Target Spot, Yellow Leaf Curl Virus, mosaic virus, healthy).
- **Split:** stratified 70/15/15 → **16,914 / 3,625 / 3,625** (`data/dataset_split.csv`).
- **Imbalance:** ~35×, from `Potato___healthy` (152 train images) to
  `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (5,357).
- **Augmentation** (train split only): affine rotate/scale, flips,
  brightness/contrast, color-temperature jitter, plus realism transforms — motion
  blur, JPEG artifacts, CoarseDropout — plus a custom **`BackgroundReplace`**
  (GrabCut leaf segmentation, feathered alpha blend onto procedural
  soil/grass/wood/hand textures, p=0.4).

### The dominant bias: PlantVillage is a lab dataset

Every training image is a **single detached leaf, photographed against a uniform
studio background under controlled lighting**. Field photos are not like this.
`BackgroundReplace` exists specifically to attack this gap, but it is a synthetic
approximation, not field data. **The generalization gap to real phone photos has not
yet been measured** — no self-labelled field test set exists (`real_world_test/` is
scaffolded and empty on purpose). A synthetic stress test puts a lower bound on it:
see §5b and `docs/DOMAIN_SHIFT.md`. Treat 0.9401 as a *benchmark* number, not a
deployment number.

Secondary biases: PlantVillage's own label noise is inherited unexamined, and the
geographic origin of the images is not controlled, so pathogen strains and cultivars
represented here may not match a given region.

---

## 4. Evaluation

Held-out test split, 3,625 images, never used for training or model selection.

| Model | Top-1 | Top-3 |
|---|---|---|
| Float32 Keras | **0.9465** | **0.9961** |
| Production dynamic-range TFLite | **0.9401** | 0.9953 |

Per-class metrics: `outputs/classification_report.json`. Confusion matrix:
`outputs/confusion_matrix.png`.

**Macro-averaged** (float32): precision 0.9332, recall 0.9484, F1 0.9402. Macro
sitting close to weighted (0.9477 / 0.9465 / 0.9467) is the evidence that class
weighting worked rather than the model riding the majority classes.

**Weakest classes by F1** (float32):

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| Tomato Target Spot | 0.847 | 0.895 | 0.870 | 210 |
| Tomato Early blight | 0.872 | 0.867 | 0.870 | 150 |
| Tomato Septoria leaf spot | 0.925 | 0.883 | 0.904 | 266 |
| Potato healthy | 0.880 | 0.957 | 0.917 | 23 |
| Corn Gray leaf spot | 0.872 | 0.974 | 0.920 | 77 |

**Rare-class check** — the two smallest training classes did *not* collapse:
`Potato___healthy` (152 train images) → recall 0.957; `Tomato___mosaic_virus`
(373) → recall 1.000.

**Confusion is structured, not noise:**

- Tomato brown-lesion cluster: Early blight ↔ Target Spot ↔ Septoria ↔ Spider mites.
  These are genuinely hard to separate from a single leaf photo; extension guides
  distinguish them partly by lesion halo and by distribution across the plant —
  information a cropped leaf image may not contain.
- One corn pair: Northern Leaf Blight → Gray leaf spot.

---

## 5. Confidence behaviour (measured)

The app applies **temperature scaling** (T = 0.8878) to the model's softmax before
reading any confidence, then gates on thresholds **derived** from the validation split.
Run `python -m src.calibration`; results in `outputs/calibration_report.json`, write-up
in `docs/CALIBRATION.md`. Production TFLite over the 3,625-image test split:

| Tier | Rule (calibrated p) | Coverage | Top-1 acc | Top-3 acc |
|---|---|---|---|---|
| HIGH | p ≥ 0.945 | 74.4% | **0.9930** | 0.9993 |
| MODERATE | 0.595 ≤ p < 0.945 | 20.9% | 0.8393 | 0.9895 |
| LOW | p < 0.595 | 4.7% | 0.5529 | 0.9588 |

The gate works in the intended direction: accuracy falls monotonically across tiers, and
the LOW tier — where the app refuses to give a headline diagnosis — is where top-1 is
barely better than a coin flip while top-3 still holds at 0.9588, which is why three
candidates are offered there. **0.52% of all test images are confidently wrong** (HIGH
tier, incorrect); that residue is the failure mode the tier system does *not* catch.

Calibration quality on the test split (T fit on validation only): **ECE 0.0187 → 0.0075**,
Brier 0.0934 → 0.0925, NLL 0.1847 → 0.1819. Mean confidence was 1.8 points *below*
accuracy before scaling, i.e. the model was mildly under-confident, not over-confident.
Selective prediction: AURC 0.0067; answering only the most confident 70% of images gives
0.9972 accuracy.

The previous, intuited thresholds (0.85 / 0.50 on raw softmax) delivered HIGH coverage
82.6% at 0.9860 with **1.16%** confidently wrong — the earlier operating point, kept
measurable via `python -m src.eval_tiers` and `outputs/tier_report.json`.

Coverage/accuracy tradeoff behind the choice of 0.945 (validation split, calibrated —
each row is the smallest threshold meeting that accuracy target):

| Accuracy target | Threshold | Coverage | Confidently wrong |
|---|---|---|---|
| 0.95 | 0.510 | 98.0% | 4.88% |
| 0.97 | 0.680 | 92.8% | 2.76% |
| 0.98 | 0.815 | 87.4% | 1.74% |
| **0.99 (shipped)** | **0.945** | **75.4%** | **0.69%** |
| 0.995 | 0.975 | 66.7% | 0.28% |

Note that a 0.95 target is meaningless against a 0.940 unconditional accuracy — it is met
by answering nearly everything, which *raises* the absolute confident-error rate. The
choice of 0.99 is a product decision about acceptable harm, not a statistical one.

**Caveat — the thresholds are derived on clean PlantVillage validation data.** They
transfer usefully to out-of-distribution photos (§5b) on samples far too small to prove
it, a single scalar temperature cannot correct per-class miscalibration, and calibration
says nothing about inputs that are not leaves at all.

### 5b. Behaviour on out-of-distribution photographs

**Primary set — 17 images, all 17 classes** (`real_world_test/`, hashed against all
162,916 PlantVillage images: zero duplicates, so no training-set leakage):

| | This set (n=17) | Clean test |
|---|---|---|
| Top-1 | **0.5294** (95% CI [0.31, 0.74]) | 0.9401 |
| Top-3 | 0.7059 | 0.9953 |
| Crop identified correctly | 0.882 | — |
| Full class correct | 0.588 | — |
| **Confidently wrong** (pre-calibration) | **23.5%** | 1.16% |
| **Confidently wrong** (calibrated, shipped) | **5.9%** | 0.52% |

Accuracy is a function of taxonomic granularity: **corn 1.000** (4 classes), **potato
0.667** (3), **tomato 0.300** (10). The crop is right 88% of the time while the class is
right 59% — fine-grained lesion discrimination within a crop is what fails.

**The confidence gate did not hold on this set before calibration.** HIGH coverage fell
only to 58.8% while HIGH accuracy fell to 0.600, producing a 23.5% confidently-wrong rate
— twenty times the benchmark. Rescored with temperature scaling and the derived
thresholds (`python -m src.eval_real_world --calibrated`), HIGH coverage drops to 35.3%,
HIGH accuracy rises to 0.833, and **one** of the four confident errors survives
(Bacterial spot → Septoria at 0.971) — 5.9%. A substantial mitigation on a 17-image
sample, not a fix. All four confident errors are plausible confusions between adjacent classes
(Potato Late blight → *Tomato* Late blight; Bacterial spot → Septoria; Septoria → Early
blight; mosaic virus → Leaf Mold), and each would have been shown to a user as a full
diagnosis with a treatment panel. **Abstention is a mitigation, not a guarantee** — it
degraded gracefully on the two sets below and failed here.

All 3 healthy leaves were correctly identified, and 0 of 14 diseased leaves were called
healthy.

**Secondary set — web-sourced (n=20)**

20 leaf photos from Wikimedia Commons, hand-vetted (`web_sourced_test/README.md`),
scored with `python -m src.eval_real_world --dir web_sourced_test`:

| | Web-sourced | Clean test |
|---|---|---|
| Top-1 | **0.40** (95% CI [0.22, 0.61]) | 0.9401 |
| Top-3 | 0.75 | 0.9953 |
| Crop identified correctly | 0.80 | — |
| HIGH-tier coverage / accuracy | **20%** / 0.750 | 82.6% / 0.9860 |
| Confidently wrong | 1 of 20 | 1.16% |

Top-1 roughly halves. The gate holds: HIGH coverage collapses to 20%, so the app would
have declined to give a confident diagnosis on 80% of these rather than misdiagnosing.
The model still identifies the **crop** 80% of the time — fine-grained lesion
discrimination is what fails. All 3 healthy photos were called diseased and no diseased
photo was called healthy: errors run toward false alarms, not missed disease.

Caveats that must travel with this number: labels are uploader captions rather than
verified ground truth, n=20, selection is not random, and 7 of 17 classes are absent.

### 5c. Behaviour under distribution shift (synthetic)

`python -m src.eval_domain_shift`, 602 stratified test images, production artifact,
each corruption applied at p=1.0. Full table and discussion in `docs/DOMAIN_SHIFT.md`.

| | Top-1 | Δ vs clean | Top-3 | HIGH coverage | HIGH accuracy | Confidently wrong |
|---|---|---|---|---|---|---|
| Clean (this subsample) | 0.9336 | — | 0.9967 | 80.1% | 0.994 | 0.5% |
| Stacked field-like composite | 0.7641 | −0.1694 | 0.9352 | 46.0% | **0.971** | 1.3% |
| Underexposed | 0.7674 | −0.1661 | 0.9086 | 55.8% | 0.961 | 2.2% |
| Severe sensor noise | 0.5399 | **−0.3937** | 0.7990 | 47.3% | 0.761 | **11.3%** |

Two readings. **The tier system degrades gracefully:** under the field composite,
top-1 falls 17 points while HIGH-tier accuracy holds at 0.971 — HIGH *coverage*
collapses instead (80% → 46%), routing degraded inputs to a caution banner rather than
to a confident wrong answer. **But robustness tracks the augmentation recipe:** every
corruption present in `src/augmentation.py` costs ≤ 0.06, while Gaussian sensor noise —
absent from it — costs 0.39 and breaks the confidence gate with it. These numbers are a
**lower bound** on real field degradation, precisely because the corruptions overlap
what the model was trained on.

---

## 6. How the app uses the model

`web/src/lib/confidenceTier.ts` + `web/src/app/page.tsx`:

`web/src/lib/calibration.ts` applies temperature scaling first, so every percentage below
— and every percentage shown to the user — is a calibrated probability, not a raw softmax
score.

- **HIGH (≥94.5%)** — full diagnosis plus the treatment/prevention panel.
- **MODERATE (59.5–94.5%)** — caution banner, top-3 alternatives, treatment shown with a caveat.
- **LOW (<59.5%)** — **no headline diagnosis and no treatment panel** (showing it would
  contradict the stated uncertainty); top-3 offered as "possible matches", with a
  prompt to retake the photo or consult an expert.

Design principle: **a false-confident wrong diagnosis is worse than no diagnosis.**

Nothing leaves the device. Inference runs in the browser via LiteRT.js/WASM; there is
no inference server and no image upload.

---

## 7. Quantization

1. Full INT8 PTQ: 0.9465 → **0.7630**.
2. Hypothesis "uniform calibration starved rare classes" → stratified calibration
   (30/class) made it **worse** (0.6836), disproving it.
3. Bug hunt (channel order, normalization, indexing, pixel range) found **no bug**.
4. Dynamic-range (weights-only INT8) as a diagnostic: **0.9401** — weights quantize
   fine; **activation** quantization is the problem.
5. Root cause: MobileNetV3's hard-swish and squeeze-excite blocks are sensitive to
   per-tensor INT8 activation quantization — the reason the paper ships a
   "minimalistic" variant (no SE, plain ReLU).
6. QAT unavailable: `tensorflow-model-optimization` 0.8.1 requires legacy Keras 2.
7. **Shipped dynamic-range**, costing 0.64 accuracy points.

Full write-up: `docs/quantization_findings.md`.

---

## 8. Known failure modes

1. **Out-of-distribution inputs are confidently misclassified — gated in the UI, not
   fixed in the model.** Whole-plant photo → 79.7% on `Corn_healthy`; across a
   101-image negative set, 9.9% reached the HIGH tier and 62.4% got a diagnosis shown,
   with a maximum calibrated confidence of 0.9998. A class-mean-cosine gate on the
   576-d embedding (threshold 0.5981, AUROC 0.9711 against field leaf photos) cuts
   those to **2.0% / 5.9%** while accepting 90% of real field leaves. Residual: it
   detects "leaf", not "my crop" — oak-leaf close-ups pass. Its threshold is a
   percentile of 37 field photos. See `docs/OOD.md`.
2. **Confident-and-wrong survives the tier gate.** 0.52% of test images after
   calibration (1.16% before); separately, a stock photo of unverified provenance scored
   93.65%. Tiering catches model *uncertainty*, not confident errors on input that merely
   resembles the training distribution.
3. **Large, measured out-of-distribution gap.** Top-1 falls to **0.5294** on a 17-class
   set and **0.40** on a 20-image web-sourced set, from 0.9401 on the benchmark; a
   synthetic stacked-corruption proxy gives 0.7641 vs 0.9336 (§5b, §5c). Concentrated in
   tomato, which carries 10 of the 17 classes.
4. **The confidence gate is not a safety guarantee.** 5.9% of the 17-class set is still
   confident *and* wrong after calibration (23.5% before). Tiering degraded gracefully on
   two test distributions and failed on a third; calibration narrowed that failure but
   did not close it, so it cannot be relied on to prevent a false-confident diagnosis.
   The OOD gate (§8.1) removes the non-leaf part of the remainder; the part that is a
   real leaf of the wrong class is untouched by either mechanism.
5. **Brown-lesion tomato diseases are systematically confused** with each other — and
   that is where the one surviving confident error on the 17-class set sits. No
   confidence threshold can fix a discrimination failure.
6. **Calibration is one scalar.** T = 0.8878 corrects the average confidence gap (test
   ECE 0.0075) but not per-class miscalibration, which with 35× class imbalance is
   likely and unmeasured.
7. **Robust only to what it was augmented with.** Ranking corruptions by damage
   reproduces the contents of `src/augmentation.py`: blur, JPEG, white balance and
   background replacement each cost ≤ 0.06, while Gaussian sensor noise — the one
   family absent from the training augmentation — costs 0.39 and takes the confidence
   gate down with it (11.3% of images confidently wrong, versus 0.5% clean). Adding a
   noise transform and re-measuring is a Phase 5 ablation row.
8. **No explainability evidence.** It has not been verified that the model attends to
   the lesion rather than to background or leaf-shape shortcuts — a pointed gap given
   `BackgroundReplace` exists precisely because background leakage was suspected.
9. **No baselines or ablations.** The architecture, the two-phase schedule, the class
   weights, and `BackgroundReplace` are currently justified by argument rather than by
   a controlled comparison.

---

## 9. Reproducing

```bash
python -m src.data_pipeline   # scan dataset -> data/dataset_split.csv
python -m src.train           # -> models/cropguard_v1.keras
python -m src.evaluate        # -> outputs/classification_report.json, confusion_matrix.png
python -m src.quantize        # TFLite conversion + accuracy/latency
python -m src.eval_tiers      # -> outputs/tier_report.json (uncalibrated baseline)
python -m src.calibration     # -> outputs/calibration_report.json + the two plots
python -m src.eval_domain_shift  # -> outputs/domain_shift_report.json
python -m src.eval_real_world    # -> outputs/real_world_report.json (needs real_world_test/ photos)
```

Seed 42; the stratified split is saved to CSV so it is fixed across runs.
`src/quantize.py` skips conversion if the target `.tflite` already exists.
