# CropGuard AI — Project Plan & Status

*Onboarding context for a fresh coding-agent session, plus the forward roadmap.
Reflects the actual implementation on disk, not the original proposal — several
decisions changed during development.*

**Revision goal (2026-08-23):** the project is currently strong on *deployment
engineering* and thin on *core ML methodology*. This revision adds the ML
fundamentals work (calibration, explainability, OOD detection, baselines/ablations)
that an AI/ML interview will actually probe. See Part C for the concept→artifact map.

---

## 1. What This Project Is

A **plant disease diagnosis web app**: upload or photograph a leaf, get an instant
on-device AI prediction plus general treatment/prevention info. Core design
principle: **the AI is not a feature bolted onto an app — it is the app.**

**Scope (deliberate MVP decision):** 3 crops — Tomato, Potato, Corn — 17 classes
(diseases + healthy) from PlantVillage. NOT all 38 classes / 14 crops; deferred
until this narrow pipeline is proven end-to-end.

**Scope correction from the original plan:** this is a **plain website, not an
installable/offline PWA**. No service worker, no offline caching, no "Add to Home
Screen." Camera capture, upload, and inference are all still fully client-side and
private (nothing leaves the device) — the only thing dropped is working at *zero*
connectivity. Deliberate decision, not a late discovery.

---

# PART A — Current Status (audited against files on disk)

## A.1 Done and verified

| Component | Evidence on disk |
|---|---|
| Data pipeline (scoping, stratified 70/15/15 split) | `src/data_pipeline.py`, `data/dataset_split.csv` |
| Realism-focused augmentation (incl. custom `BackgroundReplace`) | `src/augmentation.py`, `outputs/aug_preview*.png` |
| Training — MobileNetV3-Small, 2-phase transfer learning | `src/train.py`, `models/cropguard_v1.keras`, `outputs/training_history.json` |
| Evaluation — per-class metrics + confusion matrix | `src/evaluate.py`, `outputs/classification_report.json`, `outputs/confusion_matrix.png` |
| Quantization + the INT8 investigation | `src/quantize.py`, `docs/quantization_findings.md`, `scripts/investigations/*` |
| Production artifact | `models/cropguard_v1_production.tflite` (1.15 MB, 0.9401 acc) |
| Web frontend (Next.js/TS, upload + camera + drag-drop, on-device inference) | `web/src/app/page.tsx`, `web/src/lib/useCropGuardModel.ts` |
| Preprocessing parity with Python | `web/src/lib/preprocess.ts` (verified to 4+ decimals on 3 images) |
| Treatment info lookup, 17 entries | `web/src/lib/diseaseInfo.ts` |
| Confidence-tier gating (HIGH/MODERATE/LOW) | `web/src/lib/confidenceTier.ts` + all 3 branches rendered in `page.tsx` |

**Numbers to quote:** float32 test acc **0.9465**, top-3 **0.9961**; production
dynamic-range TFLite **0.9401**, **1.15 MB**, **22 ms** median latency (XNNPACK).

## A.2 Pending — carried over from the previous plan

1. **Uncommitted work.** Still uncommitted (git is handled manually, D.11): modified
   `web/src/app/page.tsx`, `web/src/lib/diseaseInfo.ts`, `README.md`; new
   `web/src/lib/confidenceTier.ts`, `MODEL_CARD.md`, `src/eval_tiers.py`,
   `outputs/tier_report.json`, `CLAUDE.md`, `plan.md`.
2. ~~**Real-world phone-photo test set**~~ — **DONE (2026-08-23).** 17-image all-classes
   set scored at **0.5294** top-1, plus a 20-image web-sourced set at 0.40 and a
   synthetic corruption sweep. See Phase 1 below and `docs/DOMAIN_SHIFT.md`.
3. ~~**README.md is stale**~~ — **DONE (2026-08-23).** Rewritten against reality:
   Next.js/TS stack, `cd web && npm run dev`, measured tier table, honest
   "real-world testing not yet measured" section, all three `[TODO]`s and the stale
   status table removed.
4. ~~**Link-check** the 17 `referenceUrl`s~~ — **DONE (2026-08-23).** 7 were dead
   (404). All replaced and re-verified; **17/17 now return 200.** Replacements: PSU
   gray-leaf-spot and the CPN S3 PDF → cropprotectionnetwork.org encyclopedia
   entries; PSU potato late blight → UMN late blight; UNH potato/tomato fact sheets →
   UMN growing-potatoes / growing-tomatoes; CSU early blight → UW-Madison early
   blight; PSU septoria → UW-Madison septoria. (Note: `extension.umn.edu` returns 403
   to a bare `curl` — that is bot-blocking, not a dead link; re-check with a browser
   User-Agent.)
5. ~~**Empirical confirmation of the tier system**~~ — **DONE (2026-08-23).** New
   `src/eval_tiers.py` runs the production `.tflite` over the test split →
   `outputs/tier_report.json`:

   | Tier | Coverage | Top-1 acc | Top-3 acc |
   |---|---|---|---|
   | HIGH (≥0.85) | 82.6% | **0.9860** | 0.9990 |
   | MODERATE (0.50–0.85) | 14.9% | 0.7681 | 0.9852 |
   | LOW (<0.50) | 2.5% | 0.4505 | 0.9341 |

   Overall 0.9401 / top-3 0.9953, matching the quoted production number. Accuracy is
   monotone across tiers, so the gate works in the intended direction. **1.16% of all
   test images are HIGH-tier and wrong** — the residue tiering does not catch. The
   sweep (coverage/accuracy at 0.70→0.99) is in the JSON and feeds Phase 2's
   risk–coverage curve. Numbers cited in a comment in `confidenceTier.ts`.

## A.3 Pending — newly identified methodology gaps (the interview-relevant ones)

| Gap | Why it matters in an interview |
|---|---|
| **No baseline or ablation study** | "Why MobileNetV3-Small? Why transfer learning? Did augmentation actually help?" — currently unanswerable with numbers. |
| ~~**No calibration analysis**~~ — **DONE (Phase 2)** | Was: the 0.85 / 0.50 thresholds are magic numbers. Now: T = 0.8878, test ECE 0.0187 → 0.0075, thresholds derived at 0.945 / 0.595, `docs/CALIBRATION.md`. |
| ~~**No explainability**~~ — **DONE (Phase 3)** | Was: "how do you know it looks at the lesion and not the background?" Now: CAM mass sits 1.66× denser on the leaf than area alone (86% of peaks on-leaf, n=200), the brown-lesion cluster is shown to be a discrimination failure with on-leaf attention, and OOD photos are shown latching onto one leaf-shaped blob. `docs/EXPLAINABILITY.md`. |
| **No OOD / not-a-leaf detection** | Known limitation 1 (whole-plant photo → 79.7% confidence on a wrong class) is currently unmitigated. |
| **No tests, no experiment tracking** | `notebooks/` is empty; there is no test suite; results live in prose, not a machine-readable registry. |

---

# PART B — Revised Roadmap

Ordered so each phase produces a standalone, defensible artifact. Phases 0–4 are
the core; 5–6 are the differentiators; 7 is optional.

## Phase 0 — Housekeeping & honesty (half a day)

Cheap, unblocks everything else.

- [ ] Commit the pending work (developer does this manually — D.11).
- [x] **Rewrite `README.md`** to match reality: Next.js/TS frontend (not vanilla
      PWA), `cd web && npm run dev` (not `app/`), treatment panel done, tier system
      done. Stale status table and all three `[TODO]`s removed.
- [x] Link-check the 17 `referenceUrl`s — 7 dead, replaced, 17/17 now 200.
- [x] Add `MODEL_CARD.md`: intended use, training data and its lab-photo bias,
      metrics by class, known failure modes (D.7), explicit "not a substitute for an
      agronomist" statement.
- [x] *(added)* `src/eval_tiers.py` + `outputs/tier_report.json` — the empirical tier
      confirmation from A.2.5, which also front-runs part of Phase 2.

## Phase 1 — Real-world test set + honest generalization number — ✅ **COMPLETE (2026-08-23)**

- [x] Test set assembled: **17 images, one per class, all 17 classes** in
      `real_world_test/`. Sourced from the web rather than shot in the field, labelled by
      folder placement.
- [x] `src/eval_real_world.py` — scores the production `.tflite`, reports per-photo
      prediction/confidence/tier, per-class accuracy, tier coverage, every
      confidently-wrong photo, and the drop vs the 0.9401 baseline →
      `outputs/real_world_report.json`.
- [x] `scripts/check_test_photos.py` — validates a photo set (unrecognised folder names,
      unreadable files, duplicates, undersized images) before scoring. No TF import, ~2s.
      `--fix` canonicalises folder names.
- [x] `scripts/harvest_commons.py` + `web_sourced_test/` — 20 hand-vetted Wikimedia
      Commons photos with per-image licence/attribution in `provenance.json`.
- [x] `src/eval_domain_shift.py` — synthetic corruption stress test, 602 images.
- [x] Write-up: **`docs/DOMAIN_SHIFT.md`**, all three sections.
- [x] `real_world_test/HOW_TO_ADD_PHOTOS.md` — folder routing and vetting rules for
      extending the set.

### Result A — 17-class test set (n=17): **0.5294**

| | This set | Clean test | Drop |
|---|---|---|---|
| Top-1 | **0.5294** (CI [0.31, 0.74]) | 0.9401 | −0.4107 |
| Top-3 | 0.7059 | 0.9953 | −0.2894 |
| Crop identified correctly | 0.882 | — | — |
| Full class correct | 0.588 | — | — |
| **Confidently wrong** | **23.5%** | 1.16% | — |

**Integrity:** all 17 hashed against all 162,916 PlantVillage files → zero duplicates, no
leakage. Resolutions 150×200 to 1200×675. All four confident errors inspected by eye and
confirmed as model errors, not label errors.

Three findings:

1. **Accuracy tracks taxonomic granularity.** Corn (4 classes) **1.000**, potato (3)
   0.667, tomato (10) **0.300**. Crop right 88% vs class right 59% — the model knows what
   plant it is looking at and cannot separate similar lesions within it.
2. **The confidence gate failed here.** HIGH coverage fell only to 58.8% while HIGH
   accuracy fell to 0.600 → **23.5% confidently wrong**, twenty times the benchmark rate.
   This *contradicts* Results B and C, where the model went unconfident under shift and
   the tier system converted that into abstention. **Abstention is a mitigation, not a
   guarantee** — which promotes Phase 2 (calibration) and Phase 4 (OOD gate) from
   nice-to-have to load-bearing.
3. **Healthy leaves handled correctly** — 3/3 right, 0/14 diseased called healthy. This
   *refutes* the Result B hypothesis (3/3 healthy called diseased). Both samples are n=3,
   so neither supports a healthy-bias claim; drop it.

Caveats to quote with the number: n=17, CI [0.31, 0.74], one image per class so every
per-class figure is 0 or 1, web-sourced so labels are placement decisions rather than
plant examinations, and framing skews toward clear illustrative symptom photos.

### Result B — web-sourced photos (n=20), 2026-08-23

Harvested with `scripts/harvest_commons.py`, **vetted one by one visually**, scored via
`--dir web_sourced_test` → `outputs/web_sourced_report.json`.

| | Web-sourced | Clean test | Drop |
|---|---|---|---|
| Top-1 | **0.4000** (CI [0.22, 0.61]) | 0.9401 | −0.54 |
| Top-3 | 0.7500 | 0.9953 | −0.25 |
| Crop identified correctly | 0.80 | — | — |
| HIGH coverage / accuracy | **20%** / 0.750 | 82.6% / 0.986 | — |
| Confidently wrong | 1 of 20 | 1.16% | — |

Here the gate *did* hold — HIGH coverage collapsed 83% → 20% with one confidently-wrong
image. Same crop-right/disease-wrong pattern (80% vs 40%).

**The vetting is the story.** 49 candidates → 20 accepted. Rejected: 6 non-photographs (an
1882 book scan, a disease-cycle diagram, a line drawing, a phylogenetic tree), 2 wrong
species (sweet potato, pepper), **2 with actively wrong labels** (Septoria images returned
for `Tomato___healthy`), 2 wrong plant part, 1 caption/symptom mismatch, 4 whole-plant →
`_ood/`, 12 duplicates. More than half unusable, two mislabelled.

### Result C — synthetic corruptions (602 stratified test images, seed 42)

| | Top-1 | Δ vs clean | Top-3 | HIGH coverage | HIGH acc | Confidently wrong |
|---|---|---|---|---|---|---|
| clean (subsample) | 0.9336 | — | 0.9967 | 80.1% | 0.994 | 0.5% |
| field_composite | 0.7641 | −0.1694 | 0.9352 | 46.0% | **0.971** | 1.3% |
| underexposed | 0.7674 | −0.1661 | 0.9086 | 55.8% | 0.961 | 2.2% |
| background_replace | 0.8721 | −0.0615 | 0.9850 | 61.5% | 0.989 | 0.7% |
| sensor_noise_severe | 0.5399 | **−0.3937** | 0.7990 | 47.3% | 0.761 | **11.3%** |

Full table in `docs/DOMAIN_SHIFT.md` §2 / `outputs/domain_shift_report.json`.

1. **Robustness tracks the augmentation recipe, and nothing else.** Ranking corruptions
   by damage reproduces the contents of `src/augmentation.py` — blur, JPEG, white
   balance, off-angle and background replacement each cost ≤ 0.06. Gaussian sensor noise,
   the one family *not* in the training augmentation, costs 0.39. So the measured
   robustness is evidence of augmentation coverage, not general robustness, and every
   number here is a lower bound. **Action: add a noise transform and re-measure — a
   Phase 5 ablation row, not a guess.**
2. **The tier system degraded gracefully here** (HIGH accuracy held at 0.971 under the
   field composite while coverage collapsed 80% → 46%) — but see Result A, where it did
   not. Two of three distributions is not a guarantee.
3. **Underexposure is the realistic failure mode to worry about** — −0.17 from an
   ordinary shade/indoor shot, whereas overexposure costs nothing.

### What Phase 1 changes about the roadmap

Result A's 23.5% confidently-wrong rate is the single most important number produced by
this phase, and it re-prioritises what follows:

- **Phase 2 (calibration) and Phase 4 (OOD gate) are now load-bearing**, not polish. The
  product's core safety claim — "a false-confident wrong diagnosis is worse than no
  diagnosis, so we abstain" — is only true on some distributions.
- **Phase 5 gains a concrete first ablation row**: add Gaussian noise augmentation and
  re-run Result C.
- **Phase 3 (Grad-CAM) gains a target**: the tomato brown-lesion cluster, where accuracy
  falls to 0.300, is exactly where attention maps would be most informative.

## Phase 2 — Calibration — ✅ **COMPLETE (2026-08-23)**

Write-up: **`docs/CALIBRATION.md`**. Report: `outputs/calibration_report.json`.

- [x] `src/calibration.py` on the **validation** split: ECE / MCE (15 bins), Brier,
      NLL, per-tier accuracy, reliability diagram → `outputs/reliability_diagram.png`.
- [x] **Temperature scaling**, single scalar T fit by NLL minimization on val,
      applied unchanged to test. Shipped in `web/src/lib/calibration.ts`; no
      retraining, no model change (the graph already ends in softmax, so `p**(1/T)`
      renormalized *is* temperature scaling on the hidden logits — verified against
      the Python path to 5.6e-16).
- [x] **Thresholds re-derived from data** → `confidenceTier.ts` 0.945 / 0.595, with
      a comment citing the script and the numbers. Mirrored in `src/config.py`;
      `src/eval_tiers.py` deliberately keeps 0.85 / 0.50 on raw softmax as the
      before-picture.
- [x] **Risk–coverage curve** + AURC → `outputs/risk_coverage.png`.
- [x] *(added)* `--calibrated` on `src/eval_real_world.py`, to test the new gate
      against Phase 1's failure case.

### Result A — the model was under-confident, not over-confident

**T = 0.8878 (< 1, i.e. sharpening).** Mean confidence sat 1.8 points *below*
accuracy on test (0.9223 vs 0.9401) — the opposite of the textbook overconfident-CNN
result. Likely cause: phase-2 fine-tuning at 1e-5 / 5e-6 under heavy realism
augmentation, both of which suppress logit scale. ECE test **0.0187 → 0.0075**
(val 0.0152 → 0.0084); Brier and NLL improve slightly; no prediction changes, since
the transform is monotone.

All-bins MCE *rises* (val 0.2947 → 0.7431) purely because of a one-image bin, so the
report carries `mce_robust` (bins with n ≥ 10) alongside it: 0.1098 → 0.0798.

### Result B — a 0.95 accuracy target is worthless when accuracy is 0.94

Running the plan's suggested "smallest τ meeting 0.95" literally yields τ = 0.510,
98% coverage, and **4.88% confidently wrong** — four times worse than the intuited
0.85 it was meant to replace. A conditional-accuracy target at or near the
unconditional accuracy imposes no abstention at all; raising coverage raises the
absolute count of confident errors. The target is a *product* decision, so the script
prints the full sweep (0.95 → 0.995) to make its price visible. Chosen: **0.99**.

The MODERATE floor needed a different fix: the cumulative "band accuracy ≥ 0.50"
criterion is satisfied down to τ = 0.01 (the band's own high-confidence mass drags the
average up) and erases the LOW tier. Replaced with a **local** criterion — accuracy
inside a 0.05-wide window, ≥ 50 images — which is where the reliability curve actually
crosses a coin flip: **0.595**.

### Result C — shipped: 8 points of coverage for half the confident errors

Held-out test split:

| | HIGH coverage | HIGH top-1 | LOW top-3 | confidently wrong |
|---|---|---|---|---|
| raw, 0.85 / 0.50 | 0.826 | 0.9860 | 0.9341 | **1.16%** |
| calibrated, 0.945 / 0.595 | 0.744 | **0.9930** | 0.9588 | **0.52%** |

Nothing is hidden by the trade — images leaving HIGH land in MODERATE, which still
shows a diagnosis behind a caution banner. AURC 0.0067 (test): abstaining on the least
confident 30% reaches 0.9972, and at 50% coverage there are zero errors in 1,812
images. 0.945 sits at the knee of that curve.

### Result D — it substantially mitigates the Phase 1 failure

Phase 1 Result A's 23.5% confidently-wrong rate was the finding that promoted this
phase to load-bearing. Rescored with `--calibrated`:

| set | | HIGH coverage | HIGH accuracy | confidently wrong |
|---|---|---|---|---|
| 17-class photos (n=17) | before | 0.588 | 0.600 | **23.5%** |
| | after | 0.353 | 0.833 | **5.9%** |
| Commons (n=20) | before | 0.200 | 0.750 | 5.0% |
| | after | 0.100 | 1.000 | **0%** |

Top-1 accuracy is unchanged (0.5294 / 0.4000) — calibration makes uncertainty legible,
it cannot make the model right. **Caveat hard: n=17 and n=20, so "before" is 4 images
and 1 image.** The one surviving confident error
(`Tomato___Bacterial_spot → Septoria_leaf_spot`, 0.971) is inside the tomato
brown-lesion cluster — a discrimination failure no threshold can reach, which is
Phase 3 and Phase 4 territory.

### What Phase 2 changes about the roadmap

- **Phase 4 (OOD) stays load-bearing; Phase 2's own limits point straight at it.**
  Calibration answers "how confident, given a leaf" and says nothing about a photo of
  a chair.
- **A per-class calibration check is now the obvious cheap follow-up.** One scalar T
  cannot fix per-class miscalibration, and with 35× imbalance that likely exists.
- **Phase 5 gains a metric.** Every ablation row should report ECE and AURC next to
  accuracy — the calibration harness already computes both from a probability matrix.

*Concepts: softmax is not a probability, ECE and reliability diagrams, temperature
scaling, selective prediction / abstention, precision-coverage tradeoff.*

## Phase 3 — Explainability: Grad-CAM — ✅ **COMPLETE (2026-08-23)**

- [x] `src/explain.py` — Grad-CAM over the Keras model at two layers (deep `activation_17`
      7×7, and mid `activation_11` 14×14), with the softmax head re-applied inside the
      tape to recover true logits (asserted against `model.predict()`, max|Δ| = 1.2e-07).
      Panels for five cohorts → `outputs/gradcam/` (57 panels): `correct_high`,
      `cluster` (rendered for predicted *and* true class), `real_world`, `web_sourced`,
      `ood`.
- [x] Background-leakage question answered and quantified: CAM mass inside the
      `_segment_leaf` GrabCut mask, normalised by the mask's own area
      (`lift`), over a 200-image random test sample split by correct/wrong.
- [x] `docs/EXPLAINABILITY.md`.
- [ ] *Stretch (not done):* export a second `.tflite` with the conv feature map as an
      extra output and render the heatmap in the browser.

### Result A — no background shortcut (n = 200 test images, seed 42)

| group | n | leaf area | CAM mass in leaf | **lift** | peak in leaf |
|---|---|---|---|---|---|
| all | 200 (197 usable masks) | 0.387 | 0.625 | **1.66** | **0.858** |
| correct | 190 | 0.390 | 0.629 | 1.66 | 0.861 |
| wrong | 10 | 0.335 | 0.551 | 1.56 | 0.800 |

Mid layer: lift 1.39 overall, peak-in-leaf 0.61 (correct 0.63 / wrong 0.30).

`BackgroundReplace`'s premise was that the studio background *could* be learned. It was
not: attention is 1.66× denser on the leaf than area alone would give, which lines up with
`background_replace` costing only 0.0615 accuracy in the Phase 1 corruption sweep. **The
domain-shift collapse is not background reliance.** Whether the augmentation *caused* this
is a Phase 5 ablation, not something these maps can settle.

Attention also does **not** flag its own errors — the correct/wrong gap is consistent in
direction across both layers but tiny, on n = 10 wrong images. Not an abstention signal at
this sample size; the mid-layer gap (1.40 → 1.10) is the one worth re-measuring on a few
hundred errors.

### Result B — the three open failures now have explanations

1. **Brown-lesion cluster = discrimination failure, not leakage.** On cluster errors the
   peak is on-leaf 87.5% of the time and deep lift is 1.32. Rendering predicted and true
   class side by side gives two plausible on-leaf maps over *different* lesion regions.
   Brown spots versus brown spots at a 7×7 grid → resolution/capacity, i.e. Phase 5, not
   data cleanup.
2. **The two surviving confident errors are class confusion with correct localisation.**
   `Potato → Tomato Late blight` (0.957) puts both class maps on the same necrotic lesion —
   right pathology, wrong host. `Bacterial_spot → Septoria` (0.977, the error
   `docs/CALIBRATION.md` §5 calls unreachable by any threshold) has near-identical maps for
   both classes on the same spotted lamina.
3. **OOD photos: the map picks a leaf-shaped blob and commits.** Deep lift 0.38, peak never
   inside the mask. One whole-plant tomato scene scores **0.983 calibrated — above the
   shipped HIGH gate**. Phase 4 stays load-bearing, and the maps suggest a gate feature:
   OOD maps are diffuse or a small blob far from any leaf mask, in-distribution maps are
   lesion-locked.

### Result C — attention survives the shift that accuracy does not

| cohort | n | acc (Keras) | mean conf | deep lift | mid lift |
|---|---|---|---|---|---|
| correct_high | 8 | 1.000 | 0.987 | 1.58 | 1.16 |
| cluster | 8 | 0.250 | 0.685 | 1.32 | 0.79 |
| real_world | 17 | 0.588 | 0.734 | 1.56 | 1.21 |
| web_sourced | 20 | 0.350 | 0.665 | 1.50 | 1.25 |
| ood | 4 | — | 0.690 | 0.38 | 0.27 |

Accuracy falls 1.00 → 0.35 while deep lift barely moves — the model keeps finding the
diseased tissue and keeps mislabelling it. Consistent with Phase 1's "crop right, disease
wrong".

### Side findings

- **Grad-CAM at the final layer is provably plain CAM here.** With GAP + one Dense layer,
  `d logit_c / d A[i,j,k] = W[k,c]/49`, constant in space — verified numerically (gradient
  spatial variance 3.1e-17; channel weights match `W[:,c]/49` to 3.7e-09). Hence the
  second 14×14 map, where the gradient actually varies.
- **The shipped quantized model and the float model disagree under shift.**
  `keras_tflite_agree`: 0.980 on the clean test sweep, but 0.824 / 0.800 / 0.750 on
  real_world / web_sourced / cluster. So the domain-shift numbers are properties of the
  *quantized artifact*, not the architecture — a cheap unmeasured experiment — and the
  heatmaps explain the float model, not always what the browser computed.
- **JPEG decoder sensitivity.** cv2 and tf.io decode the same JPEG up to ~4 grey levels
  apart, enough to flip a borderline cluster image from Target Spot 0.488 to Spider mites
  0.662. `src/explain.py` uses `tf.io` to stay consistent with the rest of the repo. Good
  argument for the Phase 6 preprocessing-parity test.

### What Phase 3 changes about the roadmap

- **Phase 4 (OOD) is confirmed as the highest-value remaining item**, and now has a
  candidate feature beyond MSP/energy/Mahalanobis: CAM dispersion.
- **Phase 5 gains two rows**: train-without-`BackgroundReplace` (to test whether the
  augmentation caused Result A), and a float-vs-quantized comparison on the shifted sets.
- **Phase 6 gains a concrete test**: decoder/preprocessing parity, which is now known to
  change predictions and not merely pixels.
- A hand-annotated leaf-mask set (~30 images) would upgrade Result A from indicative to
  solid; the GrabCut mask is the weakest link in the measurement.

*Concepts: CAM / Grad-CAM mechanics, shortcut learning and spurious correlation,
qualitative error analysis.*

## Phase 4 — OOD gate: "is this even a leaf?" (1–2 days)

Directly fixes the project's worst known limitation.

- [ ] Build a small negative set (~100 images): faces, furniture, sky, text, whole
      plants, blank walls. Store in `real_world_test/_ood/`.
- [ ] New `src/ood.py` comparing three detectors on penultimate-layer embeddings /
      logits, scored by **AUROC and FPR@95TPR**:
      1. Maximum Softmax Probability (the baseline everyone quotes)
      2. **Energy score** (`-logsumexp(logits)`) — usually beats MSP, and is cheap
      3. Mahalanobis distance to per-class embedding means
- [ ] Ship the winner as a **pre-classification gate** in the UI: below threshold →
      "this doesn't look like a single crop leaf — retake the photo," and skip the
      diagnosis entirely. Requires exporting logits or the embedding as a second
      TFLite output.
- [ ] Record the AUROC table in `docs/OOD.md`.

*Concepts: open-set recognition, OOD detection, AUROC / FPR@95, why closed-set
softmax classifiers fail on unseen input classes.*

## Phase 5 — Baselines & ablations: the table that answers "why?" (2–3 days)

Every architecture and data decision currently rests on assertion. This produces
numbers. Each run is short (~17k images at 224², and training already works), so
this is throughput, not difficulty.

- [ ] New `src/experiments.py` — a small runner writing every run to
      `outputs/experiments.json` (config hash, params, val/test acc, train time,
      model size) and rendering `outputs/experiments_table.md`.
- [ ] **Baselines:**
      - Majority class, and "most common class per crop" (the sanity floor)
      - Small CNN trained from scratch (shows what transfer learning buys)
      - MobileNetV2 and EfficientNet-Lite0 at the same budget (shows why V3-Small)
- [ ] **Ablations on MobileNetV3-Small:**
      - no augmentation / standard augmentation only / plus `BackgroundReplace`
        (**does the realism augmentation actually pay off?**)
      - with vs without class weights (does `Potato_healthy` recall collapse?)
      - frozen backbone only vs two-phase fine-tuning
- [ ] **The quantization capstone:** train **MobileNetV3-Small "minimalistic"** (no
      squeeze-excite, plain ReLU instead of hard-swish) and run full INT8 PTQ on it.
      The existing investigation *predicted* this variant would survive INT8 where
      the standard one collapsed to 0.7630. Confirming or refuting that with a number
      turns a debugging write-up into a validated hypothesis — the strongest single
      result the project could add.

*Concepts: baselines before SOTA, controlled ablation, hypothesis → experiment →
result discipline, architecture/quantization interaction.*

## Phase 6 — Engineering rigor (1 day, can interleave)

- [ ] `pytest` suite: split determinism and no train/test leakage; augmentation
      output shape and range; class-name order parity between the Python-derived
      labels and `web/src/lib/constants.ts` (**the contract most likely to break
      silently**); TFLite output vs Keras output on fixed inputs.
- [ ] A cross-language preprocessing parity test — run `preprocess.ts` under node
      against the Python pipeline on a fixed image, assert max-abs-diff < 1e-4.
      Currently verified manually; make it automatic.
- [ ] `npm run lint` plus `pytest` in CI (GitHub Actions), CPU-only, skipping
      anything that needs the dataset.

## Phase 7 — Optional stretch (only if time remains)

- Scan history via IndexedDB; kNN "nearest training examples" explanation using the
  exported embedding (pairs naturally with Phase 4's embedding export).
- Expand to all 38 PlantVillage classes — a data-filter change, not an architecture
  change; good material for a "how does this scale" answer.
- Re-add PWA / service-worker offline support (the originally planned scope).

---

# PART C — Interview Concept Map

What to be ready to whiteboard, and which artifact in this repo backs it up.

| Concept | Backed by |
|---|---|
| Transfer learning, layer freezing, two-phase fine-tuning, why BN stays frozen | `src/train.py`, Phase 5 ablation |
| Class imbalance — class weights vs resampling vs focal loss, and why per-class recall is the metric that shows it worked | `outputs/classification_report.json`, Phase 5 ablation |
| Data augmentation and closing a domain gap | `src/augmentation.py` `BackgroundReplace`, verified by the Phase 3 CAM check (lift 1.66) |
| Stratified splitting, leakage, reproducible seeds | `data/dataset_split.csv`, Phase 6 tests |
| Metrics beyond accuracy — precision/recall/F1, top-k, confusion structure | `src/evaluate.py` |
| **Calibration** — ECE, reliability diagrams, temperature scaling | `src/calibration.py`, `docs/CALIBRATION.md`, `outputs/reliability_diagram.png` |
| **Selective prediction / abstention**, risk–coverage | `outputs/risk_coverage.png` (AURC 0.0067) plus the derived thresholds in `confidenceTier.ts` |
| **OOD / open-set** — MSP vs energy vs Mahalanobis, AUROC | Phase 4 |
| **Explainability** — Grad-CAM, shortcut learning | `src/explain.py`, `docs/EXPLAINABILITY.md` (Phase 3) |
| **Baselines and ablations** — experimental discipline | Phase 5 |
| Quantization — PTQ dynamic-range vs full INT8, weights vs activations, QAT, architecture sensitivity (hard-swish, SE blocks) | `docs/quantization_findings.md`, Phase 5 capstone |
| Edge / on-device deployment — model size, latency, XNNPACK, WASM, preprocessing parity | `web/src/lib/`, `src/benchmark_pipeline.py` |
| Distribution shift, benchmark vs field accuracy, corruption robustness | `docs/DOMAIN_SHIFT.md`, `src/eval_domain_shift.py`; field number pending Phase 1 photos |
| MLOps hygiene — experiment tracking, model cards, CI, contract tests | Phases 0, 5, 6 |

**The three stories to be able to tell cold:**

1. *"INT8 collapsed my accuracy and I found out why"* — the full hypothesis chain in
   `docs/quantization_findings.md`, ending in the Phase 5 minimalistic-variant
   confirmation.
2. *"My confidence numbers were lies until I measured them"* — Phase 2 calibration:
   the model turned out *under*-confident (T = 0.8878, not > 1), the plan's own 0.95
   accuracy target proved worthless below a 0.94 unconditional accuracy, and the
   thresholds that shipped (0.945 / 0.595) halve the confident-error rate. Ends in
   `docs/CALIBRATION.md` and code in `web/src/lib/calibration.ts`.
3. *"94.65% on the test set, X% on photos I took myself"* — Phase 1 domain shift,
   with the Phase 4 OOD gate as the mitigation.

---

# PART D — Reference: Implementation Details

## D.1 Tech stack

**ML (Python):** TensorFlow 2.21.0 / Keras 3.15.0; MobileNetV3-Small
(`tf.keras.applications`, ImageNet pretrained); albumentations 2.0.8;
opencv-python-headless 4.11.0.86 (pinned — 5.0 is blocked by a Windows Application
Control policy); scikit-learn 1.9.0. Full pins in `requirements.txt`.

**Frontend:** Next.js 16 (App Router) + React 19 + TypeScript + Tailwind 4;
**LiteRT.js** (`@litertjs/core`) — Google's browser WASM runtime for `.tflite`,
successor to the abandoned `tfjs-tflite`, running XNNPACK on CPU. No backend.

**Dev env:** Windows locally (primary). Colab T4 was attempted and abandoned for
training due to setup friction — training happened locally on CPU.

## D.2 Repository layout

- `src/` — Python pipeline. Run as packages from the repo root (`python -m
  src.train`), not as file paths.
- `web/` — **current** frontend (Next.js/TS).
- `app/` — **legacy** vanilla HTML/JS LiteRT harness. Keep as reference; new work
  goes in `web/`.
- `docs/` — `ARCHITECTURE.md`, `quantization_findings.md`, `DOMAIN_SHIFT.md`.
- `real_world_test/` — hand-taken field photos for `src.eval_real_world`; **empty on purpose**,
  never populate with stock/internet images.
- `scripts/investigations/` — one-off diagnostics from the quantization hunt; not
  part of the pipeline.
- `data/`, `venv/` gitignored; `models/`, `outputs/` committed. `notebooks/` empty.

## D.3 Model & data

- **Dataset:** PlantVillage `color` (unsegmented) — deliberately not the
  background-removed variant, since real photos have natural backgrounds.
- **Classes (17):** Corn — Cercospora/Gray leaf spot, Common rust, Northern Leaf
  Blight, healthy. Potato — Early blight, Late blight, healthy. Tomato — Bacterial
  spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites,
  Target Spot, Yellow Leaf Curl Virus, mosaic virus, healthy.
- **Split:** stratified 70/15/15 → 16,914 / 3,625 / 3,625, saved to
  `data/dataset_split.csv`.
- **Imbalance:** ~35x — `Potato_healthy` (152) vs `Tomato_Yellow_Leaf_Curl_Virus`
  (5,357). Handled via `sklearn.compute_class_weight('balanced', ...)`.
- **Augmentation** (train split only): affine rotate/scale (migrated from the
  deprecated `ShiftScaleRotate`), flips, brightness/contrast, color-temperature
  jitter; plus realism transforms — motion blur, JPEG artifacts, `CoarseDropout`;
  plus the custom **`BackgroundReplace`** (GrabCut leaf segmentation at 128px for
  ~3x speed, feathered alpha blend, composited onto procedural soil/grass/wood/hand
  textures, p=0.4). Applied via `unbatch()` → `tf.numpy_function` → `batch()`.
- **Model:** MobileNetV3-Small (`include_top=False`) → `GlobalAveragePooling2D` →
  `Dropout(0.3)` → `Dense(17, softmax)`. 948,929 params, 3.62 MB pre-quantization.
- **Training:** Phase 1 frozen backbone, `Adam(1e-3)`, EarlyStopping plus
  ReduceLROnPlateau on val_accuracy → 0.8908. Phase 2 unfroze the last ~25% (first
  117/157 layers frozen, BN frozen throughout), `Adam(1e-5)` → 0.9454, then resumed
  15 epochs at `Adam(5e-6)` → **0.9481**.
- **Float32 results:** test **0.9465**, top-3 **0.9961**, no overfitting signal.

## D.4 The quantization investigation — settled findings, do not redo

Read `docs/quantization_findings.md` first.

1. Full INT8 PTQ collapsed accuracy: 0.9465 → **0.7630**.
2. Hypothesis: uniform calibration sampling starved rare classes (`Potato_healthy`
   got 1 of 400 calibration images).
3. Stratified calibration (30/class) made it **worse**: → **0.6836**.
4. Bug hunt on that surprising direction: channel order (RGB confirmed via
   `decode_jpeg`), normalization, `.iloc`/`.loc` indexing, pixel ranges — all
   correct, **no bug**.
5. Dynamic-range quantization (weights-only INT8, float32 activations, no
   calibration) as a diagnostic: **0.9401** — weights quantize fine; the problem is
   specifically **activation** quantization.
6. **Root cause:** MobileNetV3's hard-swish and squeeze-excite blocks are genuinely
   sensitive to per-tensor INT8 activation quantization. Documented phenomenon — the
   MobileNetV3 paper's own "minimalistic" variant (no SE, plain ReLU) exists for this
   reason. **Phase 5 tests that prediction directly.**
7. QAT blocked: `tensorflow-model-optimization` 0.8.1 requires legacy `tf_keras`, not
   native Keras 3.
8. **Decision: shipped dynamic-range as production.**

**Production artifact `models/cropguard_v1_production.tflite`:** dynamic-range,
**float32 in/out** (no scale/zero-point math downstream), **static batch size 1**
(LiteRT.js's JS wrapper doesn't handle TFLite's dynamic `-1` batch dim — fixed by
re-exporting the Keras model with a fixed `batch_size=1` Input before conversion),
1.15 MB, **0.9401** accuracy, **22 ms** median latency.

## D.5 Evaluation findings

- Class weighting **worked**: `Potato_healthy` (152 train images) → precision 0.88 /
  recall 0.96; `Tomato_mosaic_virus` (373) → recall 1.00. Neither rare class sits at
  the bottom of the per-class F1 ranking.
- Confusion is concentrated and explainable, not noise:
  - Tomato brown-lesion cluster: Early blight ↔ Target Spot ↔ Septoria ↔ Spider mites
  - One corn pair: Northern Leaf Blight → Gray leaf spot
- Full metrics: `outputs/classification_report.json`; matrix:
  `outputs/confusion_matrix.png`.

## D.6 Frontend implementation

- **LiteRT.js lifecycle** lives entirely in `web/src/lib/useCropGuardModel.ts`: a
  **module-level init promise** guards React StrictMode double-mount, the compiled
  model lives in a ref (not state), and input/output `Tensor`s are explicitly
  `.delete()`d to avoid WASM heap leaks. Client-only (`'use client'` plus
  `useEffect`) — SSR cannot touch WASM/browser APIs.
- **WASM binaries vendored** in `web/public/litert_wasm/`, served from
  `/litert_wasm/` — never from a CDN.
- **Preprocessing (do not "optimize"):** 224×224 canvas resize with
  `imageSmoothingQuality = 'medium'` (bilinear — matches TF's default
  `tf.image.resize`, NOT `'high'`/bicubic), normalize to [0,1] float32. Verified
  numerically against Python.
- **Regression check — re-run after ANY change to preprocessing, model loading, or
  the inference path.** This has caught real bugs:

  | Image | Expected (raw softmax) | Expected (calibrated, what the UI shows) |
  |---|---|---|
  | `Corn_(maize)___Common_rust_/RS_Rust 2370.JPG` | 99.9954% Corn_Common_rust | ~99.998% |
  | `Potato___healthy/...RS_HL 1951.JPG` | 99.8074% Potato_healthy | ~99.91% |
  | `Tomato___Leaf_Mold/...Crnl_L.Mold 7082.JPG` | 99.7541% Tomato_Leaf_Mold | ~99.90% |

  Since Phase 2 the UI displays the **calibrated** value (T = 0.8878), so the third column
  is the one to compare against on screen; the raw column is still visible in the debug
  panel. The calibrated figures come from the Python path — expect agreement to a few
  decimals, not to the last digit.

- **Input methods:** file upload, camera (`<input type="file"
  capture="environment">` — the native camera app, not a live `getUserMedia()`
  preview), drag-and-drop. All three funnel into one preprocess/infer path.
- **Treatment panel** (`web/src/lib/diseaseInfo.ts`): static, typed, 17 entries —
  cause, prevention bullets, general treatment category (deliberately no specific
  chemical dosing), and a `.edu` extension link. Researched against real university
  extension sources.
- **Confidence tiers** (`web/src/lib/confidenceTier.ts`, gated in `page.tsx`):
  - HIGH (≥85%): full diagnosis plus treatment panel
  - MODERATE (50–85%): caution banner, top-3 alternatives, treatment with a caveat
  - LOW (<50%): no headline, **no treatment panel** (showing it would contradict the
    uncertainty), top-3 as "possible matches", suggests retaking or consulting an
    expert
  - Implements the core safety principle: **a false-confident wrong diagnosis is
    worse than no diagnosis.** Phase 2 replaces these thresholds with measured ones.
- **`web/next.config.ts`** sets `logging.browserConsole: false` because the
  Emscripten runtime writes via low-level `_fd_write`, which floods the Turbopack
  terminal. `allowedDevOrigins` holds a hardcoded LAN IP for phone testing — update
  it per network.
- **`web/AGENTS.md`** (auto-regenerated by `next dev`) requires reading
  `web/node_modules/next/dist/docs/` before writing Next.js code — this is Next 16,
  which differs from older conventions.

## D.7 Known limitations (for README and honest self-assessment)

1. **OOD inputs can be confidently wrong.** A whole-plant photo (multiple leaves,
   fruit, garden background) returned 79.74% on the *wrong* class (`Corn_healthy`).
   The model saw only single-leaf close-ups; it has no "whole plant" category.
   → **Phase 4 addresses this.**
2. **Confidence tiers help but do not fully solve false confidence.** A stock photo
   (unverified label) scored 93.65% — above the old HIGH threshold. Tiering catches
   genuine model uncertainty, not confidently-wrong predictions on inputs that merely
   *resemble* the training distribution. Phase 2 improved this a lot — confidently
   wrong fell 1.16% → 0.52% on test and 23.5% → 5.9% on the 17 real-world photos —
   but one confident error survives there, inside the tomato brown-lesion cluster.
   → **Phase 4 attacks the remaining half.**
3. **Only 3 of 14 PlantVillage crops covered** — deliberate scope decision;
   expanding is a data-filter change, not an architecture change.
4. **Real-world accuracy is not yet measured** with a trustworthy self-labeled set.
   Two anecdotal tests exist but are not data (one had no verified ground truth, one
   was OOD input). Two stand-ins now bound it: **0.40 top-1 on 20 hand-vetted
   web-sourced photos** (CI [0.22, 0.61]), and 0.7641 under stacked synthetic
   corruption vs 0.9336 clean. → **Phase 1 §1, still blocked on hand-taken photos.**
5. **No baselines or ablations yet**, so architecture and augmentation choices rest
   on assertion rather than measurement. → **Phase 5.**
6. **Robust only to the corruptions it was augmented with.** Gaussian sensor noise —
   the one family absent from `src/augmentation.py` — costs 0.39 accuracy and pushes
   11.3% of images into confidently-wrong, versus ≤ 0.06 for every augmented family.
   → add a noise transform as a **Phase 5** ablation row.

## D.8 Commands

Python (repo root, venv activated; no test suite yet — Phase 6 adds one):

```bash
python -m src.data_pipeline   # scan dataset -> data/dataset_split.csv
python -m src.train           # two-phase transfer learning -> models/cropguard_v1.keras
python -m src.evaluate        # test/top-3 acc, confusion matrix -> outputs/
python -m src.quantize        # TFLite conversion + accuracy/latency report
python -m src.eval_tiers      # confidence-tier report -> outputs/tier_report.json
python -m src.calibration     # ECE/temperature/thresholds -> outputs/calibration_report.json
python -m src.eval_domain_shift  # corruption stress test -> outputs/domain_shift_report.json
python -m src.eval_real_world    # scores real_world_test/ (blocked: needs hand-taken photos)
python -m src.augmentation    # augmentation previews -> outputs/
python -m src.benchmark_pipeline
```

`src/quantize.py` **skips conversion if the target `.tflite` already exists** —
delete the file to force re-conversion.

Web:

```bash
cd web
npm run dev     # next dev
npm run build
npm run lint
```

## D.9 The model/web contract (breaks silently — Phase 6 adds tests)

Four things must stay in sync between Python and `web/src/lib/`:

1. **Class index order.** Python uses `sorted(df['label'].unique())` over
   PlantVillage folder names; `constants.ts` `CLASS_NAMES` is the same order with
   *display-sanitized* names. Editing one list without the other silently mislabels
   every prediction.
2. **Preprocessing.** `preprocess.ts` reproduces bilinear resize plus `/255.0`.
3. **Model artifact.** `models/cropguard_v1_production.tflite` is copied to
   `web/public/` **and** `app/public/` — regenerating means recopying all copies.
4. **Accelerator.** Dynamic-range quantized (float32 activations), loaded with
   `{ accelerator: 'wasm' }` (XNNPACK). Do not switch the converter to full-integer
   quantization expecting it to work (see D.4).

## D.10 Environment quirks

- TF ≥ 2.11 has no native Windows GPU support. A local Windows Application Control
  policy also blocks some native DLLs used during TFLite calibration, so conversion
  can fail locally while inference and eval work. **If a DLL-load error appears,
  check `requirements.txt` for an already-known pinned-version workaround before
  debugging from scratch.**
- `tensorflow-model-optimization` 0.8.1 needs legacy Keras 2 and crashes on this
  Keras 3 project — QAT unavailable.
- Models save as `.keras`, not `.h5`: Keras 3 cannot round-trip a MobileNetV3
  backbone from `.h5` (hard-swish fails on load).
- Albumentations params are version-sensitive; keep to the `requirements.txt` pins.

## D.11 Working agreements

- **Read `docs/quantization_findings.md` and `docs/ARCHITECTURE.md` before touching
  model or quantization code.** Do not re-litigate settled findings without a
  specific new reason.
- **The 3 reference images (D.6) are the standard regression check.**
- **Git operations are handled manually by the developer** — do not put git
  instructions in task prompts or scripts unless asked.
- **Show real output and numbers; do not assert that something "should" work.**
  Verify by running, not by code review.
