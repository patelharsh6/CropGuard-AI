# Out-of-distribution detection — the "is this even a leaf?" gate

**Phase 4.** Produced by `python -m src.ood` (report: `outputs/ood_report.json`,
plots: `outputs/ood_scores.png`, `outputs/ood_roc.png`) and shipped through
`src/export_ood_model.py` → `models/cropguard_v1_gate.tflite` +
`models/ood_gate.json` → `web/src/lib/oodGate.ts`.

---

## 1. The problem this fixes

The classifier is **closed-set**. Softmax over 17 leaf classes sums to 1 whatever
arrives, so every input gets a disease name — a chair, a face, a whole tomato plant.
This is the project's worst known limitation (README limitation 1), and neither of
the earlier phases touches it:

- **Calibration (Phase 2)** answers *"how confident, given a leaf"*. Temperature
  scaling is a monotone transform of the same softmax; it cannot represent "none of
  the above".
- **Explainability (Phase 3)** found the smoking gun: on OOD photos the Grad-CAM
  picks a leaf-shaped blob and commits, and one whole-plant tomato scene scored
  **0.983 calibrated — above the shipped HIGH gate of 0.945**.

Measured on the negative set built for this phase, **9.9% of non-leaf photos reach
the HIGH tier** (full diagnosis + treatment panel) and **62.4% reach MODERATE or
above** (a diagnosis is displayed). The maximum calibrated confidence on a non-leaf
image is **0.9998**.

## 2. The negative set

97 hand-vetted Wikimedia Commons photos in `real_world_test/_ood/`, plus the 4
whole-plant images already in `web_sourced_test/_ood/` — **n = 101** across 15
categories:

| | |
|---|---|
| Easy (the app is clearly being misused) | face 6, hand 6, furniture 6, device 7, text 6, sky 7, wall 5, vehicle/street 6, animal 8, soil/ground 6 |
| **Hard** (plant material, not a single crop leaf) | whole_plant_crop 12, other_foliage 6, flower 7, fruit_veg 9 |
| From `web_sourced_test/_ood` | uncategorised 4 |

Built by `scripts/harvest_ood.py` (Commons API, per-category queries) and
`scripts/curate_ood.py`. Every candidate was viewed in a contact sheet;
**19 of 116 were rejected** and the reasons are recorded per image in
`real_world_test/_ood/provenance.json` — mostly seed-catalogue engravings and
paintings that the plant queries rank highly on Commons, plus near-duplicates that
would have let one scene dominate a category rate. Images are stored downscaled to a
640 px long edge (the pipeline resizes to 224) because this directory is committed.

Internet images are acceptable *here* although `real_world_test/README.md` forbids
them for the labelled set: the label is "not a single crop leaf", which is verifiable
by looking, and a negative cannot leak the training distribution. Titles already used
by `web_sourced_test/` are skipped so the two sets stay disjoint.

## 3. The detectors

All four are computed on the **Keras float model** — the shipped artifact ends in
softmax and exposes neither logits nor embeddings.

| | needs | note |
|---|---|---|
| **MSP** — max softmax probability | nothing extra | the baseline everyone quotes |
| **Energy** — `logsumexp(logits)` | logits | keeps the logit magnitude softmax normalizes away |
| **Mahalanobis** — distance to the nearest class Gaussian, shared covariance (Ledoit-Wolf, shrinkage 0.0302) | 576-d embedding | fitted on 2,026 train images (120/class) |
| **Class-mean cosine** — `max_c cos(embedding, mean_c)` | 576-d embedding | **not in the Phase 4 plan**; added after the results below |

Sign convention: higher = more in-distribution, so ID is the positive class and the
gate rejects *below* a threshold.

One thing worth stating because it looks like an oversight: **energy cannot be
computed from the production artifact at all.** Calibration got away with recovering
logits as `log(p)` because softmax is invariant to an additive constant — but that is
exactly the constant energy measures. `logsumexp(log p) == 0` for every input, always.
Energy needs a real logits output, which is why `src/export_ood_model.py` exists.

## 4. Result A — the clean-split AUROC is a trap

The obvious experiment is ID test vs OOD. Run it and the embedding detectors look
finished:

| detector | AUROC (clean test vs OOD) | FPR@95TPR |
|---|---|---|
| MSP | 0.9100 | 0.5446 |
| energy | 0.9275 | 0.3861 |
| Mahalanobis | **0.9997** | **0.0000** |
| class-mean cosine | **0.9997** | **0.0000** |

Perfect separation, zero false positives. Then apply the textbook threshold — 95% TPR
on the clean validation split — to the 37 real field photos, which are **in-scope
inputs a gate must accept**:

| detector | threshold | keeps of clean test | **keeps of field leaves** | accepts of OOD |
|---|---|---|---|---|
| MSP | 0.6165 | 0.948 | 0.676 | 0.505 |
| energy | 2.8793 | 0.941 | 0.514 | 0.366 |
| Mahalanobis | −997.97 | 0.940 | **0.000** | 0.000 |
| class-mean cosine | 0.7802 | 0.943 | **0.054** | 0.000 |

**The 0.9997-AUROC detector rejects every single real photograph of a leaf.** It was
never detecting leaves. Every training image is a detached leaf on a uniform grey
studio card, so "in-distribution" as measured on the clean split means *studio*, and
a field photo is as foreign to it as a chair.

This is the central result of the phase, and it generalises past this project: when
the training distribution is narrower than the deployment distribution, OOD AUROC
against the training distribution measures the wrong thing.

## 5. Result B — re-asking the question, and a detector that was not in the plan

The honest comparison is **field leaves vs OOD**: both are photographs taken outside
the studio, and only one should be accepted.

| detector | AUROC clean vs OOD | **AUROC field vs OOD** | params to ship |
|---|---|---|---|
| MSP | 0.9100 | 0.5949 | 0 |
| energy | 0.9275 | 0.6313 | 0 |
| Mahalanobis | 0.9997 | 0.9649 | 341,568 (1.33 MB) |
| **class-mean cosine** | 0.9997 | **0.9711** | **9,792 (38 KB)** |

MSP and energy fall to near-chance. They are functions of the logits, and the logits
are what domain shift damages: a field photo produces a low-magnitude, flat logit
vector that looks exactly like a photo of a wall.

The embedding detectors survive, and the **cheaper one wins**. Dropping the
covariance and L2-normalizing is not a compromise here, it is the fix: a field leaf's
embedding points in nearly the same *direction* as a studio leaf's but with a
different magnitude, and Mahalanobis charges for that magnitude while cosine does
not. The 1.3 MB precision matrix would have more than doubled the 1.15 MB the browser
downloads; 17 unit class-mean vectors cost 38 KB of float32 (96 KB as the shipped
`ood_gate.json`, since it is JSON text rather than a binary blob).

### The PCA trap, again

The obvious way to shrink Mahalanobis is to project the embedding first. It preserves
the clean number almost exactly and destroys the real one:

| variant | AUROC clean vs OOD | AUROC field vs OOD | params |
|---|---|---|---|
| PCA-32 + Mahalanobis | 0.9943 | 0.6387 | 20,576 |
| PCA-64 + Mahalanobis | 0.9981 | 0.7367 | 42,624 |
| PCA-128 + Mahalanobis | 0.9996 | 0.8250 | 92,864 |
| full Mahalanobis | 0.9997 | 0.9649 | 341,568 |
| class-mean cosine | 0.9997 | 0.9711 | 9,792 |

The leading principal components of the embedding carry the studio distribution; the
information that separates a leaf from a non-leaf lives in the tail. Anyone tuning
this against the clean split alone would have shipped PCA-64 and a gate that works on
nothing real.

## 6. What ships

**Detector:** class-mean cosine. **Threshold: 0.5981**, set to keep 90% of the 37
field photos — *not* on the clean split.

At that operating point (float model):

| population | n | accepted |
|---|---|---|
| clean test (studio leaves) | 3,625 | 0.999 |
| field leaves — `real_world_test` | 17 | 0.941 |
| field leaves — `web_sourced_test` | 20 | 0.850 |
| **OOD negatives** | 101 | **0.079** |

**Effect on the UI**, over the 101 negatives:

| | before the gate | after |
|---|---|---|
| reach HIGH tier (diagnosis + treatment panel) | 9.9% | **2.0%** |
| reach MODERATE or above (a diagnosis is shown) | 62.4% | **5.9%** |

### What still gets through, and why it makes sense

Eight of the 101 negatives score above 0.5981, and the list is short enough to read
in full:

| score | category | image |
|---|---|---|
| 0.669 | other_foliage | new oak leaves with female flowers |
| 0.644 | wall | Roman wall-plaster fragment, photographed on white |
| 0.626 | whole_plant_crop | close shot of a bush tomato plant |
| 0.618 | other_foliage | oak tree leaves |
| 0.611 | fruit_veg | stewed milkfish with tomatoes and eggplant |
| 0.607 | animal | sleeping cat |
| 0.603 | other_foliage | red oak autumn leaves, close-up |
| 0.601 | fruit_veg | vegetable stall |

By category: other_foliage 3/6, fruit_veg 2/9, wall 1/5, animal 1/8,
whole_plant_crop 1/12, and **0/56 across the other ten categories**.

Half of them are the expected failure: **the gate rejects non-leaves and accepts
leaves, but does not know *whose* leaf**. Oak foliage is a leaf close-up, and a
17-class head has no way to say "leaf, but not one of mine" — that is species
recognition, not OOD detection, and would need a crop-species class rather than a
better score. The bush-tomato shot is a genuinely borderline call: it is close enough
that a user might reasonably expect a diagnosis.

The other three are honest false negatives with no such excuse — a plaster fragment
(the one clue being that it is an isolated object on a plain background, structurally
the same framing as the studio training images), a cat, and a market stall.

## 7. Shipping it: the second artifact

`models/cropguard_v1_gate.tflite` (1.141 MB) is the same weights and the same
dynamic-range quantization as the production model, re-exported with three outputs:
softmax probs, logits, and the 576-d GAP embedding. Built and verified by
`python -m src.export_ood_model` (`outputs/gate_export_report.json`), on 200 clean
test images plus all 101 OOD and all 37 field photos:

- probabilities vs `cropguard_v1_production.tflite`: **max |Δ| = 0.0** (bit-identical)
- argmax agreement: **1.000** — swapping the artifact cannot change a diagnosis
- `softmax(logits)` vs production probs: max |Δ| = 1.4e-7, confirming the logits
  output really is the pre-softmax value

**The gate decision under quantization is not free**, and this is the caveat to carry:
the cosine score computed from the quantized embedding differs from the float-model
score by up to **0.061**, and **5 of 338 images (1.5%) land on the other side of the
threshold** — 2 OOD, 3 field, 0 clean. The float and quantized models are already
known to disagree under shift (`docs/EXPLAINABILITY.md`: 0.980 agreement clean,
0.750–0.824 shifted), so this is the same effect reaching the gate. Net effect at the
shipped threshold is slightly favourable (quantized accepts 5.9% of OOD and 91.9% of
field leaves, against 7.9% / 89.2% float) but that is luck, not design.

Browser side: `web/src/lib/oodGate.ts` fetches `/ood_gate.json` (96 KB: 17 unit
class-mean vectors + the threshold), computes one dot product per class against the embedding,
and `page.tsx` renders a rejection panel *instead of* any diagnosis when the score is
below threshold. The model output stays reachable behind an explicit "show what the
model returned anyway" disclosure, labelled as not a diagnosis. If `/ood_gate.json`
fails to load the verdict is `null` — the UI warns and falls back to the tier system
rather than silently declaring everything a leaf. The TypeScript implementation was
cross-checked against the Python scores on 18 embeddings: max |Δ| = 4.9e-9.

## 8. Limits

1. **n = 37 field photos sets the threshold**, 17 of them one-per-class. The 90th
   percentile of 37 samples moves if four images change. Treat 0.5981 as provisional
   and re-run `python -m src.ood` when `real_world_test/` grows — this is now a second
   thing blocked on that directory, alongside the Phase 1 accuracy number.
2. **The negative set is Commons, not user photos.** Real misuse would be blurrier,
   closer, and worse-lit than a curated Commons upload, and hard negatives —
   half-leaf-half-background framing, a leaf held at arm's length — are
   under-represented relative to faces and furniture.
3. **The class means are fitted on studio images** (2,026 train photos). They work on
   field photos because cosine is scale-invariant, not because anything about them is
   field-calibrated.
4. **Both AUROCs are computed against 101 negatives.** The per-category rates are over
   5–12 images each; they show a pattern, not a precise rate.
5. **The gate cannot tell a tomato leaf from an oak leaf** (§6), and it inherits every
   quantization discrepancy in §7.
