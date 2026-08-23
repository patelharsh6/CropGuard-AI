# Explainability — does it look at the lesion, or at the background?

**Date:** 2026-08-23 · **Script:** `python -m src.explain` · **Report:**
`outputs/gradcam_report.json` · **Figures:** `outputs/gradcam/<cohort>/*.jpg`

`BackgroundReplace` exists in `src/augmentation.py` because every PlantVillage image is a
detached leaf on a uniform studio background, and a network can reach 0.94 by reading the
background instead of the lesion. That augmentation was written on the *assumption* that
the shortcut was a risk, and never checked. This phase checks it, and uses the same
attention maps on the three failures the earlier phases left open: the tomato brown-lesion
cluster, the confidently-wrong field photos, and the whole-plant out-of-distribution images.

Analysis runs on `models/cropguard_v1.keras` (the float model), because TFLite exposes no
gradients. §7 measures how much that matters.

---

## 1. At the final layer, Grad-CAM here *is* CAM — verified, not assumed

The head is `GlobalAveragePooling2D -> Dropout -> Dense(17, softmax)`. For the final
conv activation `A` (7×7×576), the logit for class *c* is exactly
`sum_k W[k,c] * mean(A[:,:,k])`, so

```
d logit_c / d A[i,j,k] = W[k,c] / 49        (constant in i, j)
```

Grad-CAM averages that gradient over space to get its channel weights, so the weights
*are* the classifier's own weights and the map degenerates to Zhou et al.'s CAM. Measured
on one image: spatial variance of the gradient `3.1e-17`, and
`max |grad_weight − W[:,c]/49| = 3.7e-09` against a weight scale of `1.6e-02`. This is not
a defect — it means the deep map is honest and cheap — but it does mean the deep map
carries no information beyond CAM and lives on a 7×7 grid (32-pixel cells).

So `src/explain.py` computes a **second** map at `activation_11` (14×14×288), the finest
layer where the gradient genuinely varies spatially. Every table below reports both.

The logits themselves are not an output of the graph (it ends in softmax), so the head is
re-applied by hand inside the tape. That reconstruction is asserted against
`model.predict()` every run: `max|Δ| = 1.2e-07`.

## 2. Quantifying leakage — and being honest about the ruler

Metric: fraction of CAM mass inside a leaf mask, where the mask is the *same* rough
GrabCut segmentation `BackgroundReplace` uses (`_segment_leaf`, thresholded at α ≥ 0.5).
A raw "mass inside leaf" number is meaningless on its own — a CAM that ignores the image
entirely scores `mass == area` — so the reported quantity is

```
lift = (CAM mass inside mask) / (mask area fraction)
```

`lift > 1` = attention concentrates on the leaf, `≈ 1` = indifferent, `< 1` = the model is
looking at the background. `peak_in_leaf` is whether the single hottest pixel is on-leaf.

**Where this ruler is valid.** `_segment_leaf` assumes one leaf on a plain background —
PlantVillage framing. It is *not* a general leaf segmenter, and on field photos it
frequently returns something else: on `real_world_test/Potato___Late_blight/image.png` it
segmented the necrotic lesion alone (area 0.045, "lift" 10.04), and on the corn
Northern-Leaf-Blight photo it also picked the lesion (area 0.058, lift 3.78). Masks
outside a 0.05–0.95 area band are recorded but excluded from every mean. Treat the
**test-split sweep in §3 as the measurement**, and the photo-cohort numbers in §5 as
descriptive only.

Also note the mask is conservative — it hugs the lamina and drops feathered edges — so
off-leaf mass is an **upper bound** on true leakage.

## 3. The answer: no gross background shortcut

200 random held-out test images (seed 42), full metrics in
`outputs/gradcam_report.json → leakage_sweep`:

| group | n | leaf area | CAM mass in leaf | **lift** | peak in leaf |
|---|---|---|---|---|---|
| all | 200 (197 masks usable) | 0.387 | 0.625 | **1.66** | **0.858** |
| correct | 190 | 0.390 | 0.629 | 1.66 | 0.861 |
| wrong | 10 | 0.335 | 0.551 | 1.56 | 0.800 |

Mid-layer (14×14) on the same images: lift **1.39** overall, and `peak_in_leaf` 0.61.

**Verdict: the shortcut hypothesis is not supported.** Attention is 1.66× more
concentrated on the leaf than area alone would give, and the hottest pixel is on the leaf
in 86% of images. The augmentation's purpose is satisfied on the distribution it was
trained for. What the number does *not* say is that 37% of CAM mass off-leaf is fine — it
is a real fraction, part of it mask conservatism, part of it genuine surround context
(a 7×7 grid cell is 32×32 pixels and straddles the leaf edge).

**Attention does not detect its own errors.** The correct-vs-wrong gap is
1.66 → 1.56 (deep) and 1.40 → 1.10 (mid), with `peak_in_leaf` 0.86 → 0.80 (deep) and
0.63 → 0.30 (mid). The direction is consistent across both layers, and **n = 10 wrong
images** makes it a hint, not a result. The mid-layer separation is the wider one and
would be the thing to re-measure on a few hundred errors before anyone considers CAM mass
as an abstention signal. As it stands it is not one.

## 4. What "working" attention looks like

`outputs/gradcam/correct_high/` — 8 correct, calibrated-HIGH test images, one per class
until the classes run out. Mean confidence 0.987, deep lift 1.58.

The maps are readable and lesion-specific: `04_Potato_Early_blight` puts the deep peak on
the concentric-ring necrosis and the mid map resolves it into individual lesions;
`01_Corn_Common_rust` covers the pustule field, with the mid map firing on individual
pustules; `02_Corn_Northern_Leaf_Blight` (a field-framed image where the mask caught only
the lesion) puts the peak precisely inside it.

One honest exception: `03_Corn_healthy` at 0.998 has its deep hot spots along the top
edge, partly off-leaf. "Healthy" has no lesion to point at, so there is nothing to
localize — the class is defined by the *absence* of evidence, and CAM has no vocabulary
for that. Worth remembering before reading any healthy-class heatmap as insight.

## 5. The three open failures, explained

### 5a. The tomato brown-lesion cluster is a discrimination failure, not leakage

`outputs/gradcam/cluster/` — 8 test images the shipped model confuses *within* the
Early blight / Target Spot / Septoria / Spider-mites cluster.

Deep lift **1.32**, `peak_in_leaf` **0.875** — attention is on the leaf for these errors
just as it is for correct predictions. The mid layer is where they differ: lift **0.79**
versus 1.39 on the sweep, i.e. at 14×14 the *wrong* class's evidence drifts off-leaf.
`07_Tomato_Target_Spot` (predicted Spider mites, 0.876) is the clearest case: the
predicted class's map has substantial mass in the top-left background, while the true
class's map sits entirely on the lamina.

Rendering both classes side by side is what makes this diagnosable, and the pattern in
`00_`, `04_` and `05_` is the same: both maps are on-leaf, on *different* regions, each
plausible. Nothing about these pictures suggests a fixable shortcut. The model is being
asked to separate brown spots with brown spots, at 7×7 for the deep map, and losing.
That points at input resolution and feature capacity — Phase 5 ablation territory — not
at data cleanup.

### 5b. The surviving confident errors are host confusion and lookalike confusion

Two errors were left standing by Phase 2's calibration. Both are now explained, and
neither is a localization failure:

- `real_world/05_real_world_test_Potato_Late_blight` → **Tomato** Late blight at 0.957. The CAM for the
  predicted class and for the true class are the same blob, dead centre on the necrotic
  lesion. The model found the right pathology and picked the wrong host.
- `real_world/07_real_world_test_Tomato_Bacterial_spot` → Septoria leaf spot at 0.977 (the error
  `docs/CALIBRATION.md` §5 flags as unreachable by any threshold). Predicted-class and
  true-class maps are nearly identical, both on the spotted lamina, deep mass in leaf
  0.923. The two classes are read off the *same* pixels; the difference is spot
  morphology at a scale this feature map does not resolve.

This is the useful negative result of the phase: **for these errors, explainability
confirms there is nothing to fix in the data pipeline.** The evidence is in the right
place and the decision boundary is wrong.

### 5c. On out-of-distribution images the CAM picks a leaf-shaped blob and commits

`outputs/gradcam/ood/` — 4 whole-plant / non-leaf images. Deep lift **0.38**,
`peak_in_leaf` **0.0** (2 of 4 masks usable; on these images the mask is not a leaf, so
read the pictures, not the table).

- `ood/01_web_sourced_test_00_A_young_corn_plant` → Tomato Early blight, 0.520: the map collapses onto the single
  brightest central leaf and ignores the rest of the field.
- `ood/03_web_sourced_test_t03_Passalora_fulva_a1__1_` → Tomato Early blight, **0.968 raw / 0.983 calibrated** — a
  whole-plant tomato scene that clears the shipped HIGH gate (0.945). The map is spread
  over several patches of foliage with no single lesion anywhere.

A closed-set softmax has no way to say "that is not one leaf." Confidence cannot express
it and attention does not suppress it — the model finds leaf-like texture, and 17 classes
is all it can answer. This is the strongest evidence yet that **Phase 4's OOD gate is
load-bearing, not polish**, and it also suggests a concrete gate feature: on all four OOD
images the CAM is either diffuse or a single small blob far from the mask, whereas
in-distribution maps are lesion-locked.

### 5d. Field photos: attention holds up better than accuracy does

| cohort | n | acc (Keras) | mean conf | deep lift | mid lift | peak in leaf |
|---|---|---|---|---|---|---|
| correct_high (clean test) | 8 | 1.000 | 0.987 | 1.58 | 1.16 | 0.63 |
| cluster (clean test errors) | 8 | 0.250 | 0.685 | 1.32 | 0.79 | 0.88 |
| real_world | 17 | 0.588 | 0.734 | 1.56 | 1.21 | 0.56 |
| web_sourced | 20 | 0.350 | 0.665 | 1.50 | 1.25 | 0.26 |
| ood | 4 | — | 0.690 | 0.38 | 0.27 | 0.00 |

Accuracy falls from 1.00 to 0.35 across those rows while deep lift barely moves
(1.58 → 1.50). Attention *localization* survives the domain shift that *classification*
does not — consistent with Phase 1's "crop right, disease wrong" finding, and with §5a:
the model keeps finding the diseased tissue and keeps mislabelling it.

Note the mask-area column behind these rows: mean leaf area is 0.20 on the web-sourced
set versus 0.39 on the test split, which is the framing difference itself showing up in
the segmentation, and another reason to read §5d as descriptive.

## 6. Answering the question `BackgroundReplace` was written for

The augmentation's premise was that the model *could* learn the studio background. Three
results together say it did not:

1. Attention concentrates on the leaf at 1.66× area on held-out test data (§3).
2. `background_replace` costs only 0.0615 accuracy in the corruption sweep
   (`docs/DOMAIN_SHIFT.md` §2) — the model is close to indifferent to the background.
3. On field photos with real cluttered backgrounds, lift holds at 1.50–1.56 (§5d).

None of that proves the augmentation *caused* the outcome — that requires training without
it and re-measuring, which is exactly the Phase 5 ablation row. What can be said now is
that the failure mode it was written to prevent is not present, and that the accuracy
collapse under shift has a different cause: fine-grained discrimination, not background
reliance.

## 7. Side finding: the float model and the shipped model disagree under shift

`keras_tflite_agree` compares the Keras float prediction against
`cropguard_v1_production.tflite` on the identical preprocessed tensor:

| cohort | agreement |
|---|---|
| clean test sweep (n=200) | **0.980** |
| correct_high | 1.000 |
| real_world | 0.824 |
| web_sourced | 0.800 |
| cluster (borderline by construction) | 0.750 |

On clean data dynamic-range quantization is nearly lossless, as
`docs/quantization_findings.md` reports. On shifted data, where confidences sit near the
boundary, it flips **1 prediction in 5**. Two consequences, both worth stating plainly:

- Every heatmap in `outputs/gradcam/` explains the **float** model. On the ~18% of
  shifted images where the two disagree, it is not explaining what the browser computed.
- The domain-shift numbers in `docs/DOMAIN_SHIFT.md` are properties of the *quantized*
  artifact specifically, not of the architecture. A float-vs-quantized comparison on the
  shifted sets is a cheap, previously invisible experiment.

Related fragility found while building this: OpenCV and TensorFlow decode the same JPEG
up to ~4 grey levels apart, and that is enough to flip a borderline cluster image from
Target Spot (0.488) to Spider mites (0.662). `load_image()` therefore uses `tf.io` to
match the rest of the repo, with cv2 only as a fallback for formats TF cannot read. This
is a good argument for the cross-language preprocessing parity test in Phase 6.

## 8. Limits

- **The mask is the weakest link.** GrabCut with a centre rectangle is a PlantVillage-shaped
  assumption; §2 shows it segmenting lesions instead of leaves on field photos. Every
  leakage number is only as good as it, and a hand-annotated mask set (even 30 images)
  would turn §3 from indicative into solid.
- **n = 10 wrong images** in the sweep. The correct-vs-wrong comparison is underpowered
  and is reported as a direction, not a finding.
- **The deep map is CAM on a 7×7 grid.** Fine-grained lesion morphology — precisely what
  §5a says the model is failing on — is below its resolution. The 14×14 map helps and is
  still coarse.
- **CAM says where, never why.** "Attention is on the lesion" is compatible with reading
  colour, texture or shape, and the maps cannot separate those. Occlusion or feature
  ablation would be needed.
- **Healthy classes have no positive evidence to localize** (§4), so their maps should not
  be over-read.
- **Cohort selection uses the cached TFLite predictions** from `src/calibration.py`, while
  the CAMs are computed on Keras — so a picked "error" can render as correct (2 of 8 in
  the cluster cohort). That is the §7 disagreement showing up, not a bug.
- Nothing here is a browser-side feature yet. The plan's stretch goal — exporting the
  feature map as a second TFLite output and rendering the heatmap on-device — is untouched.

## Reproduce

```bash
python -m src.calibration                 # first, to populate outputs/cache/ (cohort selection reads it)
python -m src.explain                     # all cohorts + 200-image leakage sweep, ~9 min CPU
python -m src.explain --cohort cluster --n 12
python -m src.explain --sweep 600         # tighter leakage statistics
python -m src.explain --no-mask           # heatmaps only, no segmentation
```
