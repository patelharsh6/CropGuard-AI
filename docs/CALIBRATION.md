# Calibration — replacing the magic thresholds with measured ones

**Date:** 2026-08-23 · **Script:** `python -m src.calibration` · **Report:**
`outputs/calibration_report.json` · **Figures:** `outputs/reliability_diagram.png`,
`outputs/risk_coverage.png`

Everything below is measured on `models/cropguard_v1_production.tflite` — the exact
1.15 MB dynamic-range artifact the browser loads, not the Keras model. The temperature
is fit on the **validation** split (3,625 images) and every generalization number is
read off the **test** split (3,625 images), which the fit never saw.

The question this phase answers: the UI gates on top-1 softmax probability at 0.85 and
0.50, thresholds chosen by intuition in Phase 0. A softmax score is not a probability.
Are those numbers defensible, and what should they actually be?

---

## 1. The model was already well calibrated — and *under*-confident

| | accuracy | mean confidence | conf − acc | ECE | MCE (n≥10) | Brier | NLL |
|---|---|---|---|---|---|---|---|
| val, raw | 0.9399 | 0.9252 | **−0.0147** | 0.0152 | 0.1098 | 0.0909 | 0.1856 |
| val, T-scaled | 0.9399 | 0.9364 | −0.0035 | **0.0084** | 0.0798 | 0.0899 | 0.1835 |
| test, raw | 0.9401 | 0.9223 | **−0.0179** | 0.0187 | 0.1457 | 0.0934 | 0.1847 |
| test, T-scaled | 0.9401 | 0.9335 | −0.0067 | **0.0075** | 0.1265 | 0.0925 | 0.1819 |

**Fitted temperature: T = 0.8878.** Note T < 1, which *sharpens* the distribution. This
is the opposite of the textbook finding — modern CNNs are famously overconfident and
need T > 1 — and it is the most interesting result here. This model's mean confidence
sits **1.8 points below** its accuracy on test. Plausible cause, given the recipe in
`src/train.py`: phase-2 fine-tuning ran at 1e-5 then 5e-6 with heavy realism
augmentation (`BackgroundReplace`, motion blur, JPEG artifacts). Both suppress logit
magnitude — the augmentation makes many training images genuinely ambiguous, so the
loss never pushes the logit scale up the way a clean high-LR fit does.

Temperature scaling roughly halves ECE (test 0.0187 → 0.0075) and improves NLL and
Brier slightly. It changes **no prediction**: `p ** (1/T)` renormalized is strictly
monotone, so the argmax is identical. Only the confidence value — and therefore the
tier — moves.

### On MCE, and why the report carries two of them

All-bins MCE is 0.2947 on val raw and 0.7431 on val calibrated, which looks like
calibration made things dramatically worse. It didn't: that 0.74 comes from a bin
holding **one image**. The report therefore records both `mce` (all bins) and
`mce_robust` (bins with ≥ 10 images, which moves 0.1098 → 0.0798, i.e. the honest
direction). The reliability diagram fades sparse bins and labels their counts, and
omits empty bins entirely rather than drawing them as zero-accuracy — an empty bin is
absence of data, not miscalibration.

---

## 2. Deriving the thresholds — and why a 0.95 target is worthless here

The plan's suggestion was "pick the HIGH threshold as the smallest confidence where
empirical accuracy meets a target (e.g. 0.95)." Running that literally exposes the
trap:

| accuracy target | derived τ | HIGH coverage | selective accuracy | **confidently wrong** |
|---|---|---|---|---|
| 0.950 | 0.510 | 0.980 | 0.9502 | **4.88%** |
| 0.960 | 0.580 | 0.958 | 0.9603 | 3.81% |
| 0.970 | 0.680 | 0.928 | 0.9703 | 2.76% |
| 0.980 | 0.815 | 0.874 | 0.9801 | 1.74% |
| **0.990** | **0.945** | **0.754** | **0.9909** | **0.69%** |
| 0.995 | 0.975 | 0.667 | 0.9959 | 0.28% |

(validation split, calibrated; "confidently wrong" = share of *all* images that land in
HIGH and are incorrect)

The model's unconditional accuracy is 0.940. A 0.95 target is therefore satisfied by
answering **98% of everything** — it buys almost no abstention, and because coverage
rises, the absolute rate of confident errors goes *up* (4.88% vs 1.16% under the old
intuited 0.85). A conditional-accuracy target below, at, or barely above the
unconditional accuracy is not a safety constraint at all.

So the target is a product decision, not a statistical one, and the sweep is the
artifact that makes its price visible. **Chosen: 0.99**, on the principle already
written into the UI — a confidently wrong diagnosis is worse than no diagnosis — giving
**HIGH ≥ 0.945**. The default in `src/calibration.py` is `--high-target 0.99`.

Two details of the derivation:

- **Monotone-safe**: a threshold only counts if it meets the target *and every larger
  threshold also does*, so one lucky bin can't set the boundary.
- **The MODERATE floor needs a local criterion, not a cumulative one.** The first
  attempt asked for band accuracy ≥ 0.50 over `[τ, HIGH)`, which is satisfied all the
  way down to τ = 0.01 — the high-confidence mass inside the band drags the average
  above any sane target, and the LOW tier vanishes entirely. The right question is
  local: at *this* confidence, is the top-1 still better than a coin flip? Measured
  over a 0.05-wide window with ≥ 50 images, that gives **MODERATE ≥ 0.595**.

## 3. What shipped

`web/src/lib/calibration.ts` (T = 0.8878) and `web/src/lib/confidenceTier.ts`
(0.945 / 0.595), mirrored in `src/config.py`. On the held-out test split:

| | tier | coverage | top-1 | top-3 | confidently wrong |
|---|---|---|---|---|---|
| old: raw, 0.85 / 0.50 | HIGH | 0.826 | 0.9860 | 0.9990 | **1.16%** |
| | MODERATE | 0.149 | 0.7681 | 0.9852 | |
| | LOW | 0.025 | 0.4505 | 0.9341 | |
| **new: calibrated, 0.945 / 0.595** | HIGH | 0.744 | **0.9930** | 0.9993 | **0.52%** |
| | MODERATE | 0.209 | 0.8393 | 0.9895 | |
| | LOW | 0.047 | 0.5529 | 0.9588 | |

The trade is explicit: **8 points of HIGH coverage for less than half the confident
error rate.** Every image that leaves HIGH lands in MODERATE, which still shows a
diagnosis behind a caution banner plus top-3 — so nothing is lost, it is caveated. And
the LOW tier's 0.9588 top-3 accuracy is what justifies showing three candidates there
rather than nothing at all.

Frontend implementation is one line of arithmetic, no model change: the production
graph already ends in softmax, and `softmax(log(p)/T) === softmax((z−c)/T)` for any
constant `c`, so raising each probability to `1/T` and renormalizing *is* temperature
scaling on the hidden logits. Verified against the Python path to 5.6e-16 max abs
difference over 200 validation vectors.

## 4. Risk-coverage — the selective-prediction view

`outputs/risk_coverage.png`. **AURC: 0.0070 (val), 0.0067 (test)** — the area under the
risk-coverage curve, where 0 would mean every error is ranked below every correct
prediction. Test split, calibrated:

| coverage | accuracy | τ |
|---|---|---|
| 1.00 | 0.9401 | 0.233 |
| 0.90 | 0.9739 | 0.745 |
| 0.80 | 0.9890 | 0.909 |
| 0.70 | 0.9972 | 0.964 |
| 0.50 | 1.0000 | 0.995 |

Abstaining on the least-confident 30% of images reaches 99.7% accuracy on the rest, and
at 50% coverage there are **zero** errors in 1,812 images. Confidence ranks errors
well; the tier system is a coarse three-bucket read of this curve, and 0.945 sits at the
knee.

## 5. Does calibration fix the Phase 1 failure?

Phase 1's worst finding (`docs/DOMAIN_SHIFT.md`, Result A) was that on 17 web-sourced
photos the gate *failed*: HIGH coverage stayed at 58.8% while HIGH accuracy fell to
0.600, so **23.5% of images got a confident wrong diagnosis** — twenty times the
benchmark rate. Rescoring both out-of-distribution sets with temperature scaling and
the derived thresholds (`--calibrated`):

| set | | HIGH coverage | HIGH accuracy | confidently wrong |
|---|---|---|---|---|
| 17-class photo set (n=17) | before | 0.588 | 0.600 | 4/17 = **23.5%** |
| | after | 0.353 | 0.833 | 1/17 = **5.9%** |
| Wikimedia Commons (n=20) | before | 0.200 | 0.750 | 1/20 = 5.0% |
| | after | 0.100 | 1.000 | **0/20** |

Reports: `outputs/real_world_report_calibrated.json`,
`outputs/web_sourced_report_calibrated.json`. Top-1 accuracy is unchanged (0.5294 and
0.4000) — calibration cannot make the model right, only make its uncertainty legible.

**This is a substantial mitigation, not a fix, and the sample sizes are tiny** (n=17 and
n=20; the "before" figures are 4 and 1 images). One confident error survives on the
17-image set: `Tomato___Bacterial_spot → Tomato___Septoria_leaf_spot` at 0.971, inside
the tomato brown-lesion cluster that `src/evaluate.py` already flags as the model's main
confusion. That failure is a *discrimination* problem, not a confidence problem, and no
threshold can reach it — it needs Phase 3's attention maps and Phase 4's OOD gate.

## 6. Limits of these numbers

- The thresholds are derived on clean PlantVillage validation data. §5 shows they
  transfer usefully to shifted data, on samples far too small to be a guarantee.
- A single scalar T is the crudest calibration map there is. It cannot fix *per-class*
  miscalibration, and with 35× class imbalance that almost certainly exists —
  per-class or vector scaling is unmeasured here.
- ECE with 15 equal-width bins is itself a biased estimator, and 92% of validation
  images sit in the top two bins, so the low-confidence region rests on tens of images.
- Calibration measures confidence *given* an in-distribution input. It says nothing
  about a photo of a chair, which is exactly why Phase 4 exists.

## Reproduce

```bash
python -m src.calibration                       # cached predictions after the first run
python -m src.calibration --high-target 0.98    # see what a looser target ships
python -m src.eval_real_world --calibrated --json outputs/real_world_report_calibrated.json
python -m src.eval_real_world --dir web_sourced_test --calibrated \
    --json outputs/web_sourced_report_calibrated.json
python -m src.eval_tiers                        # the uncalibrated before-picture
```
