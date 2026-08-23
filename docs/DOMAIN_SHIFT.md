# Domain shift: from 0.9401 on the benchmark to *what* in the field?

Phase 1 of `plan.md`. Three parts:

| § | What | Status | Headline |
|---|---|---|---|
| 1 | Developer-assembled set, all 17 classes (n=17) | done | **0.5294** top-1 vs 0.9401 |
| 1b | Web-sourced, hand-vetted photos (n=20) | done | **0.40** top-1 vs 0.9401 |
| 2 | Synthetic corruption stress test (n=602) | done | 0.7641 under stacked field-like corruption |

All three agree on the direction and rough size of the problem. They disagree on one
important point — whether the confidence gate protects the user under shift — and §1 is
the one that says it does not.

---

## 1. The developer-assembled test set — **done: 0.5294**

17 photographs, one per class, covering **all 17 classes** — assembled by the developer
from web sources rather than shot in the field, and labelled by folder placement. Scored
with:

```bash
python scripts/check_test_photos.py     # validate
python -m src.eval_real_world           # -> outputs/real_world_report.json
```

| | This set (n=17) | Clean test (n=3,625) | Drop |
|---|---|---|---|
| Top-1 | **0.5294** (95% CI [0.31, 0.74]) | 0.9401 | **−0.4107** |
| Top-3 | 0.7059 (CI [0.47, 0.87]) | 0.9953 | −0.2894 |
| Mean confidence | 0.8025 | 0.9223 | — |
| Confidently wrong | **23.5%** (4 of 17) | 1.16% | — |

**Integrity checks run before accepting the number.** All 17 files were hashed against
all 162,916 images in `data/plantvillage_dataset/` — **zero exact duplicates**, so there
is no training-set leakage. Resolutions range 150×200 to 1200×675, i.e. genuine web
images rather than 256×256 PlantVillage crops. The four confidently-wrong cases were
inspected individually and are **model errors, not label errors** (see below).

### Finding 1 — accuracy collapses along crop granularity

| Crop | Classes in the taxonomy | n | Accuracy |
|---|---|---|---|
| Corn | 4 | 4 | **1.000** |
| Potato | 3 | 3 | 0.667 |
| Tomato | **10** | 10 | **0.300** |

This is the clearest signal in the set. Corn is perfect, potato is decent, tomato — which
carries 10 of the 17 classes and contains the entire brown-lesion cluster — falls to 0.30.
It is not that the model fails to understand the image: **the crop is identified correctly
in 15/17 (88%)** while the full class is right in only 10/17 (59%). What breaks under
shift is *fine-grained discrimination between similar lesions on the same crop*, exactly
where the clean-test confusion matrix was already weakest.

### Finding 2 — the confidence gate did **not** hold here

This is the important negative result, and it contradicts what the other two sections
suggested.

| | Coverage | Accuracy |
|---|---|---|
| HIGH (p ≥ 0.85) | 58.8% | 0.600 |
| MODERATE | 35.3% | 0.500 |
| LOW | 5.9% | 0.000 |

Compare with the clean test split, where HIGH was 82.6% coverage at 0.9860 accuracy. Here
HIGH coverage only fell to 59% while HIGH *accuracy* fell to 0.60 — so **23.5% of all
images were confident and wrong**, twenty times the clean-test rate of 1.16%.

In §1b (web-sourced) and §2 (synthetic) the model responded to unfamiliar input by
becoming *unconfident*, and the tier system converted that into abstention. On this set it
stayed confident and was wrong. Mean confidence is 0.80 here versus 0.69 on the
web-sourced set. **Abstention is therefore not a reliable safety net** — it worked on two
distributions and failed on a third, which is a much weaker claim than the one §1b alone
would have supported.

> **Update (Phase 2, 2026-08-23).** This result is what promoted calibration from polish
> to load-bearing, and calibration answered it partly. Rescored with temperature scaling
> (T = 0.8878) and the derived thresholds 0.945 / 0.595 —
> `python -m src.eval_real_world --calibrated`, report
> `outputs/real_world_report_calibrated.json`:
>
> | | Coverage | Accuracy |
> |---|---|---|
> | HIGH (p ≥ 0.945) | 35.3% | 0.833 |
> | MODERATE | 52.9% | 0.333 |
> | LOW | 11.8% | 0.500 |
>
> Confidently wrong: **4 of 17 → 1 of 17 (23.5% → 5.9%)**. On the web-sourced set of §1b
> it goes 1 of 20 → **0 of 20**. Top-1 accuracy is unchanged in both cases, as it must be
> — the transform is monotone. The claim above survives in weaker form: abstention is
> better calibrated than it was, still not a guarantee, and the surviving error
> (Bacterial spot → Septoria, 0.971) is a discrimination failure inside the tomato
> brown-lesion cluster that no threshold can reach. Full analysis: `docs/CALIBRATION.md`.

The four confident errors, all inspected by eye:

| True | Predicted | Confidence | Verdict |
|---|---|---|---|
| Potato Late blight | **Tomato** Late blight | 0.924 | Right disease, wrong crop — potato and tomato are both Solanaceae with similar leaflets |
| Tomato Bacterial spot | Tomato Septoria leaf spot | 0.954 | Both present as small dark spots; a genuinely hard pair |
| Tomato Septoria leaf spot | Tomato Early blight | 0.917 | The known brown-lesion cluster |
| Tomato mosaic virus | Tomato Leaf Mold | 0.857 | Leaf mold also yellows the upper surface in blotches |

Every one is a plausible confusion between visually adjacent classes, not a nonsense
prediction — and every one would have been shown to a user as a full diagnosis with a
treatment panel.

### Finding 3 — healthy leaves were handled correctly, refuting the §1b hypothesis

All **3/3 healthy leaves were correctly identified as healthy**, and **0 of 14 diseased
leaves were called healthy.**

§1b saw the opposite (3/3 healthy called diseased) and raised it as a hypothesis to test.
This set refutes it. Both samples are tiny — 3 images each — so the honest conclusion is
that *neither* result is evidence of a systematic healthy-class bias, and the §1b finding
should not be quoted. The one consistent point across both: **no diseased leaf was ever
called healthy**, so errors run toward false alarms rather than missed disease.

### What this result does and does not license

**Can say:** "On a 17-image, all-classes test set assembled from outside the training
distribution, top-1 fell from 0.94 to 0.53 and top-3 to 0.71. Accuracy is a function of
taxonomic granularity — 1.00 on corn's 4 classes, 0.30 on tomato's 10 — and the crop is
still identified 88% of the time. Critically, the confidence gate did not protect the
user here: 23.5% of images were confident *and* wrong, against 1.16% on the benchmark —
which Phase 2's calibrated thresholds later cut to 5.9% against 0.52%."

**Cannot say:** "Real-world accuracy is 53%." n=17 with a 95% CI of [0.31, 0.74]; one
image per class, so every per-class number is 0 or 1; the images were sourced from the
web rather than photographed in the field, so the labels are still placement decisions
rather than examinations of a plant; and the framing skews toward clear, illustrative
symptom photos.

**Still worth doing:** photographs taken from actual plants, several per class, would
tighten the interval and remove the last label-provenance caveat. But the headline
conclusion — a large drop, concentrated in fine-grained tomato classes, with an
unreliable confidence gate — is now measured rather than assumed.

---

## 1b. Web-sourced stand-in — done, and the result is stark

Since no plants were available to photograph, 20 leaf photos were harvested from
Wikimedia Commons, hand-vetted one by one, and scored with the same script:

```bash
python -m src.eval_real_world --dir web_sourced_test
```

Full set-up, vetting log and licensing: `web_sourced_test/README.md`. Raw output:
`outputs/web_sourced_report.json`.

| | Web-sourced (n=20) | Clean test (n=3,625) | Drop |
|---|---|---|---|
| Top-1 accuracy | **0.4000** | 0.9401 | **−0.5401** |
| Top-3 accuracy | 0.7500 | 0.9953 | −0.2453 |
| Mean confidence | 0.686 | 0.922 | — |

95% Wilson interval on that 0.40 is **[0.22, 0.61]**. Field-only subset (14 images):
0.4286. Excluding the one moderate-confidence label: 0.3684. **Quote this as "roughly
half", never as 0.4000** — n=20 does not support a fourth decimal place.

### The tier system is the thing that survives

| Tier | Clean test coverage | Clean acc | Web-sourced coverage | Web-sourced acc |
|---|---|---|---|---|
| HIGH | 82.6% | 0.9860 | **20.0%** | 0.750 |
| MODERATE | 14.9% | 0.7681 | 65.0% | 0.385 |
| LOW | 2.5% | 0.4505 | 15.0% | 0.000 |

HIGH coverage collapses from 82.6% to 20%, and **only one image out of 20 was
confidently wrong**. Faced with input it has never seen, the model mostly declines to
be confident rather than being confidently wrong — the app would have shown a caution
banner or refused to diagnose on 80% of these. This is the same graceful degradation
the synthetic test showed in §2, now confirmed on real photographs, and it is the
single most defensible design decision in the project.

The one confident error is instructive: an aged-slide-film photo of corn **common
rust** predicted as **gray leaf spot** at 0.909. Both are corn foliar lesions; the
colour cast of the film stock is a corruption type nothing in training covers.

### Finding: it gets the crop right and the disease wrong

**Crop identified correctly in 16/20 (80%)** while the full class was right in only
8/20. The failure is not "the model has no idea what it is looking at" — it is that
fine-grained lesion discrimination is what breaks under shift, exactly the axis on
which PlantVillage's studio conditions are least representative. Top-3 at 0.75 versus
top-1 at 0.40 says the same thing: the right answer is usually still in the shortlist.

### Finding: every healthy leaf was called diseased

All 3 healthy photos (2 potato, 1 tomato) were predicted as some disease — and no
diseased photo was called healthy. On n=3 that is barely evidence, but the direction
matters for the product: the errors here are **false alarms, not missed diseases**,
which is the safer direction to fail in, and it is consistent with a model whose
"healthy" class was learned from spotless studio specimens. Real leaves have dust,
mechanical nicks, uneven colour and shadow. This is a concrete, testable hypothesis
for the hand-taken set to confirm.

### What this result does and does not license

**Can say:** "On 20 hand-vetted photographs from Wikimedia Commons — not from the
training distribution — top-1 fell from 0.94 to roughly 0.40, top-3 to 0.75, and the
crop was still identified 80% of the time. The confidence gate held: HIGH-tier coverage
dropped from 83% to 20%, so the app would have abstained rather than misdiagnosed on
most of them. n=20, CI [0.22, 0.61]."

**Cannot say:** "Real-world accuracy is 40%." The labels are uploader captions, the
sample is not random, seven classes are missing, and the framing skews toward
in-canopy and multi-leaflet shots that are outside the app's stated single-leaf
intended use. Section 1 is still the number that matters.

---

## 2. Synthetic stress test — done

`src/eval_domain_shift.py` takes clean held-out test images and corrupts them the way a
handheld outdoor phone photo differs from a PlantVillage studio shot, one factor at a
time, then scores the production `.tflite` on each.

**Setup:** 602 images, stratified from the test split, seed 42, production
dynamic-range artifact, `p=1.0` on every transform. Raw output:
`outputs/domain_shift_report.json`.

| Corruption | Top-1 | Δ vs clean | Top-3 | Mean conf | HIGH coverage | HIGH acc | Confidently wrong |
|---|---|---|---|---|---|---|---|
| sensor_noise_severe | 0.5399 | **−0.3937** | 0.7990 | 0.768 | 0.473 | 0.761 | **0.1130** |
| field_composite | 0.7641 | **−0.1694** | 0.9352 | 0.756 | 0.460 | 0.971 | 0.0133 |
| underexposed | 0.7674 | −0.1661 | 0.9086 | 0.800 | 0.558 | 0.961 | 0.0216 |
| background_replace | 0.8721 | −0.0615 | 0.9850 | 0.844 | 0.615 | 0.989 | 0.0066 |
| defocus_blur | 0.8837 | −0.0498 | 0.9850 | 0.876 | 0.721 | 0.988 | 0.0083 |
| jpeg_artifacts | 0.8887 | −0.0449 | 0.9850 | 0.885 | 0.729 | 0.979 | 0.0150 |
| sensor_noise (mild) | 0.9003 | −0.0332 | 0.9900 | 0.892 | 0.752 | 0.978 | 0.0166 |
| off_angle | 0.9053 | −0.0282 | 0.9967 | 0.893 | 0.744 | 0.991 | 0.0066 |
| motion_blur | 0.9203 | −0.0133 | 0.9850 | 0.887 | 0.729 | 0.991 | 0.0066 |
| white_balance | 0.9269 | −0.0066 | 0.9934 | 0.911 | 0.789 | 0.989 | 0.0083 |
| **clean** | **0.9336** | — | 0.9967 | 0.911 | 0.801 | 0.994 | 0.0050 |
| overexposed | 0.9435 | +0.0100 | 0.9967 | 0.912 | 0.797 | 0.988 | 0.0100 |

(Clean reads 0.9336 rather than 0.9401 because this is a 602-image stratified
subsample, not the full 3,625 — the stratification over-weights rare classes relative to
the natural test distribution. Every Δ is against this subsample's own clean row, so the
comparisons are internally consistent.)

### Finding 1 — the model is robust to exactly what it was trained on, and only that

Rank the corruptions by damage and the training recipe falls out of the table. Motion
blur, JPEG compression, white balance, brightness-up, off-angle framing — all in
`src/augmentation.py`, all cost ≤ 0.03. Background replacement, also in the training
augmentation at p=0.4, costs 0.06.

**Gaussian sensor noise is the one corruption family that is *not* in the training
augmentation, and it is by far the most damaging: −0.39 at a severity (std ≈ 25–38 grey
levels) that is unpleasant but still a recognisable photo.** That is not a coincidence,
and it is the honest reading of this whole table: the measured robustness is *evidence
of augmentation coverage*, not evidence of general robustness. Every number here is
therefore a **lower bound** on the real field drop — real photos will differ along axes
nobody thought to augment.

Actionable: add a noise transform to the training augmentation and re-measure. That is a
Phase 5 ablation row, not a guess.

### Finding 2 — underexposure is the realistic failure mode to worry about

−0.17 from a 30–45% brightness reduction, and the second-highest confidently-wrong rate
(2.2%). Shooting a leaf in shade or indoors is completely ordinary user behaviour,
unlike severe sensor noise. Overexposure, by contrast, costs nothing (it slightly
*helps*), which fits a dataset of brightly and evenly lit studio images.

### Finding 3 — the confidence tier degrades gracefully, which is the good news

Under `field_composite`, top-1 falls to 0.7641 but **HIGH-tier accuracy holds at 0.971**
— the model does not stay confident while becoming wrong. Instead HIGH coverage
collapses from 80.1% to 46.0%: more inputs get routed to MODERATE/LOW, where the app
shows a caution banner or refuses to diagnose. That is the selective-prediction system
working under shift, and it is the strongest argument for the tier design.

The exception is severe noise, where the gate breaks down properly: HIGH accuracy drops
to 0.761 and **11.3% of all images become confidently wrong** (versus 0.5% clean). When
the input is far enough out of distribution, softmax confidence stops carrying
information — which is precisely the argument for the Phase 4 OOD gate, since a
confidence threshold alone cannot catch this.

### Finding 4 — top-3 is much more durable than top-1

Under the field composite, top-1 loses 0.17 but top-3 loses only 0.06 (0.9967 → 0.9352).
The model usually still has the right answer in hand; shift degrades its *ranking*
between visually similar classes rather than its understanding. This is what makes the
MODERATE tier's top-3 list genuinely useful rather than a consolation prize.

---

## 3. What this does and does not license you to say

**Can say:** "Under stacked field-like corruption the production model drops from 0.93
to 0.76 top-1, but HIGH-tier accuracy holds at 0.97 because the tier system routes the
degraded inputs away from a confident diagnosis. The corruption it has never been
trained on — sensor noise — is where both accuracy and the confidence gate fail
together."

**Cannot say:** "The model gets 76% on real field photos." Nothing here has touched a
real field photo. Section 1 is still open.

Reproduce:

```bash
python -m src.eval_domain_shift            # 600 images, seed 42
python -m src.eval_domain_shift --n 3625   # full test split
```
