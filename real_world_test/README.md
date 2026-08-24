# Real-world test set (Phase 1)

**Status: 17 photos, one per class — still the project's biggest open credibility gap.**
More photos here now improve *two* results, not one: the field accuracy number, and
the OOD gate's threshold, which is a percentile of these 37 field images
(`docs/OOD.md` §8).

> 👉 **Adding photos? Read [`HOW_TO_ADD_PHOTOS.md`](HOW_TO_ADD_PHOTOS.md)** — step-by-step
> folder routing, what counts as a usable image, and a validator that checks your work.
> This file is the *why*; that one is the *how*.

Every image the model has ever seen is a single detached leaf on a uniform studio
background under controlled lighting (PlantVillage). Nothing in this repo yet says what
happens on a photo taken outdoors with a phone. This directory is where that evidence
goes, and `python -m src.eval_real_world` turns it into the number.

## Why you have to take these yourself

Stock photos and internet images are **not trustworthy ground truth**. Their labels are
unverified, their provenance is unknown, and some are themselves PlantVillage images —
which would leak the training distribution back into the "field" test and produce a
flatteringly wrong result. One anecdote already in the project makes the point: a stock
photo of unverified provenance scored 93.65%, above the HIGH gate, and there is no way
to know whether that was right.

A generated or scraped set here is worse than an empty directory, because an empty
directory is honest.

## How to shoot them

- **15–25 photos** total, spread across tomato, potato and corn, and across a few
  disease classes plus healthy. More classes beats more photos per class.
- **One leaf, filling the frame.** The model has no whole-plant category — a whole-plant
  shot returned 79.7% confidence on a wrong class. That is a Phase 4 OOD problem, not a
  Phase 1 accuracy problem, so keep it out of this set.
- **Natural light, natural background** — hand, soil, grass, other foliage. Do *not*
  reproduce the studio look; the whole point is the distribution the model has not seen.
- **Phone camera, as a user would hold it.** Slight blur, shadow and off-angle framing
  are wanted, not defects.
- Vary lighting across the set: overcast, direct sun, shade, indoor window.

## How to label them

Put each photo in the folder named for its true class. **You must be confident in the
label** — if you are not sure whether it is Early blight or Septoria, either get it
confirmed by a local extension service / plant clinic, or leave it out. An uncertain
label silently becomes a fake error in the report.

Folder names accept either form:

```
real_world_test/Tomato___Early_blight/IMG_0001.jpg      # PlantVillage folder name
real_world_test/Tomato_Early_blight/IMG_0001.jpg        # sanitized name from constants.ts
```

`.jpg`, `.jpeg`, `.png`, `.webp` and `.bmp` are read. Empty class folders are fine and
are simply skipped.

`_ood/` is ignored by `eval_real_world.py`. It is the Phase 4 negative set — anything
that is *not* a single crop leaf — and it is **populated** (97 curated Wikimedia
Commons photos across 14 categories; see `_ood/README.md`). The "take them yourself"
rule above does not apply there: "not a leaf" is a label anyone can verify by looking,
and a negative cannot leak the training distribution into a leaf test. It is scored by
`python -m src.ood`, not by `eval_real_world.py`.

## Then run

```bash
python scripts/check_test_photos.py   # validate folders/files first
python -m src.eval_real_world         # -> outputs/real_world_report.json
```

It reports top-1 and top-3 accuracy, the drop against the 0.9401 clean-test baseline,
the confidence-tier breakdown, and every confidently-wrong photo individually.

**Expect the number to be worse than 0.9401, and report it anyway.** The size of that
drop is the most interview-valuable single result in the project; hiding it would waste
the exercise. With 15–25 photos the confidence interval is wide — quote it as an
indicative gap, not a precise accuracy.

## Meanwhile

Two stand-ins exist. Both bound the answer; neither replaces this directory.

- `python -m src.eval_real_world --dir web_sourced_test` — 20 hand-vetted Wikimedia
  Commons photos. **0.40 top-1**, CI [0.22, 0.61]. Labels are uploader captions, so
  label error is baked in; see `web_sourced_test/README.md`.
- `python -m src.eval_domain_shift` — synthetic corruptions of the clean test split
  (background replacement, blur, lighting, JPEG, noise). A lower bound, since the
  corruptions overlap the training augmentation.

Photos you took and labelled yourself remove the one error source both of them have.
