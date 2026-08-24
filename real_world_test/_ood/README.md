# OOD negative set (Phase 4)

97 photographs of things that are **not a single crop leaf**, used to measure and
threshold the "is this even a leaf?" gate that now runs before any diagnosis is shown.

- Scored by `python -m src.ood` → `outputs/ood_report.json`
- Full analysis and the shipped threshold: **`docs/OOD.md`**
- Per-image licence, attribution, source caption and review decision:
  **`provenance.json`** (do not delete it — it is the only record of where these
  came from and why each was kept)

## Why internet images are allowed here

`real_world_test/README.md` forbids stock and internet images in the *class* folders,
for two reasons: the labels are unverified, and some circulating "field" photos are
themselves PlantVillage images, which would leak the training distribution into the
test. Neither applies to a negative:

- the label is "not a single crop leaf", which anyone can confirm by looking;
- a photo of a chair cannot leak the training distribution.

The one real hazard runs the other way — a `whole_plant_crop` query can return
something that is effectively a leaf close-up, which would count as a false negative
against the gate. That is why every candidate was reviewed rather than bulk-imported.

## How it was built

```bash
python scripts/harvest_ood.py --out <staging> --per-category 9
python scripts/curate_ood.py --staging <staging>
```

The harvester queries the Commons API per category and records title, page URL,
licence, author and caption for each file. The curator applies an explicit
accept/reject table (in the script, so the reasoning is in version control),
downscales to a 640 px long edge, and writes `provenance.json`.

**116 candidates harvested, 97 accepted, 19 rejected.** Rejections were: seed-catalogue
engravings, botanical plates and paintings (the plant queries rank these highly on
Commons and they are not photographs), a postage stamp, and near-duplicate frames that
would have let one scene dominate a category's rate. Two images were re-filed into the
category that matched what they actually showed.

Images are stored downscaled because this directory is committed, unlike `data/`. The
pipeline resizes everything to 224×224, so nothing is lost — 640 px keeps them
reviewable at ~7 MB total instead of ~40 MB.

## Categories

| | n |
|---|---|
| whole_plant_crop | 12 |
| fruit_veg | 9 |
| animal | 8 |
| device, flower, sky | 7 each |
| face, furniture, hand, other_foliage, soil_ground, text, vehicle_street | 6 each |
| wall | 5 |

`web_sourced_test/_ood/` holds 4 more (whole-plant and macro shots kept from the
Phase 1 harvest); `src/ood.py` reads both directories, for **n = 101**.

The last four categories in the list above are the ones that matter. Faces and
furniture are easy — any detector rejects them. Plant material that is not a single
crop leaf is where a gate either works or does not, and it is where the 8 images that
still pass the shipped threshold come from (see `docs/OOD.md` §6).

## Adding to it

Drop images into a category folder (or a new one — the walker uses folder names as
categories), then re-run `python -m src.ood --no-cache`. Update `provenance.json` by
re-running the curator rather than editing by hand. Note that adding negatives changes
the reported AUROC but **not** the threshold, which is set on the field leaf photos in
the class folders above.
