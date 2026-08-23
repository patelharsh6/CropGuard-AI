# Web-sourced test set (Phase 1 stopgap)

20 leaf photos harvested from Wikimedia Commons, hand-vetted, and scored with
`python -m src.eval_real_world --dir web_sourced_test`. Results:
`outputs/web_sourced_report.json`, discussion in `docs/DOMAIN_SHIFT.md` §1b.

**Headline: 0.40 top-1, versus 0.9401 on the clean benchmark.**

---

## What this is, and what it is not

This exists because the hand-taken set in `real_world_test/` is blocked — no plants
were available to photograph. It is **weaker evidence** than that set, in three
specific ways, and no conclusion drawn from it should ignore them:

1. **The labels come from the uploader's caption, not from anyone who examined the
   plant.** Mitigated but not solved: 11 of the 20 come from university plant-pathology
   archives (Bugwood, Iowa State, Clemson, Virginia Tech, Colorado State, SDSU, TNAU),
   where the caption is an expert determination. The rest are hobbyist uploads.
2. **n = 20.** The 95% Wilson interval on 0.40 is **[0.22, 0.61]** — enormous. This
   result identifies a *direction and rough magnitude*, not a number to quote to 4
   decimal places.
3. **Selection is not random.** These are the images Commons happens to have for these
   17 classes, which is heavily skewed toward dramatic, photogenic symptoms and away
   from healthy foliage. Seven of the 17 classes have no image at all.

`real_world_test/` remains the thing that would settle this. This does not replace it.

## How it was built

```bash
python scripts/harvest_commons.py --out <staging> --per-class 4
```

That queries the Commons API per class (common name + pathogen binomial), downloads
thumbnails, and records title, page URL, license, author and caption for every file.
Then **every candidate was viewed individually** before being kept.

49 candidates were harvested; 20 were accepted. What the vetting removed, and why it
was necessary:

| Rejected | Count | Example |
|---|---|---|
| Not a photograph | 6 | an 1882 book scan, a disease-cycle diagram, a histological line drawing, a phylogenetic tree |
| Wrong species | 2 | *Ipomoea batatas* (sweet potato) returned for `Potato___healthy`; a pepper leaf for `Tomato___Bacterial_spot` |
| **Wrong label from the search** | 2 | two images captioned *Septoria leaf spot* landed in the `Tomato___healthy` folder |
| Wrong plant part | 2 | a lesion on the maize stalk sheath; a macro of leaf trichomes |
| Symptoms don't match the caption | 1 | a sunken concentric lesion on an entire-margined leaf captioned *Passalora fulva* |
| Whole plant / canopy → routed to `_ood/` | 4 | corn canopy, young corn plant with weeds, tomato plant with fruit |
| Duplicates across search passes | 12 | |

That table *is* the argument for why "just grab images off the internet" is not a
substitute for labelling your own: **more than half the candidates were unusable, and
two arrived with actively wrong labels.** An unvetted scrape of these 49 files would
have produced a confidently meaningless accuracy number.

## Layout

```
web_sourced_test/
    <PlantVillage class name>/*.jpg     scored by eval_real_world
    _ood/                               4 whole-plant/macro images, NOT scored (Phase 4)
    provenance.json                     per-image source, license, author, caption,
                                        plus my `setting`, `label_confidence` and review note
```

Coverage is 10 of 17 classes: gray leaf spot (1), common rust (3), northern leaf blight
(2), potato late blight (3), potato healthy (2), tomato early blight (1), tomato late
blight (3), tomato leaf mold (1), tomato septoria (3), tomato healthy (1). **No images**
for potato early blight, tomato bacterial spot, target spot, spider mites, yellow leaf
curl virus, mosaic virus, or corn healthy — Commons has too few usable photos.

`setting` in `provenance.json` distinguishes `field` (14), `studio` (3), `semi-studio`,
`macro`, `indoor-detached` (1 each), because a detached leaf on a black background is
not a test of domain shift.

## PlantVillage leakage

Every accepted image was checked for the PlantVillage look — a single detached leaf,
centred, on a uniform grey background. None matched. The three `studio` images are on
black or white, not PlantVillage grey, and two of them are multi-leaflet. Leakage is
unlikely, though it has not been ruled out by pixel-level matching against the training
set.

## Licensing

All 20 are freely licensed (CC0, CC BY 2.0/3.0/4.0, CC BY-SA 2.0/3.0/4.0). Attribution
for each — author, licence and the Commons page — is in `provenance.json`, which must
travel with the images if they are redistributed. Please keep it in the repository.

## Reproduce

```bash
python -m src.eval_real_world --dir web_sourced_test --json outputs/web_sourced_report.json
```
