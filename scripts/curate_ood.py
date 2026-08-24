"""
Move vetted Commons candidates from a staging dir into real_world_test/_ood/.

`scripts/harvest_ood.py` downloads candidates; this script records what a human (or
the agent, via contact sheets) decided about each one and writes the result. Every
candidate ends up in `real_world_test/_ood/provenance.json` as accepted or rejected
*with a reason*, the same discipline `web_sourced_test/provenance.json` follows —
a negative set whose contents were never examined is not evidence.

Two things happen on the way in:

  * images are downscaled to a 640 px long edge (JPEG q85). The pipeline resizes to
    224x224 regardless, and unlike `data/` this directory is committed, so shipping
    1024 px originals would add ~40 MB to the repo for no measurable difference.
  * the category folder can be corrected, because Commons search results do not
    always land in the category that asked for them.

Reject/recategorise decisions live in REJECTS / RECATEGORISE below, keyed by
`<category>/<filename-prefix>`, so the rationale is in version control rather than in
someone's shell history.

Run:  python scripts/curate_ood.py --staging <dir> [--staging <dir2>] [--dry-run]
"""

import argparse
import json
import os

import cv2
import numpy as np

DEST = os.path.join('real_world_test', '_ood')
PROVENANCE = os.path.join(DEST, 'provenance.json')
MAX_EDGE = 640
JPEG_QUALITY = 85

# key = '<category>/<filename prefix>' (prefix match), value = why it was dropped.
REJECTS = {
    # Not photographs. The model's failure mode on drawings is uninteresting and
    # they are not something a phone camera produces.
    'furniture/05_Book_of_styles': 'line drawing, not a photograph',
    'vehicle_street/05_Kirchner': 'painting, not a photograph',
    'whole_plant_crop/00_Bolgiano': 'seed-catalogue engraving, not a photograph',
    'whole_plant_crop/01_Satoimo': 'archival drawing, not a photograph',
    'whole_plant_crop/03_Burpee': 'seed-catalogue engraving, not a photograph',
    'whole_plant_crop/04_Annual': 'seed-catalogue engraving, not a photograph',
    'hand/05_Palm_leaf': 'postage-stamp scan, and depicts a palm-leaf fan',
    # Near-duplicates. Several copies of one scene would let a single easy or hard
    # image dominate a per-category rate computed over ~9 images.
    'device/05_Camera_zoom': 'near-duplicate of device/00 (same zoom-burst series)',
    'device/07_Camera_zoom': 'near-duplicate of device/00 (same zoom-burst series)',
    'fruit_veg/03_013Stewed': 'near-duplicate of fruit_veg/02 (same meal)',
    'fruit_veg/04_013Stewed': 'near-duplicate of fruit_veg/02 (same meal)',
    'text/02_Twenty-five': 'near-duplicate plate from the same 1895 album',
    'text/03_Twenty-five': 'near-duplicate plate from the same 1895 album',
    'text/05_Twenty-five': 'near-duplicate plate from the same 1895 album',
    'wall/00_Blue_roman': 'near-duplicate archaeological find photo (kept 2 of 4)',
    'wall/01_Painted_roman': 'near-duplicate archaeological find photo (kept 2 of 4)',
    'fruit_veg/00_Childs': 'seed-catalogue engraving, not a photograph',
    'fruit_veg/07_Lucas_van': 'painting, not a photograph',
    'whole_plant_crop/05_Greenhouse': 'greenhouse exterior — a building, already '
                                      'covered by the built-environment categories',
}

# Commons put these in the wrong bucket; the per-category table is only readable if
# the folder says what the picture is.
RECATEGORISE = {
    'hand/07_Hand-book': 'text',        # scan of an 1892 book page, no hand visible
    'flower/02_Singapore': 'vehicle_street',  # aerial of a building complex
    'fruit_veg/02_Stuckenia': 'other_foliage',  # pondweed mat, not produce
}


def _prefix_key(cat, fname):
    return f'{cat}/{fname}'


def _decide(cat, fname):
    """(action, detail) for one candidate: 'reject' + reason, or 'accept' + category."""
    key = _prefix_key(cat, fname)
    for pref, reason in REJECTS.items():
        if key.startswith(pref):
            return 'reject', reason
    for pref, newcat in RECATEGORISE.items():
        if key.startswith(pref):
            return 'accept', newcat
    return 'accept', cat


def _downscale(src, dst):
    buf = np.fromfile(src, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'unreadable image: {src}')
    h, w = img.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f'encode failed: {src}')
    enc.tofile(dst)
    return img.shape[1], img.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staging', action='append', required=True,
                    help='staging dir from harvest_ood.py (repeatable)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    accepted, rejected = [], []
    for staging in args.staging:
        mpath = os.path.join(staging, 'manifest.json')
        with open(mpath, encoding='utf-8') as f:
            manifest = json.load(f)

        for rec in manifest:
            src = rec['file']
            cat = rec['category']
            fname = os.path.basename(src)
            action, detail = _decide(cat, fname)
            if action == 'reject':
                rejected.append({**rec, 'reject_reason': detail})
                continue

            out_name = os.path.splitext(fname)[0] + '.jpg'
            rel = f'{detail}/{out_name}'
            entry = {
                'file': rel,
                'category': detail,
                'original_category': cat,
                'commons_title': rec['commons_title'],
                'page_url': rec['page_url'],
                'license': rec['license'],
                'artist': rec['artist'],
                'source_caption': rec['source_caption'],
            }
            if not args.dry_run:
                out_dir = os.path.join(DEST, detail)
                os.makedirs(out_dir, exist_ok=True)
                # Two staging dirs can produce the same '<nn>_<title>' name.
                dst = os.path.join(out_dir, out_name)
                suffix = 1
                while os.path.exists(dst):
                    stem = os.path.splitext(out_name)[0]
                    dst = os.path.join(out_dir, f'{stem}_{suffix}.jpg')
                    suffix += 1
                entry['file'] = f'{detail}/{os.path.basename(dst)}'
                w, h = _downscale(src, dst)
                entry['size'] = [w, h]
            accepted.append(entry)

    by_cat = {}
    for e in accepted:
        by_cat[e['category']] = by_cat.get(e['category'], 0) + 1

    print(f'accepted {len(accepted)}, rejected {len(rejected)}')
    for c in sorted(by_cat):
        print(f'  {c:18s} {by_cat[c]}')

    if args.dry_run:
        return

    prov = {
        'purpose': ('Phase 4 OOD negative set — images that are NOT a single crop '
                    'leaf. Scored by src/ood.py; see docs/OOD.md.'),
        'source': 'Wikimedia Commons via scripts/harvest_ood.py',
        'curation': ('every candidate viewed as a contact sheet before acceptance; '
                     'rejections and their reasons are listed below. Images are '
                     'downscaled to a 640 px long edge (the pipeline resizes to 224) '
                     'because this directory is committed.'),
        'counts': {'accepted': len(accepted), 'rejected': len(rejected),
                   'by_category': by_cat},
        'accepted': accepted,
        'rejected': rejected,
    }
    os.makedirs(DEST, exist_ok=True)
    with open(PROVENANCE, 'w', encoding='utf-8') as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    print(f'provenance -> {PROVENANCE}')


if __name__ == '__main__':
    main()
