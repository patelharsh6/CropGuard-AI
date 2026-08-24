"""
Harvest the Phase 4 out-of-distribution negative set from Wikimedia Commons.

The classifier is closed-set over 17 leaf classes: every input gets a leaf label,
including a photo of a chair. `src/ood.py` needs negatives to measure how well an
OOD score separates "this is a crop leaf" from "this is anything else". This script
downloads them into a staging directory, with provenance, exactly the way
scripts/harvest_commons.py builds the web-sourced leaf set.

Why internet images are acceptable here, when real_world_test/README.md forbids them
for the *labelled* set: the label is "not a single crop leaf", which anyone can verify
by looking, and there is no training distribution to leak into a negative. The one
real hazard runs the other way — a category like `whole_plant_crop` can pull an image
that is effectively a training-style leaf close-up, which would count as a false
negative. Those are the hard negatives the gate most needs, so they are kept but must
be eyeballed; anything already in web_sourced_test/ is skipped by Commons title.

Downloads to staging only. Nothing enters real_world_test/_ood/ before review.

Run:  python scripts/harvest_ood.py --out <staging_dir> --per-category 9
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harvest_commons import (HEADERS, REJECT_SUBSTRINGS, _strip_html, image_info,
                             search)

# Categories chosen as things a user could plausibly point the camera at, ordered
# roughly easy -> hard. The last four matter most: they are plant material, so a
# detector that has merely learned "green and organic" fails on them while sailing
# through faces and furniture.
QUERIES = {
    'face': ['portrait photograph person face', 'human face closeup photo'],
    'hand': ['human hand photograph', 'open palm hand photo'],
    'furniture': ['wooden chair photograph', 'office desk table photo',
                  'sofa living room photograph'],
    'sky': ['blue sky clouds photograph', 'overcast sky photo'],
    'wall': ['blank painted wall texture', 'plaster wall surface photograph',
             'brick wall photograph'],
    'text': ['printed book page photograph', 'newspaper page scan',
             'handwritten note paper photograph'],
    'animal': ['domestic cat photograph', 'dog portrait photograph',
               'garden bird photograph'],
    'vehicle_street': ['parked car photograph street', 'urban street scene photograph'],
    'device': ['computer keyboard photograph', 'smartphone on table photograph'],
    'soil_ground': ['bare soil field photograph', 'gravel ground texture',
                    'garden soil close up'],
    # --- hard negatives: plant material that is not a single crop leaf ---
    # These queries are worded to land on photographs. The obvious phrasings
    # ("tomato plant", "corn cob") rank 19th-century seed-catalogue engravings and
    # botanical plates first on Commons, which are not what a camera produces.
    'whole_plant_crop': ['tomato plants greenhouse photograph',
                         'potato field flowering photograph',
                         'maize field summer photograph',
                         'tomato plant growing garden photo',
                         'young maize plants field photo'],
    'other_foliage': ['oak tree leaves photograph', 'lawn grass close up',
                      'fern fronds photograph', 'houseplant leaves photograph'],
    'flower': ['garden flower close up photograph', 'rose flower photograph'],
    'fruit_veg': ['ripe tomatoes photograph vegetable',
                  'potato tubers photograph food',
                  'sweetcorn cobs photograph food',
                  'vegetable market stall photograph'],
}

# Beyond harvest_commons.REJECT_SUBSTRINGS: the plant categories otherwise fill up
# with scans of seed catalogues, herbarium sheets and botanical engravings.
EXTRA_REJECT = ('catalog', 'catalogue', 'engraving', 'woodcut', 'lithograph',
                'herbarium', 'botanical_plate', 'plate_', 'etching', 'painting',
                'sketch', 'stamp', 'postcard')


def _seen_titles():
    """Commons titles already used by the web-sourced leaf set — never reuse them."""
    path = os.path.join('web_sourced_test', 'provenance.json')
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        prov = json.load(f)
    out = set()
    for key in ('accepted', 'rejected'):
        for rec in prov.get(key, []) or []:
            if rec.get('commons_title'):
                out.add(rec['commons_title'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='staging directory')
    ap.add_argument('--per-category', type=int, default=9)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--only', default=None, help='comma-separated category names')
    args = ap.parse_args()

    wanted = QUERIES
    if args.only:
        keys = [k.strip() for k in args.only.split(',')]
        wanted = {k: v for k, v in QUERIES.items() if k in keys}
        missing = set(keys) - set(wanted)
        if missing:
            raise SystemExit('unknown category(ies): %s' % missing)

    skip = _seen_titles()
    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for cat, queries in wanted.items():
        titles, seen = [], set()
        for q in queries:
            for t in search(q, args.per_category * 3):
                low = t.lower()
                if (t in seen or t in skip
                        or any(b in low for b in REJECT_SUBSTRINGS)
                        or any(b in low for b in EXTRA_REJECT)):
                    continue
                seen.add(t)
                titles.append(t)
            time.sleep(1.5)

        infos = image_info(titles[:args.per_category * 4], args.width)
        ordered = [infos[t] for t in titles if t in infos
                   and (infos[t].get('mime') or '').startswith('image/')]
        cat_dir = os.path.join(args.out, cat)
        os.makedirs(cat_dir, exist_ok=True)
        kept = 0

        for info in ordered:
            if kept >= args.per_category:
                break
            stem = info['title'].replace('File:', '').replace(' ', '_')
            stem = ''.join(ch if (ch.isalnum() or ch in '._-') else '_'
                           for ch in stem)[:80]
            fname = '%02d_%s' % (kept, stem)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fname += '.jpg'
            path = os.path.join(cat_dir, fname)
            try:
                req = urllib.request.Request(info['thumb_url'], headers=HEADERS)
                with urllib.request.urlopen(req, timeout=45) as r:
                    blob = r.read()
                if len(blob) < 5000:
                    continue
                with open(path, 'wb') as f:
                    f.write(blob)
            except Exception as e:
                print(('  ! %s: %s' % (info['title'], e))
                      .encode('ascii', 'replace').decode('ascii'))
                continue

            manifest.append({
                'category': cat,
                'file': path.replace('\\', '/'),
                'commons_title': info['title'],
                'page_url': info['page_url'],
                'license': info['license'],
                'artist': _strip_html(info['artist']),
                'source_caption': _strip_html(info['description']),
            })
            kept += 1
            time.sleep(1.0)

        print('%s: %d candidates' % (cat, kept))

    mpath = os.path.join(args.out, 'manifest.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print('\n%d candidates -> %s' % (len(manifest), mpath))


if __name__ == '__main__':
    main()
