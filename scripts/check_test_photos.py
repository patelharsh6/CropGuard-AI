"""
Validate a photo test set before scoring it.

Catches the things that silently ruin an evaluation run: a folder name that does not
map to any class (its photos are skipped, so accuracy is computed on fewer images than
you think), an unreadable or truncated file, a duplicate pasted twice, an image too
small to survive the 224x224 resize.

Run:  python scripts/check_test_photos.py                       # checks real_world_test/
      python scripts/check_test_photos.py --dir web_sourced_test
      python scripts/check_test_photos.py --fix                 # rename folders to canonical

Nothing here judges whether a label is *correct* — no script can do that. It only
checks that what you pasted will actually be read.
"""

import argparse
import hashlib
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from PIL import Image

# Deliberately does NOT import from src/: that pulls in TensorFlow and turns a
# two-second sanity check into a thirty-second one.
CSV_PATH = os.path.join('data', 'dataset_split.csv')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
MIN_SIDE = 224           # the model input size; smaller means upscaling blur
TARGET_PHOTOS = 15       # plan.md Phase 1 target is 15-25
TARGET_CLASSES = 6       # spread matters more than depth


# Fallback for a fresh clone where data/ has not been downloaded yet.
FALLBACK_CLASSES = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]


def canonical_classes():
    if os.path.exists(CSV_PATH):
        return sorted(pd.read_csv(CSV_PATH)['label'].unique())
    return sorted(FALLBACK_CLASSES)


def _norm(s: str) -> str:
    """Aggressively normalise a folder name for fuzzy matching."""
    s = s.lower()
    s = s.replace('(maize)', '').replace('maize', 'corn')
    s = s.replace('two-spotted_spider_mite', '').replace('two spotted spider mite', '')
    s = s.replace('cercospora_leaf_spot', '').replace('cercospora leaf spot', '')
    s = re.sub(r'[^a-z]+', '', s)
    return s


def build_lookup(classes):
    """folder name (normalised) -> canonical class."""
    lookup = {}
    for c in classes:
        for variant in (c, _sanitize(c)):
            lookup[_norm(variant)] = c
    return lookup


def _sanitize(label: str) -> str:
    return (label.replace('_(maize)', '')
                 .replace('Tomato___Tomato_', 'Tomato___')
                 .replace('Spider_mites Two-spotted_spider_mite', 'Spider_mites')
                 .replace('Cercospora_leaf_spot Gray_leaf_spot', 'Cercospora_Gray_leaf_spot')
                 .replace('___', '_')
                 .rstrip('_'))


def p(line=''):
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='real_world_test')
    ap.add_argument('--fix', action='store_true',
                    help='rename fuzzily-matched folders to their canonical names')
    args = ap.parse_args()

    root = args.dir
    if not os.path.isdir(root):
        raise SystemExit(f"No '{root}/' directory.")

    classes = canonical_classes()
    lookup = build_lookup(classes)

    p('=' * 74)
    p(f'Checking {root}/')
    p('=' * 74)

    problems = 0
    per_class = defaultdict(list)
    hashes = defaultdict(list)
    ood_count = 0

    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        if entry.startswith('_') or entry.startswith('.'):
            if entry == '_ood':
                # Recursive: the Phase 4 negative set is filed under per-category
                # subfolders (_ood/face/, _ood/whole_plant_crop/, ...), which a
                # flat listdir would report as zero.
                ood_count = sum(
                    1 for _, _, files in os.walk(sub) for f in files
                    if f.lower().endswith(IMAGE_EXTS))
            continue

        canon = lookup.get(_norm(entry))
        if canon is None:
            imgs = [f for f in os.listdir(sub) if f.lower().endswith(IMAGE_EXTS)]
            p(f'\n[FOLDER NAME NOT RECOGNISED]  {entry}/  ({len(imgs)} image(s) would be SKIPPED)')
            p('  Rename it to one of the folders that already exist in this directory.')
            problems += 1
            continue

        if canon != entry:
            if args.fix:
                dst = os.path.join(root, canon)
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(sub):
                    os.replace(os.path.join(sub, f), os.path.join(dst, f))
                os.rmdir(sub)
                p(f'\n[FIXED] renamed "{entry}" -> "{canon}"')
                sub = dst
            else:
                p(f'\n[NON-STANDARD NAME] "{entry}" reads as "{canon}" '
                  f'(works, but run --fix to tidy)')

        for fname in sorted(os.listdir(sub)):
            path = os.path.join(sub, fname)
            if not os.path.isfile(path):
                continue
            if fname in ('.gitkeep',):
                continue
            if not fname.lower().endswith(IMAGE_EXTS):
                p(f'  [IGNORED, not an image] {canon}/{fname}')
                continue

            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    w, h = im.size
                    im.convert('RGB')
            except Exception as e:
                p(f'  [UNREADABLE] {canon}/{fname}: {e}')
                problems += 1
                continue

            if min(w, h) < MIN_SIDE:
                p(f'  [TOO SMALL] {canon}/{fname}: {w}x{h}, '
                  f'shorter side < {MIN_SIDE}px — it will be upscaled and blurred')
                problems += 1
            elif max(w, h) / min(w, h) > 3:
                p(f'  [ODD ASPECT] {canon}/{fname}: {w}x{h} — the square resize '
                  f'will distort this badly')

            with open(path, 'rb') as f:
                hashes[hashlib.md5(f.read()).hexdigest()].append(f'{canon}/{fname}')
            per_class[canon].append(fname)

    dupes = {h: v for h, v in hashes.items() if len(v) > 1}
    if dupes:
        p('')
        for _, files in dupes.items():
            p(f'  [DUPLICATE] identical file in {len(files)} places: {", ".join(files)}')
            problems += 1

    total = sum(len(v) for v in per_class.values())
    p('')
    p('-' * 74)
    if per_class:
        p(f"{'class':<52}{'photos':>8}")
        for c in classes:
            if per_class.get(c):
                p(f'{c:<52}{len(per_class[c]):>8}')
    p('-' * 74)
    p(f'Photos: {total}   Classes covered: {len(per_class)}/17   _ood/: {ood_count}')
    p('')

    if problems:
        p(f'{problems} problem(s) above need fixing.')
    else:
        p('No structural problems found.')

    if total == 0:
        p('\nNothing to score yet. See real_world_test/README.md for what to shoot.')
    elif total < TARGET_PHOTOS or len(per_class) < TARGET_CLASSES:
        need_p = max(0, TARGET_PHOTOS - total)
        need_c = max(0, TARGET_CLASSES - len(per_class))
        bits = []
        if need_p:
            bits.append(f'{need_p} more photo(s)')
        if need_c:
            bits.append(f'{need_c} more class(es)')
        p(f'\nUsable, but thin: add {" and ".join(bits)} for a more meaningful number.')
        p(f'You can score it now anyway:')
        p(f'  python -m src.eval_real_world --dir {root}')
    else:
        p(f'\nReady to score:')
        p(f'  python -m src.eval_real_world --dir {root}')

    p('=' * 74)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
