"""
Harvest candidate leaf-disease photos from Wikimedia Commons into a staging dir.

This builds the *web-sourced* evaluation set (web_sourced_test/), which is weaker
evidence than the hand-taken set in real_world_test/ — the labels come from the
uploader's caption, not from anyone who examined the plant. Commons is used rather
than a generic image search because every file carries a machine-readable
description, author and license, so provenance can be recorded per image.

Downloads to a staging directory only. Nothing enters web_sourced_test/ until a
human (or the agent, via visual review) has looked at each image and confirmed it is
(a) plausibly the captioned disease, (b) a single leaf or small cluster, and
(c) NOT a PlantVillage studio shot — a uniform grey/black background means the
image may literally be in the training set.

Run:  python scripts/harvest_commons.py --out <staging_dir> --per-class 4
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

API = 'https://commons.wikimedia.org/w/api.php'
# Wikimedia enforces a robot policy: a generic UA gets HTTP 429 on upload.wikimedia.org.
# The thumb URL returned by the API (with its utm params) must be used verbatim, and a
# Referer helps. No personal contact details are sent.
UA = ('CropGuardAI-research/1.0 (educational plant-disease classifier evaluation; '
      'contact via project repository)')
HEADERS = {'User-Agent': UA, 'Referer': 'https://commons.wikimedia.org/'}

# Search queries per class. Multiple phrasings because Commons captions are
# inconsistent — some use the common name, some the pathogen binomial.
QUERIES = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        'gray leaf spot maize', 'Cercospora zeae-maydis', 'grey leaf spot corn'],
    'Corn_(maize)___Common_rust_': [
        'common rust maize leaf', 'Puccinia sorghi', 'corn rust leaf'],
    'Corn_(maize)___Northern_Leaf_Blight': [
        'northern corn leaf blight', 'Exserohilum turcicum', 'Setosphaeria turcica'],
    'Corn_(maize)___healthy': [
        'Zea mays leaf close', 'maize leaf blade', 'corn leaf green healthy'],
    'Potato___Early_blight': [
        'potato early blight leaf', 'Alternaria solani potato leaf'],
    'Potato___Late_blight': [
        'potato late blight leaf', 'Phytophthora infestans potato leaf'],
    'Potato___healthy': [
        'Solanum tuberosum leaves plant', 'potato haulm leaves', 'potato leaflet'],
    'Tomato___Bacterial_spot': [
        'bacterial spot tomato leaf', 'Xanthomonas tomato leaf'],
    'Tomato___Early_blight': [
        'tomato early blight leaf', 'Alternaria solani tomato leaf'],
    'Tomato___Late_blight': [
        'tomato late blight leaf', 'late blight tomato foliage'],
    'Tomato___Leaf_Mold': [
        'tomato leaf mold', 'Passalora fulva', 'Cladosporium fulvum tomato'],
    'Tomato___Septoria_leaf_spot': [
        'septoria leaf spot tomato', 'Septoria lycopersici'],
    'Tomato___Spider_mites Two-spotted_spider_mite': [
        'spider mite damage tomato leaf', 'two-spotted spider mite leaf damage',
        'Tetranychus urticae damage leaf'],
    'Tomato___Target_Spot': [
        'target spot tomato leaf', 'Corynespora cassiicola tomato'],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        'tomato yellow leaf curl virus', 'TYLCV tomato'],
    'Tomato___Tomato_mosaic_virus': [
        'tomato mosaic virus leaf', 'tobacco mosaic virus tomato leaf'],
    'Tomato___healthy': [
        'Solanum lycopersicum leaves healthy', 'tomato leaf close up',
        'tomato seedling leaves'],
}

# Illustrations and microscopy are not photographs of a leaf in the field.
REJECT_SUBSTRINGS = ('.svg', '.pdf', '.tif', 'diagram', 'chart', 'map', 'logo',
                     'micrograph', 'microscope', 'drawing', 'illustration')


def _get(params: dict, retries: int = 5) -> dict:
    """Commons rate-limits anonymous clients hard; back off rather than hammer it."""
    params = {**params, 'format': 'json'}
    url = API + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f'  (HTTP {e.code}, waiting {wait}s)')
            time.sleep(wait)
    raise RuntimeError('unreachable')


def search(query: str, limit: int) -> list:
    data = _get({'action': 'query', 'list': 'search', 'srsearch': query,
                 'srnamespace': 6, 'srlimit': limit})
    return [hit['title'] for hit in data.get('query', {}).get('search', [])]


def image_info(titles: list, width: int) -> dict:
    """Batch imageinfo lookup: thumb URL, license, author, description."""
    out = {}
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        data = _get({
            'action': 'query', 'titles': '|'.join(chunk), 'prop': 'imageinfo',
            'iiprop': 'url|extmetadata|size|mime', 'iiurlwidth': width,
        })
        for page in data.get('query', {}).get('pages', {}).values():
            info = (page.get('imageinfo') or [None])[0]
            if not info:
                continue
            meta = info.get('extmetadata', {})
            out[page['title']] = {
                'title': page['title'],
                'mime': info.get('mime'),
                'width': info.get('width'),
                'height': info.get('height'),
                'thumb_url': info.get('thumburl') or info.get('url'),
                'file_url': info.get('url'),
                'page_url': info.get('descriptionurl'),
                'license': meta.get('LicenseShortName', {}).get('value'),
                'artist': meta.get('Artist', {}).get('value'),
                'description': meta.get('ImageDescription', {}).get('value'),
            }
    return out


def _strip_html(s):
    if not s:
        return None
    out, depth = [], 0
    for ch in s:
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return ' '.join(''.join(out).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='staging directory')
    ap.add_argument('--per-class', type=int, default=4)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--only', default=None,
                    help='comma-separated class names to harvest (default: all)')
    args = ap.parse_args()

    wanted = QUERIES
    if args.only:
        keys = [k.strip() for k in args.only.split(',')]
        wanted = {k: v for k, v in QUERIES.items() if k in keys}
        missing = set(keys) - set(wanted)
        if missing:
            raise SystemExit(f'unknown class(es): {missing}')

    os.makedirs(args.out, exist_ok=True)
    manifest = []

    for cls, queries in wanted.items():
        titles, seen = [], set()
        for q in queries:
            for t in search(q, args.per_class * 3):
                low = t.lower()
                if t in seen or any(bad in low for bad in REJECT_SUBSTRINGS):
                    continue
                seen.add(t)
                titles.append(t)
            time.sleep(1.5)

        infos = image_info(titles[:args.per_class * 4], args.width)
        # Keep original search order, then take the first N usable JPEGs.
        ordered = [infos[t] for t in titles if t in infos
                   and (infos[t].get('mime') or '').startswith('image/')]
        kept = 0
        cls_dir = os.path.join(args.out, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for info in ordered:
            if kept >= args.per_class:
                break
            stem = info['title'].replace('File:', '').replace(' ', '_')
            # Windows rejects several punctuation chars in filenames, and quotes
            # break downstream shell handling — keep it to alnum . _ -
            stem = ''.join(ch if (ch.isalnum() or ch in '._-') else '_'
                           for ch in stem)[:80]
            fname = f"{kept:02d}_{stem}"
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fname += '.jpg'
            path = os.path.join(cls_dir, fname)
            try:
                req = urllib.request.Request(info['thumb_url'], headers=HEADERS)
                with urllib.request.urlopen(req, timeout=45) as r:
                    blob = r.read()
                if len(blob) < 5000:
                    continue
                with open(path, 'wb') as f:
                    f.write(blob)
            except Exception as e:
                # Console is cp1252 on Windows; Commons titles are not.
                print(('  ! %s: %s' % (info['title'], e))
                      .encode('ascii', 'replace').decode('ascii'))
                continue

            manifest.append({
                'class': cls,
                'file': path.replace('\\', '/'),
                'commons_title': info['title'],
                'page_url': info['page_url'],
                'license': info['license'],
                'artist': _strip_html(info['artist']),
                'source_caption': _strip_html(info['description']),
            })
            kept += 1
            time.sleep(1.0)

        print(f'{cls}: {kept} candidates')

    mpath = os.path.join(args.out, 'manifest.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'\n{len(manifest)} candidates -> {mpath}')


if __name__ == '__main__':
    main()
