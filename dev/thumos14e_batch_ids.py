#!/usr/bin/env python3
"""IDs de vídeo para unha tanda de THUMOS14-E.

Orde PREDECLARADO: clases por número de instancias en validation, descendente.
Regra cega, fixada antes de ver ningún resultado de mAP.
Devolve validation+test das top-N clases, para que a tanda sexa avaliable.
"""
import argparse, collections, csv, json, pathlib

ROOT = pathlib.Path('/home/pablo.garcia.seijo/event_penguins/data/thumos14_events/thumos14e_v1')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--top-n', type=int, required=True)
    ap.add_argument('--only-new', action='store_true', help='omitir os xa convertidos')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / 'manifest.csv')))
    labels = {r['video_id']: {a['label'] for a in json.loads(r['annotations_json'])} for r in rows}

    inst = collections.Counter()
    for r in rows:
        if r['official_subset'] == 'validation':
            for a in json.loads(r['annotations_json']):
                inst[a['label']] += 1
    sel = {c for c, _ in inst.most_common(args.top_n)}

    out = [r['video_id'] for r in rows if sel & labels[r['video_id']]]
    if args.only_new:
        out = [v for v in out if not (ROOT / 'v2e' / v / 'conversion.json').exists()]
    print(' '.join(out))


if __name__ == '__main__':
    main()
