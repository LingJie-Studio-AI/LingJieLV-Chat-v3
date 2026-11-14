import argparse, glob, io, json, os, collections
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro', help='占位，两变体共用同一图文整理逻辑')
p.add_argument('--coco_glob', default=os.environ.get('LJ_COCO_GLOB', ''),
                help='COCO Captions parquet 分片通配符，例如 data/raw/coco/*.parquet')
p.add_argument('--flickr_glob', default=os.environ.get('LJ_FLICKR_GLOB', ''),
                help='Flickr30k captions parquet 分片通配符')
p.add_argument('--vqa_glob', default=os.environ.get('LJ_VQA_GLOB', ''),
                help='VQAv2 parquet 分片通配符')
p.add_argument('--out_dir', default=os.environ.get('LJ_DATA_ROOT', 'data') + '/vision')
p.add_argument('--max_vqa', type=int, default=40000)
args = p.parse_args()

IMG_DIR = os.path.join(args.out_dir, 'img')
os.makedirs(IMG_DIR, exist_ok=True)


def res_crop(im, s=224):
    w, h = im.size
    if w < h:
        nw, nh = s, int(round(h * s / w))
    else:
        nh, nw = s, int(round(w * s / h))
    im = im.resize((nw, nh), Image.BICUBIC)
    l, t = (nw - s) // 2, (nh - s) // 2
    return im.crop((l, t, l + s, t + s))


def save_image(r, nm):
    dst = os.path.join(IMG_DIR, nm)
    if os.path.exists(dst):
        return True
    try:
        im = res_crop(Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB'))
        im.save(dst, quality=92)
        return True
    except Exception:
        return False


def caption_shard(shard, kind):
    rows = []
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=64):
        for r in batch.to_pylist():
            nm = r.get('file_name') or r.get('filename') or r.get('question_id')
            caps = r.get('answer') if kind == 'coco' else r.get('caption')
            if isinstance(caps, str): caps = [caps]
            caps = [c for c in (caps or []) if c and str(c).strip().lower() not in ('none', '')]
            if not caps or not nm:
                continue
            if 'image' in r and isinstance(r.get('image'), dict):
                if not save_image(r, str(nm)):
                    continue
            for c in caps[:5]:
                rows.append({'img': str(nm), 'cap': c})
    return rows


def vqa_shard(shard, limit):
    rows = []
    pf = pq.ParquetFile(shard)
    for b in pf.iter_batches(batch_size=64):
        for r in b.to_pylist():
            img = r.get('image') or {}
            if not isinstance(img, dict) or not img.get('bytes'):
                continue
            q = r.get('question') or ''
            a = str(r.get('multiple_choice_answer') or '').strip().lower()
            if not a:
                ans = r.get('answers') or []
                cc = collections.Counter([str(x.get('answer', '')).strip().lower()
                                           for x in ans if isinstance(x, dict) and x.get('answer')])
                if not cc: continue
                a = cc.most_common(1)[0][0]
            if not q or not a or len(a.split()) > 3:
                continue
            nm = f"vqa_{r.get('question_id', len(rows))}.jpg"
            if not save_image(r, nm):
                continue
            rows.append({'img': nm, 'q': q, 'a': a})
            if len(rows) >= limit:
                return rows
    return rows


def main():
    caption_rows, vqa_rows = [], []
    jobs = []
    if args.coco_glob:
        jobs += [(f, 'coco') for f in sorted(glob.glob(args.coco_glob))]
    if args.flickr_glob:
        jobs += [(f, 'flickr') for f in sorted(glob.glob(args.flickr_glob))]
    if jobs:
        with ProcessPoolExecutor(max_workers=16) as ex:
            for rows in ex.map(lambda j: caption_shard(*j), jobs):
                caption_rows += rows
    if args.vqa_glob:
        for f in sorted(glob.glob(args.vqa_glob)):
            vqa_rows += vqa_shard(f, args.max_vqa)
            if len(vqa_rows) >= args.max_vqa:
                break

    align = os.path.join(args.out_dir, 'align_v3.jsonl')
    mm = os.path.join(args.out_dir, 'mm_vl_train.jsonl')
    with open(align, 'w', encoding='utf-8') as w:
        for r in caption_rows:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(mm, 'w', encoding='utf-8') as w:
        for r in caption_rows:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
        for r in vqa_rows:
            w.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'caption pairs={len(caption_rows)} vqa pairs={len(vqa_rows)}', flush=True)


main()
