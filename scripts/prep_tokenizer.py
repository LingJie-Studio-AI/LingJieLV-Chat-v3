import argparse, json, os, array, sys
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.presets import get_preset

ROOT = os.environ.get('LJ_DATA_ROOT', 'data')

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
p.add_argument('--src', default='')
p.add_argument('--out_dir', default='')
p.add_argument('--max_chars', type=int, default=25_000_000)
p.add_argument('--val_tokens', type=int, default=8_000_000)
args = p.parse_args()

pr = get_preset(args.variant)
out_dir = args.out_dir or os.path.join(ROOT, pr['data_dir'])
src = args.src or os.path.join(ROOT, pr['data_dir'], 'pretrain_t2t.jsonl')
TOK_PATH = os.path.join(out_dir, os.path.basename(pr['tokenizer']))
os.makedirs(out_dir, exist_ok=True)


def iter_text(max_chars):
    n = 0
    with open(src, encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                t = json.loads(line).get('text') or ''
            except Exception:
                continue
            if not t.strip():
                continue
            yield t
            n += len(t)
            if n >= max_chars:
                break


print('training tokenizer vocab', pr['tok_vocab'], 'min_freq', pr['tok_min_freq'], flush=True)
tok = ByteLevelBPETokenizer()
tok.train_from_iterator(iter_text(args.max_chars), vocab_size=pr['tok_vocab'],
                        min_frequency=pr['tok_min_freq'],
                        special_tokens=['<|pad|>', '<|end|>', '<|user|>', '<|assistant|>', '<|image|>', '<|system|>'])
tok.save(TOK_PATH)
print('saved tokenizer', os.path.getsize(TOK_PATH), flush=True)

eos = tok.token_to_id('<|end|>')
full = os.path.join(out_dir, 'tokens_all.bin')
buf = array.array('H'); n = 0
with open(full, 'wb') as f, open(src, encoding='utf-8', errors='ignore') as srcf:
    batch = []
    for line in srcf:
        try:
            t = json.loads(line).get('text') or ''
        except Exception:
            continue
        if not t:
            continue
        batch.append(t)
        if len(batch) >= 2000:
            for e in tok.encode_batch(batch, add_special_tokens=False):
                buf.extend(e.ids); buf.append(eos); n += len(e.ids) + 1
            batch = []
            if len(buf) > 4_000_000:
                f.write(buf.tobytes()); buf = array.array('H')
    if batch:
        for e in tok.encode_batch(batch, add_special_tokens=False):
            buf.extend(e.ids); buf.append(eos); n += len(e.ids) + 1
    f.write(buf.tobytes())
print('total tokens', n, flush=True)

val_bytes = args.val_tokens * 2
with open(full, 'rb') as f:
    f.seek(0, 2); size = f.tell()
with open(full, 'rb') as f, open(os.path.join(out_dir, 'val.bin'), 'wb') as v:
    f.seek(max(0, size - val_bytes)); v.write(f.read())
os.truncate(full, max(0, size - val_bytes))
os.rename(full, os.path.join(out_dir, 'train.bin'))
print('train tokens', max(0, size - val_bytes) // 2, 'val', args.val_tokens, flush=True)
