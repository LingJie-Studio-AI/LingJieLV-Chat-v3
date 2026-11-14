import argparse, json, os, time, math, sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.build import get_preset, get_config, build_lm, get_tok
from src.data import SFTDS, collate_lm

ROOT = os.environ.get('LJ_DATA_ROOT', 'data')

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
p.add_argument('--epochs', type=int, default=0)
p.add_argument('--bs', type=int, default=0)
p.add_argument('--lr', type=float, default=0.0)
p.add_argument('--warmup', type=int, default=0)
p.add_argument('--max_steps', type=int, default=150000)
p.add_argument('--init', default='')
p.add_argument('--out', default='')
p.add_argument('--log', default='')
p.add_argument('--data', default='')
args = p.parse_args()

pr = get_preset(args.variant); hp = pr['hp']
epochs = args.epochs or hp['sft_epochs']; bs = args.bs or hp['sft_bs']
lr = args.lr or hp['sft_lr']; warmup = args.warmup or hp['sft_warmup']
data = args.data or os.path.join(ROOT, pr['sft_dir'], 'sft_v4.jsonl')
init = args.init or f'ckpt/{args.variant}/pretrain_final.pt'
out = args.out or f'ckpt/{args.variant}/sft_final.pt'
log = args.log or f'logs/{args.variant}_sft.jsonl'

cfg = get_config(args.variant)
model = build_lm(args.variant, cfg).cuda()
model.load_state_dict(torch.load(init, map_location='cpu')['model'])
tok = get_tok(args.variant)
ds = SFTDS(data, tok)
dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True,
                collate_fn=lambda b: collate_lm(b, tok.pad))
steps_per_epoch = len(dl); total = min(args.max_steps, steps_per_epoch * epochs)
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def lr_at(s):
    if s < warmup: return lr * (s + 1) / warmup
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))


os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
logf = open(log, 'a', encoding='utf-8'); step = 0; t0 = time.time(); model.train()
print(f'[{args.variant}] rows={len(ds)} steps={total} bs={bs}', flush=True)
for ep in range(epochs):
    for x, y in dl:
        for g in opt.param_groups: g['lr'] = lr_at(step)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hidden, aux = model(x)
            logits = torch.nn.functional.linear(hidden, model.tok_emb.weight)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                y[:, 1:].reshape(-1), ignore_index=-100)
        (loss + 0.01 * aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            el = time.time() - t0; sps = (step + 1) / max(el, 1e-9)
            rec = {'step': step, 'ep': ep, 'loss': round(loss.item(), 4), 'sps': round(sps, 3),
                   'eta_h': round((total - step) / max(sps, 1e-9) / 3600, 2), 'elapsed': round(el, 1)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
        step += 1
        if step >= total: break
    if step >= total: break

torch.save({'model': model.state_dict(), 'step': step, 'cfg': vars(cfg)}, out)
print('SFT DONE', step, round(time.time() - t0, 1), flush=True)
