import argparse, json, time, math, os, sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.build import get_preset, get_config, build_vl, get_tok, load_vision
from src.data import VLDS, collate_vl

ROOT = os.environ.get('LJ_DATA_ROOT', 'data')

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
p.add_argument('--epochs', type=int, default=1)
p.add_argument('--bs', type=int, default=0)
p.add_argument('--lr', type=float, default=0.0)
p.add_argument('--warmup', type=int, default=0)
p.add_argument('--max_steps', type=int, default=0)
p.add_argument('--init', default='')
p.add_argument('--out', default='')
p.add_argument('--log', default='')
p.add_argument('--vdata', default='')
args = p.parse_args()

pr = get_preset(args.variant); hp = pr['hp']
bs = args.bs or hp['align_bs']; lr = args.lr or hp['align_lr']
warmup = args.warmup or hp['align_warmup']; max_steps = args.max_steps or hp['align_max']
vdata = args.vdata or os.path.join(ROOT, pr['vision_dir'], 'align_v3.jsonl')
init = args.init or f'ckpt/{args.variant}/sft_final.pt'
out = args.out or f'ckpt/{args.variant}/align_final.pt'
log = args.log or f'logs/{args.variant}_align.jsonl'

cfg = get_config(args.variant); tok = get_tok(args.variant)
m = build_vl(args.variant, cfg, tok.img).cuda()
m.lm.load_state_dict(torch.load(init, map_location='cpu')['model'])
load_vision(args.variant, m)
for q in m.parameters(): q.requires_grad = False
for q in m.projector.parameters(): q.requires_grad = True
print('trainable_M', round(sum(q.numel() for q in m.parameters() if q.requires_grad) / 1e6, 2), flush=True)

ds = VLDS(vdata, tok)
dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=10, drop_last=True,
                collate_fn=lambda b: collate_vl(b, tok.pad), persistent_workers=True)
steps_per_epoch = len(dl); total_steps = min(max_steps, steps_per_epoch * args.epochs)
opt = torch.optim.AdamW(m.projector.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)


def lr_at(s):
    if s < warmup: return lr * (s + 1) / warmup
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))


os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
logf = open(log, 'a', encoding='utf-8'); step = 0; t0 = time.time(); m.train(); m.vision.eval()
print(f'[{args.variant}] rows={len(ds)} steps={total_steps}', flush=True)
for ep in range(args.epochs):
    for x, y, imgs in dl:
        for g in opt.param_groups: g['lr'] = lr_at(step)
        x, y, imgs = x.cuda(non_blocking=True), y.cuda(non_blocking=True), imgs.cuda(non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss, _ = m(x, imgs, y)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.projector.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            rec = {'step': step, 'ep': ep, 'loss': round(loss.item(), 4), 'elapsed': round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
        step += 1
        if step >= total_steps: break
    if step >= total_steps: break

torch.save({'model': m.lm.state_dict(), 'projector': m.projector.state_dict(), 'step': step, 'cfg': vars(cfg)}, out)
print('ALIGN DONE', step, round(time.time() - t0, 1), flush=True)
