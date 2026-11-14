import argparse, json, os, time, math, random, sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.build import get_preset, get_config, build_vl, get_tok, load_vision
from src.data import VLDS, SFTDS, collate_vl, collate_lm
from src.model import RMSNorm

ROOT = os.environ.get('LJ_DATA_ROOT', 'data')

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
p.add_argument('--epochs', type=int, default=1)
p.add_argument('--bs', type=int, default=0)
p.add_argument('--lr_proj', type=float, default=0.0)
p.add_argument('--lr_lm', type=float, default=0.0)
p.add_argument('--max_steps', type=int, default=0)
p.add_argument('--init', default='')
p.add_argument('--out', default='')
p.add_argument('--log', default='')
p.add_argument('--text_ratio', type=float, default=-1.0)
p.add_argument('--vdata', default='')
p.add_argument('--tdata', default='')
args = p.parse_args()

pr = get_preset(args.variant); hp = pr['hp']
bs = args.bs or hp['mm_bs']; lr_proj = args.lr_proj or hp['mm_lr_proj']
lr_lm = args.lr_lm or hp['mm_lr_lm']; max_steps = args.max_steps or hp['mm_max']
text_ratio = hp['text_ratio'] if args.text_ratio < 0 else args.text_ratio
vdata = args.vdata or os.path.join(ROOT, pr['vision_dir'], 'mm_vl_train.jsonl')
tdata = args.tdata or os.path.join(ROOT, pr['sft_dir'], 'sft_v4.jsonl')
init = args.init or f'ckpt/{args.variant}/align_final.pt'
out = args.out or f'ckpt/{args.variant}/mm_final.pt'
log = args.log or f'logs/{args.variant}_mm.jsonl'

cfg = get_config(args.variant); tok = get_tok(args.variant)
m = build_vl(args.variant, cfg, tok.img).cuda()
ck = torch.load(init, map_location='cpu')
m.lm.load_state_dict(ck['model']); m.projector.load_state_dict(ck['projector'])
load_vision(args.variant, m)

for q in m.parameters(): q.requires_grad = False
for q in m.projector.parameters(): q.requires_grad = True
for i in (0, cfg.n_layer - 1):
    for q in m.lm.blocks[i].parameters(): q.requires_grad = True
    for mod in m.lm.blocks[i].modules():
        if isinstance(mod, RMSNorm):
            for q in mod.parameters(): q.requires_grad = True
for q in m.lm.norm_f.parameters(): q.requires_grad = True
print('trainable_M', round(sum(q.numel() for q in m.parameters() if q.requires_grad) / 1e6, 2), flush=True)

vl_ds = VLDS(vdata, tok, prompt='回答问题')
tx_ds = SFTDS(tdata, tok)
vl_dl = DataLoader(vl_ds, batch_size=bs, shuffle=True, num_workers=10, drop_last=True,
                   collate_fn=lambda b: collate_vl(b, tok.pad), persistent_workers=True)
tx_dl = DataLoader(tx_ds, batch_size=bs, shuffle=True, num_workers=2, drop_last=True,
                   collate_fn=lambda b: collate_lm(b, tok.pad), persistent_workers=True)
steps_per_epoch = len(vl_dl); total_steps = min(max_steps, steps_per_epoch * args.epochs)
groups = [{'params': m.projector.parameters(), 'lr': lr_proj},
          {'params': [q for n, q in m.named_parameters() if q.requires_grad and not n.startswith('projector')],
           'lr': lr_lm}]
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.1)


def lr_at(s, base):
    if s < 50: return base * (s + 1) / 50
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 50) / max(1, total_steps - 50))))


os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
logf = open(log, 'a', encoding='utf-8'); step = 0; t0 = time.time(); m.train(); m.vision.eval()
tx_it = iter(tx_dl); rng = random.Random(0)
mix_p = text_ratio if args.variant == 'pro' else text_ratio / (1 - text_ratio)
print(f'[{args.variant}] vl_rows={len(vl_ds)} steps={total_steps} mix_p={round(mix_p, 3)}', flush=True)
for ep in range(args.epochs):
    for x, y, imgs in vl_dl:
        opt.param_groups[0]['lr'] = lr_at(step, lr_proj)
        opt.param_groups[1]['lr'] = lr_at(step, lr_lm)
        x, y, imgs = x.cuda(non_blocking=True), y.cuda(non_blocking=True), imgs.cuda(non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss_v, _ = m(x, imgs, y)
        loss = loss_v
        if rng.random() < mix_p:
            try: tx = next(tx_it)
            except StopIteration: tx_it = iter(tx_dl); tx = next(tx_it)
            txx, tyy = tx; txx, tyy = txx.cuda(non_blocking=True), tyy.cuda(non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss_t, _ = m(txx, None, tyy)
            loss = loss_v + loss_t
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            rec = {'step': step, 'ep': ep, 'loss': round(loss.item(), 4), 'elapsed': round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + '\n'); logf.flush()
        step += 1
        if step >= total_steps: break
    if step >= total_steps: break

torch.save({'model': m.lm.state_dict(), 'projector': m.projector.state_dict(), 'step': step, 'cfg': vars(cfg)}, out)
print('MM DONE', step, round(time.time() - t0, 1), flush=True)
