import argparse, json, os, time, math, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.build import get_preset, get_config, build_lm
from src.data import PretrainDS, collate_pretrain

ROOT = os.environ.get('LJ_DATA_ROOT', 'data')

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
p.add_argument('--steps', type=int, default=0)
p.add_argument('--bs', type=int, default=0)
p.add_argument('--seq', type=int, default=768)
p.add_argument('--lr', type=float, default=0.0)
p.add_argument('--warmup', type=int, default=0)
p.add_argument('--out', default='')
p.add_argument('--log', default='')
p.add_argument('--data', default='')
args = p.parse_args()

pr = get_preset(args.variant); hp = pr['hp']
steps = args.steps or hp['steps']; bs = args.bs or hp['bs']
lr = args.lr or hp['lr']; warmup = args.warmup or hp['warmup']
data = args.data or os.path.join(ROOT, pr['data_dir'])
out = args.out or f'ckpt/{args.variant}/pretrain_final.pt'
log = args.log or f'logs/{args.variant}_pretrain.jsonl'

os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
cfg = get_config(args.variant)
model = build_lm(args.variant, cfg).cuda()
total, act = model.stats()
print(f'[{args.variant}] params total={total/1e6:.2f}M active={act/1e6:.2f}M', flush=True)

ds = PretrainDS(f'{data}/train.bin', args.seq)
dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, drop_last=True,
                collate_fn=collate_pretrain, persistent_workers=True)
val_ds = PretrainDS(f'{data}/val.bin', args.seq)
val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate_pretrain)
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def lr_at(step):
    if step < warmup: return lr * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


logf = open(log, 'a', encoding='utf-8')
step = 0; t0 = time.time(); tok_seen = 0; model.train()
while step < steps:
    for x, y in dl:
        if step >= steps: break
        for g in opt.param_groups: g['lr'] = lr_at(step)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hidden, aux = model(x)
            logits = torch.nn.functional.linear(hidden, model.tok_emb.weight)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]).float(), y.view(-1))
        loss_total = loss + 0.01 * aux
        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tok_seen += x.numel()
        if step % 20 == 0:
            rec = {'step': step, 'loss': round(loss.item(), 4), 'aux': round(float(aux), 4),
                   'lr': round(lr_at(step), 6), 'gn': round(float(gn), 3), 'tok': tok_seen,
                   'elapsed': round(time.time() - t0, 1)}
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + '\n'); logf.flush()
        if step % 200 == 0:
            fracs = [b.moe.last_frac.tolist() for b in model.blocks if b.moe.last_frac is not None]
            if fracs:
                avg = np.mean(np.array(fracs), 0).round(3).tolist()
                print('EXPERT_LOAD', avg, flush=True)
        if step and step % 2000 == 0:
            model.eval(); vl = 0.0; nb = 0
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for vx, vy in val_dl:
                    vx, vy = vx.cuda(), vy.cuda()
                    h, _ = model(vx)
                    l = torch.nn.functional.cross_entropy(
                        torch.nn.functional.linear(h, model.tok_emb.weight).view(-1, cfg.vocab_size).float(), vy.view(-1))
                    vl += l.item(); nb += 1
                    if nb >= 60: break
            print('VAL', step, round(vl / max(1, nb), 4), flush=True)
            logf.write(json.dumps({'step': step, 'val_loss': round(vl / max(1, nb), 4)}) + '\n'); logf.flush()
            model.train()
        step += 1

torch.save({'model': model.state_dict(), 'step': step, 'cfg': vars(cfg)}, out)
print('DONE steps', step, 'tokens', tok_seen, 'time', round(time.time() - t0, 1), flush=True)
