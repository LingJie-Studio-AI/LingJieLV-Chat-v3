import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.build import get_preset, get_config, build_vl, get_tok, load_vision

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
p.add_argument('--init', default='')
p.add_argument('--out', default='')
args = p.parse_args()

pr = get_preset(args.variant)
init = args.init or f'ckpt/{args.variant}/mm_final.pt'
out = args.out or f'releases/{pr["model_name"]}/mm_pack.pt'

ck = torch.load(init, map_location='cpu')
cfg = get_config(args.variant); tok = get_tok(args.variant)
m = build_vl(args.variant, cfg, tok.img)
load_vision(args.variant, m)
pack = {
    'model': ck['model'],
    'projector': ck['projector'],
    'vision': m.vision.state_dict(),
    'cfg': vars(cfg),
    'step': ck['step'],
    'img_tokens': cfg.n_img_token,
}
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(pack, out)
print('packed', out, os.path.getsize(out), 'B')
