import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.build import get_preset, get_config, build_vl, get_tok
from src.tool_runtime import Agent


def load_model(variant, pack_path):
    pr = get_preset(variant)
    cfg = get_config(variant); tok = get_tok(variant)
    m = build_vl(variant, cfg, tok.img)
    ck = torch.load(pack_path, map_location='cpu')
    m.lm.load_state_dict(ck['model'])
    m.projector.load_state_dict(ck['projector'])
    if 'vision' in ck:
        m.vision.load_state_dict(ck['vision'])
    m.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok, pr['sys_prompt']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['pro', 'flash'], default='pro')
    p.add_argument('--pack', required=True, help='自包含 mm_pack.pt 路径')
    p.add_argument('--msg', default=None, help='单次提问；留空进入交互模式')
    args = p.parse_args()

    m, tok, sys_prompt = load_model(args.variant, args.pack)
    agent = Agent(m, tok, sys_prompt)

    if args.msg:
        print(agent.run(args.msg))
        return

    print('凌界 CLI（输入 exit 退出）')
    while True:
        try:
            msg = input('你：').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg in ('exit', 'quit'):
            break
        if not msg:
            continue
        print('凌界：' + agent.run(msg))


if __name__ == '__main__':
    main()
