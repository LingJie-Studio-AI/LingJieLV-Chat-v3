"""按变体（pro / flash）选择配置、LM、VL 封装与视觉塔加载。"""
import os
from .presets import get_preset

DATA_ROOT = os.environ.get('LJ_DATA_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))


def get_tok(variant):
    from .data import Tok
    pr = get_preset(variant)
    return Tok(path=os.path.join(DATA_ROOT, pr['tokenizer']), n_img=pr['n_img'], sys_prompt=pr['sys_prompt'])


def get_config(variant):
    if variant == 'pro':
        from .model_clip import Config
    else:
        from .model_flash import Config
    return Config()


def build_lm(variant, cfg):
    if variant == 'pro':
        from .model import MiniMoE
    else:
        from .model_flash import MiniMoE
    return MiniMoE(cfg)


def build_vl(variant, cfg, img_id):
    if variant == 'pro':
        from .model_clip import MiniMoEVL
    else:
        from .model_flash import MiniMoEVL
    return MiniMoEVL(cfg, img_id)


def load_vision(variant, m):
    """把冻结视觉塔初值装入 m.vision。"""
    if variant == 'pro':
        from .model_clip import load_clip_b16
        return load_clip_b16(m.vision)
    from .model_flash import load_tinyclip
    return load_tinyclip(m.vision)
