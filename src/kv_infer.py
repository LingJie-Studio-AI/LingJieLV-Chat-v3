# -*- coding: utf-8 -*-
"""KV 缓存推理：prefill 一次建缓存，decode 每步只算一个新 token。

- 不修改任何训练代码与权重，直接复用 blocks 内的 norm / attn / moe 子模块；
- Pro 与 Flash 的 MiniMoE 结构同构，本模块两变体通用；
- use_kv=False 时退化为「每步重算整段序列」，用于对照验证两条路径输出一致。
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .build import get_preset, get_config, build_vl, get_tok
from .data import CLIP_MEAN, CLIP_STD


def detect_variant(ck):
    cfg = ck.get('cfg')
    if isinstance(cfg, dict):
        n_img = cfg.get('n_img_token')
        vocab = cfg.get('vocab_size')
    else:
        n_img = getattr(cfg, 'n_img_token', None)
        vocab = getattr(cfg, 'vocab_size', None)
    if n_img == 197:
        return 'pro'
    if n_img == 49:
        return 'flash'
    if vocab == 8192 or ck.get('vocab') == 8192:
        return 'flash'
    # 兜底：按 LM 词表嵌入形状判断
    lm_sd = ck.get('model', ck.get('lm'))
    if lm_sd is not None and lm_sd['tok_emb.weight'].shape[0] == 8192:
        return 'flash'
    return 'pro'


def load_pack(pack_path, tokenizer_path=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(pack_path, map_location='cpu', weights_only=False)
    variant = detect_variant(ck)
    cfg = get_config(variant)
    if tokenizer_path:
        tok = get_tok_with(variant, tokenizer_path)
    else:
        tok = get_tok(variant)
    m = build_vl(variant, cfg, tok.img)
    lm_sd = ck.get('model', ck.get('lm'))
    m.lm.load_state_dict(lm_sd)
    m.projector.load_state_dict(ck['projector'])
    if 'vision' in ck:
        m.vision.load_state_dict(ck['vision'])
    m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    sys_prompt = get_preset(variant)['sys_prompt']
    return m, tok, sys_prompt, variant


def get_tok_with(variant, tokenizer_path):
    from .data import Tok
    pr = get_preset(variant)
    return Tok(path=tokenizer_path, n_img=pr['n_img'], sys_prompt=pr['sys_prompt'])


def pil_to_tensor(img, size=224):
    im = img.convert('RGB')
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[: x.shape[-2]].view(1, 1, -1, x1.shape[-1])
    s = sin[: x.shape[-2]].view(1, 1, -1, x1.shape[-1])
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class KVChat:
    def __init__(self, model, tok, sys_prompt, device=None):
        self.m = model
        self.lm = model.lm
        self.t = tok
        self.sys_prompt = sys_prompt
        self.device = device or next(model.parameters()).device
        self.history = []  # [(user_text, assistant_text)]

    def reset(self):
        self.history = []

    def _build_emb(self, text, image=None):
        t = self.t
        head = [t.sys] + t.enc(self.sys_prompt) + [t.end]
        if image is not None:
            head += [t.img] * t.n_img
        head += [t.user] + t.enc(text) + [t.end, t.asst]
        ids = torch.tensor(head, dtype=torch.long, device=self.device)
        emb = self.lm.tok_emb(ids).unsqueeze(0)
        if image is not None:
            x = pil_to_tensor(image, self.m.cfg.v_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.m.projector(self.m.vision(x).to(emb.dtype))  # 1,n_img,D
            mask = (ids == t.img).nonzero(as_tuple=True)[0]
            emb[0, mask] = feats.reshape(-1, feats.shape[-1]).to(emb.dtype)
        return emb

    def _build_full(self, new_text, image=None):
        """组装含历史的完整 emb；历史只含文本。"""
        t = self.t
        parts = [t.sys] + t.enc(self.sys_prompt) + [t.end]
        img_positions = []
        if image is not None:
            img_positions = list(range(len(parts), len(parts) + t.n_img))
            parts += [t.img] * t.n_img
        for u, a in self.history:
            parts += [t.user] + t.enc(u) + [t.end, t.asst] + t.enc(a) + [t.end]
        parts += [t.user] + t.enc(new_text) + [t.end, t.asst]
        ids = torch.tensor(parts, dtype=torch.long, device=self.device)
        emb = self.lm.tok_emb(ids).unsqueeze(0)
        if image is not None:
            x = pil_to_tensor(image, self.m.cfg.v_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats = self.m.projector(self.m.vision(x).to(emb.dtype))
            mask = (ids == t.img).nonzero(as_tuple=True)[0]
            emb[0, mask] = feats.reshape(-1, feats.shape[-1]).to(emb.dtype)
        return emb

    def _blocks(self, x, cache, pos):
        """x: (1,T,D)。cache: None=无缓存整段因果；list=按层拼接 past k/v。"""
        new_cache = []
        for i, blk in enumerate(self.lm.blocks):
            h = blk.norm1(x)
            qkv = blk.attn.qkv(h).view(1, -1, 3, blk.attn.n_head, blk.attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if cache is None:
                q = _rope(q, self.lm.cos, self.lm.sin)
                k = _rope(k, self.lm.cos, self.lm.sin)
                att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                pk, pv = cache[i]
                q = _rope(q, self.lm.cos[pos:], self.lm.sin[pos:])
                k = _rope(k, self.lm.cos[pos:], self.lm.sin[pos:])
                k = torch.cat([pk, k], dim=-2)
                v = torch.cat([pv, v], dim=-2)
                att = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            new_cache.append((k, v))
            x = x + blk.attn.o(att.transpose(1, 2).reshape(1, -1, blk.attn.n_head * blk.attn.head_dim))
            h2, _ = blk.moe(blk.norm2(x))
            x = x + h2
        return self.lm.norm_f(x), new_cache

    def _logits(self, emb):
        h, _ = self._blocks(emb, None, 0)
        return F.linear(h, self.lm.tok_emb.weight)

    @torch.no_grad()
    def generate(self, text, image=None, max_new=200, use_kv=True):
        emb = self._build_full(text, image)
        out_ids = []
        if use_kv:
            h, cache = self._blocks(emb, None, 0)
            pos = emb.shape[1]
            nid = int(torch.argmax(F.linear(h[:, -1], self.lm.tok_emb.weight), dim=-1))
            for _ in range(max_new):
                if nid == self.t.end or pos >= self.lm.cfg.max_seq:
                    break
                out_ids.append(nid)
                step_emb = self.lm.tok_emb(torch.tensor([[nid]], device=self.device))
                h, cache = self._blocks(step_emb, cache, pos)
                pos += 1
                nid = int(torch.argmax(F.linear(h[:, -1], self.lm.tok_emb.weight), dim=-1))
        else:
            cur = emb
            for _ in range(max_new):
                lg = self._logits(cur)[:, -1]
                nid = int(torch.argmax(lg, dim=-1))
                if nid == self.t.end or cur.shape[1] >= self.lm.cfg.max_seq:
                    break
                out_ids.append(nid)
                nxt = self.lm.tok_emb(torch.tensor([[nid]], device=self.device))
                cur = torch.cat([cur, nxt], dim=1)
        answer = self.t.t.decode(out_ids)
        self.history.append((text, answer))
        return answer, out_ids
