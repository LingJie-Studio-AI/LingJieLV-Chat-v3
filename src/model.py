import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 16384
    d_model = 704
    n_layer = 18
    n_head = 11
    inter = 1408
    n_expert = 4
    top_k = 1
    max_seq = 768
    eps = 1e-5
    v_hidden = 512
    v_layers = 12
    v_heads = 8
    v_inter = 2048
    v_patch = 32
    v_image = 224
    n_img_token = 49

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        v = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(v + self.eps)
        return (xf * self.weight.float()).to(dt)

def rope_cache(seq_len, head_dim, base=10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    s = sin[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, T, D))

class MoEFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_expert = cfg.n_expert
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_expert, bias=False)
        self.w1 = nn.Parameter(torch.empty(cfg.n_expert, cfg.d_model, cfg.inter))
        self.w2 = nn.Parameter(torch.empty(cfg.n_expert, cfg.inter, cfg.d_model))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        self.last_frac = None
    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        probs = F.softmax(self.router(xf.float()), dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        if self.training:
            frac = torch.zeros(self.n_expert, device=xf.device, dtype=torch.float32)
            frac.scatter_add_(0, top_i.view(-1), torch.ones_like(top_w.view(-1), dtype=torch.float32))
            frac = frac / float(xf.shape[0])
            aux = self.n_expert * (frac * probs.mean(0)).sum()
            self.last_frac = frac.detach()
        else:
            aux = torch.zeros((), device=xf.device)
        out = torch.zeros(xf.shape, device=xf.device, dtype=torch.float32)
        for e in range(self.n_expert):
            idx = (top_i == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            h = xf.index_select(0, idx)
            h = F.gelu(h @ self.w1[e]) @ self.w2[e]
            w = top_w.view(-1)[idx].to(h.dtype).unsqueeze(-1)
            out.index_add_(0, idx, (h * w).float())
        return out.to(xf.dtype).view(B, T, D), aux

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.eps)
        self.moe = MoEFFN(cfg)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        h, aux = self.moe(self.norm2(x))
        return x + h, aux

class MiniMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.eps)
        cos, sin = rope_cache(cfg.max_seq, cfg.d_model // cfg.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def stats(self):
        total = sum(p.numel() for p in self.parameters())
        act = self.tok_emb.weight.numel()
        for n, p in self.named_parameters():
            if n.startswith("tok_emb"):
                continue
            if ".moe.w1" in n or ".moe.w2" in n:
                act += p.numel() // self.cfg.n_expert
            else:
                act += p.numel()
        return total, act
    def forward_emb(self, emb):
        x = emb
        aux = 0.0
        for blk in self.blocks:
            x, a = blk(x, self.cos, self.sin)
            aux = aux + a
        return self.norm_f(x), aux / len(self.blocks)
    def forward(self, tokens):
        return self.forward_emb(self.tok_emb(tokens))

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

class CLIPAttention(nn.Module):
    def __init__(self, h, n_head):
        super().__init__()
        self.n_head = n_head
        self.hd = h // n_head
        self.q_proj = nn.Linear(h, h)
        self.k_proj = nn.Linear(h, h)
        self.v_proj = nn.Linear(h, h)
        self.out_proj = nn.Linear(h, h)
    def forward(self, x):
        B, T, H = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, H))

class CLIPMLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.fc1 = nn.Linear(h, inter)
        self.fc2 = nn.Linear(inter, h)
    def forward(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))

class CLIPLayer(nn.Module):
    def __init__(self, h, n_head, inter, eps=1e-5):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(h, eps)
        self.self_attn = CLIPAttention(h, n_head)
        self.layer_norm2 = nn.LayerNorm(h, eps)
        self.mlp = CLIPMLP(h, inter)
    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))

class CLIPVision(nn.Module):
    def __init__(self, h=512, layers=12, n_head=8, inter=2048, patch=32, image=224, eps=1e-5):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(h))
        self.patch_embedding = nn.Conv2d(3, h, patch, patch, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, (image // patch) ** 2 + 1, h))
        self.pre_layrnorm = nn.LayerNorm(h, eps)
        self.layers = nn.ModuleList([CLIPLayer(h, n_head, inter, eps) for _ in range(layers)])
        self.post_layernorm = nn.LayerNorm(h, eps)
    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], 1) + self.position_embedding
        x = self.pre_layrnorm(x)
        for l in self.layers:
            x = l(x)
        return self.post_layernorm(x)[:, 1:]

class Projector(nn.Module):
    def __init__(self, d_in=512, d_mid=768, d_out=512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_mid)
        self.fc2 = nn.Linear(d_mid, d_out)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class MiniMoEVL(nn.Module):
    def __init__(self, cfg, img_id):
        super().__init__()
        self.cfg = cfg
        self.img_id = img_id
        self.lm = MiniMoE(cfg)
        self.vision = CLIPVision(cfg.v_hidden, cfg.v_layers, cfg.v_heads, cfg.v_inter, cfg.v_patch, cfg.v_image)
        self.projector = Projector(cfg.v_hidden, 768, cfg.d_model)
        for p in self.vision.parameters():
            p.requires_grad = False
    def forward(self, input_ids, images=None, labels=None):
        emb = self.lm.tok_emb(input_ids)
        if images is not None:
            feats = self.projector(self.vision(images).to(emb.dtype))
            mask = input_ids == self.img_id
            emb = emb.clone()
            emb[mask] = feats.reshape(-1, feats.shape[-1]).to(emb.dtype)
        hidden, aux = self.lm.forward_emb(emb)
        logits = F.linear(hidden, self.lm.tok_emb.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss, aux
