import os, torch, torch.nn as nn, torch.nn.functional as F
from .model import Config as BaseConfig, MiniMoE, CLIPVision, Projector, CLIPLayer

CLIP_PATH = os.environ.get('LJ_CLIP_PATH',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'models', 'clip_vit_b16.bin'))

class Config(BaseConfig):
    v_hidden = 768
    v_layers = 12
    v_heads = 12
    v_inter = 3072
    v_patch = 16
    v_image = 224
    n_img_token = 197

class CLIPVisionB16(CLIPVision):
    """CLIP ViT-B/16：输出 197 个 token（CLS + 196 patch）"""
    def __init__(self, h=768, layers=12, n_head=12, inter=3072, patch=16, image=224, eps=1e-5):
        super().__init__(h, layers, n_head, inter, patch, image, eps)
    def forward(self, pixel_values):
        x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        cls = self.class_embedding.view(1, 1, -1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], 1) + self.position_embedding
        x = self.pre_layrnorm(x)
        for l in self.layers:
            x = l(x)
        return self.post_layernorm(x)

class MiniMoEVC(MiniMoE):
    pass

class MiniMoEVL(nn.Module):
    def __init__(self, cfg, img_id):
        super().__init__()
        self.cfg = cfg
        self.img_id = img_id
        self.lm = MiniMoE(cfg)
        self.vision = CLIPVisionB16(cfg.v_hidden, cfg.v_layers, cfg.v_heads, cfg.v_inter, cfg.v_patch, cfg.v_image)
        self.projector = Projector(cfg.v_hidden, 768, cfg.d_model)
        for p in self.vision.parameters():
            p.requires_grad = False
    def forward(self, input_ids, images=None, labels=None):
        emb = self.lm.tok_emb(input_ids)
        if images is not None:
            feats = self.projector(self.vision(images).to(emb.dtype))
            mask = input_ids == self.img_id
            n_img = mask.sum().item() // feats.shape[1]
            if feats.shape[0] != n_img:
                raise RuntimeError(f'n_img mismatch: placeholders={mask.sum().item()} images={feats.shape[0]} tokens/img={feats.shape[1]}')
            emb = emb.clone()
            emb[mask] = feats.reshape(-1, feats.shape[-1]).to(emb.dtype)
        hidden, aux = self.lm.forward_emb(emb)
        logits = F.linear(hidden, self.lm.tok_emb.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss, aux

def load_clip_b16(vision, path=CLIP_PATH):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'vision_model' not in sd and 'state_dict' in sd:
        sd = sd['state_dict']
    out = {}
    for k, v in sd.items():
        if not k.startswith('vision_model.'):
            continue
        k2 = k[len('vision_model.'):]
        if k2.startswith('embeddings.'):
            k2 = 'position_embedding' if k2.startswith('embeddings.position_embedding') else k2[len('embeddings.'):]
        elif k2.startswith('encoder.'):
            k2 = k2[len('encoder.'):]
        out[k2] = v
    pe = out.get('position_embedding')
    if pe is not None and pe.dim() == 2:
        out['position_embedding'] = pe.unsqueeze(0)
    missing, unexpected = vision.load_state_dict(out, strict=False)
    bad = [m for m in missing if 'position_embedding' not in m]
    print('clip load missing', bad[:6], 'unexpected', unexpected[:6], flush=True)
    return len(out)
