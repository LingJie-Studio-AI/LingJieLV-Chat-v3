import json, os, io
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from tokenizers import Tokenizer

# 数据根目录：可用环境变量 LJ_DATA_ROOT 覆盖，默认仓库内 ./data
DATA_ROOT = os.environ.get('LJ_DATA_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
IMG_DIR = os.path.join(DATA_ROOT, 'vision', 'img')

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


DEFAULT_SYS = '你是凌界（LingJieLV-Chat-v3-Pro），由凌界团队训练的中文多模态对话模型。'


class Tok:
    def __init__(self, path=None, n_img=197, sys_prompt=None):
        path = path or os.path.join(DATA_ROOT, 'v3', 'tokenizer_v3.json')
        self.t = Tokenizer.from_file(path)
        self.pad = self.t.token_to_id('<|pad|>')
        self.end = self.t.token_to_id('<|end|>')
        self.user = self.t.token_to_id('<|user|>')
        self.asst = self.t.token_to_id('<|assistant|>')
        self.img = self.t.token_to_id('<|image|>')
        self.n_img = n_img
        self.sys = self.t.token_to_id('<|system|>')
        self.sys_prompt = sys_prompt or DEFAULT_SYS
        self.vocab = self.t.get_vocab_size()

    def enc(self, s):
        return self.t.encode(s, add_special_tokens=False).ids


def load_image(path, size=224):
    im = Image.open(os.path.join(IMG_DIR, path)).convert('RGB')
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


class PretrainDS(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq = seq_len
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq
        x = self.data[s:s + self.seq].astype(np.int64)
        y = self.data[s + 1:s + 1 + self.seq].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def collate_pretrain(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return x, y


class SFTDS(Dataset):
    def __init__(self, path, tok, max_seq=768, max_n=None):
        self.rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('caps'):
                    for c in r['caps']:
                        if c: self.rows.append({'img': r['img'], 'cap': c})
                else:
                    self.rows.append(r)
                if max_n and len(self.rows) >= max_n: break
        self.tok = tok
        self.max_seq = max_seq

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if r.get('prompt') is not None:
            u = r['prompt']; a = r.get('response') or ''
        elif r.get('u') is not None:
            u = r['u']; a = r.get('a') or ''
        else:
            ins = (r.get('instruction') or '').strip()
            inp = (r.get('input') or '').strip()
            u = ins + ('\n' + inp if inp else '')
            a = (r.get('output') or '').strip()
        sysp = (r.get('system') or getattr(self.tok, 'sys_prompt', DEFAULT_SYS))
        sids = [self.tok.sys] + self.tok.enc(sysp) + [self.tok.end]
        u = u[: max(0, self.max_seq - len(sids) - 16)]
        ids = sids + [self.tok.user] + self.tok.enc(u) + [self.tok.end, self.tok.asst]
        n_prompt = len(ids)
        ans = self.tok.enc(a)[: max(0, self.max_seq - n_prompt - 1)] + [self.tok.end]
        ids = ids + ans
        labels = [-100] * n_prompt + ans
        ids = ids[:self.max_seq]; labels = labels[:self.max_seq]
        return torch.tensor(ids), torch.tensor(labels)


def collate_lm(batch, pad_id=0):
    L = max(len(b[0]) for b in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(batch):
        x[i, :len(ids)] = ids
        y[i, :len(lab)] = lab
    return x, y


class VLDS(Dataset):
    """通用图文数据集：每行 {"img":..., "cap"/"q":"...", "a":...}"""
    def __init__(self, path, tok, max_seq=768, prompt='描述这张图片', max_n=None):
        self.rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('caps'):
                    for c in r['caps']:
                        if c: self.rows.append({'img': r['img'], 'cap': c})
                else:
                    self.rows.append(r)
                if max_n and len(self.rows) >= max_n: break
        self.tok = tok
        self.max_seq = max_seq
        self.prompt = prompt

    def __len__(self):
        return len(self.rows)

    def build(self, r):
        t = self.tok
        if 'a' in r and r.get('a'):
            p_txt = r.get('q') or self.prompt
            ans = r['a']
        else:
            p_txt = self.prompt
            ans = r['cap']
        head = [t.img] * getattr(t, 'n_img', 49) + [t.user] + t.enc(p_txt)[: 200] + [t.end, t.asst]
        n_prompt = len(head)
        ans_ids = t.enc(ans)[: max(0, self.max_seq - n_prompt - 1)] + [t.end]
        ids = head + ans_ids
        labels = [-100] * n_prompt + ans_ids
        ids = ids[:self.max_seq]; labels = labels[:self.max_seq]
        return ids, labels, r['img']

    def __getitem__(self, i):
        ids, labels, img = self.build(self.rows[i])
        return torch.tensor(ids), torch.tensor(labels), img


def collate_vl(batch, pad_id=0, size=224):
    L = max(len(b[0]) for b in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    imgs = []
    for i, (ids, lab, img) in enumerate(batch):
        x[i, :len(ids)] = ids
        y[i, :len(lab)] = lab
        imgs.append(load_image(img, size))
    return x, y, torch.stack(imgs)
