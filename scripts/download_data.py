import argparse, os, subprocess, sys

# 第三方数据集 / 权重，本仓库不附带原始数据，仅通过脚本拉取或由环境变量指到本地副本。
HF = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')

PRETRAIN_URL = f'{HF}/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_t2t.jsonl'
SFT_URL = f'{HF}/datasets/jingyaogong/minimind_dataset/resolve/main/sft_t2t.jsonl'
CLIP_URL = f'{HF}/openai/clip-vit-base-patch16/resolve/main/pytorch_model.bin'

CEVAL_SUBJ = """computer_network operating_system college_programming computer_architecture
college_physics college_chemistry advanced_mathematics high_school_mathematics
high_school_physics high_school_chemistry high_school_biology college_biology
clinical_medicine basic_medicine anatomy veterinary_medicine college_medicine
business_administration college_economics high_school_history middle_school_history
high_school_geography middle_school_geography high_school_politics middle_school_politics
chinese_language_and_literature art_studies law professional_law
education_science professional_psychology sports_science""".split()


def curl(url, dst, retries=6):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(retries):
        r = subprocess.run(['curl', '-fL', '--retry', '3', '--retry-delay', '2', '--max-time', '1800',
                            '-sS', '-o', dst, url])
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f'got {os.path.basename(dst)} {os.path.getsize(dst)} B', flush=True)
            return True
        print(f'retry {os.path.basename(dst)} ({i + 1}/{retries})', flush=True)
    return False


def download_pro(data_root, with_ceval):
    v3 = os.path.join(data_root, 'v3'); models = os.path.join(data_root, 'models')
    os.makedirs(v3, exist_ok=True); os.makedirs(models, exist_ok=True)
    for name, url in [('pretrain_t2t.jsonl', PRETRAIN_URL), ('sft_t2t.jsonl', SFT_URL)]:
        dst = os.path.join(v3, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f'skip {name} (exists)'); continue
        if not curl(url, dst):
            sys.exit(f'download failed: {name}')
    clip = os.path.join(models, 'clip_vit_b16.bin')
    if not (os.path.exists(clip) and os.path.getsize(clip) > 0):
        if not curl(CLIP_URL, clip):
            sys.exit('download failed: CLIP ViT-B/16')
    if with_ceval:
        ce = os.path.join(data_root, 'ceval'); os.makedirs(ce, exist_ok=True)
        for s in CEVAL_SUBJ:
            dst = f'{ce}/{s}_val.parquet'
            if os.path.exists(dst) and os.path.getsize(dst) > 1000:
                continue
            curl(f'{HF}/datasets/ceval/ceval-exam/resolve/main/{s}/val-00000-of-00001.parquet', dst)


def download_flash(data_root):
    fl = os.path.join(data_root, 'flash'); models = os.path.join(data_root, 'models')
    os.makedirs(fl, exist_ok=True); os.makedirs(models, exist_ok=True)
    # 预训练语料复用公开 minimind 预训练文本，词表/编码阶段再按 flash 规模截断。
    dst = os.path.join(fl, 'pretrain_t2t.jsonl')
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        if not curl(PRETRAIN_URL, dst):
            sys.exit('download failed: flash pretrain')
    # 以下两项无内置可复用的稳定下载地址，需通过环境变量指到本地副本：
    tiny = os.environ.get('LJ_TINYCLIP_BIN', os.path.join(models, 'tinyclip.safetensors'))
    alpaca = os.environ.get('LJ_ALPACA_JSON', '')
    print('[flash] TinyCLIP 权重：', tiny,
          '（若不存在，请将 MIT TinyCLIP ViT-B/32 的 safetensors 放到该路径或设置 LJ_TINYCLIP_BIN）', flush=True)
    print('[flash] Alpaca-GPT4-ZH：', alpaca or '未设置 LJ_ALPACA_JSON',
          '（请将公开 alpaca-gpt4-zh 的 json 放到该路径）', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['pro', 'flash', 'all'], default='pro')
    p.add_argument('--data_root', default=os.environ.get('LJ_DATA_ROOT', 'data'))
    p.add_argument('--with_ceval', action='store_true')
    args = p.parse_args()

    if args.variant in ('pro', 'all'):
        download_pro(args.data_root, args.with_ceval)
    if args.variant in ('flash', 'all'):
        download_flash(args.data_root)

    print('''
[图文数据集需自行准备] COCO Captions / Flickr30k / VQAv2 分片地址未内置。
从公开来源下载 parquet 后，用环境变量传给 scripts/prep_vision.py：
  LJ_COCO_GLOB='data/raw/coco/*.parquet'
  LJ_FLICKR_GLOB='data/raw/flickr/*.parquet'
  LJ_VQA_GLOB='data/raw/vqav2/*.parquet'
''', flush=True)


main()
