"""两个变体的静态预设：身份、数据子目录、视觉塔、训练超参。

Pro  = LingJieLV-Chat-v3-Pro（满血，CLIP ViT-B/16，197 token）
Flash= LingJieLV-Chat-v2-Flash（快速复现，TinyCLIP ViT-B/32，49 token）
"""

PRO_SYS = ('你是凌界（LingJieLV-Chat-v3-Pro），由凌界团队训练的中文多模态对话模型。')
FLASH_SYS = ('你是凌界（LingJieLV-Chat-v2-Flash），由凌界团队训练的中文多模态对话模型。')

PRESETS = {
    'pro': {
        'model_name': 'LingJieLV-Chat-v3-Pro',
        'tokenizer': 'v3/tokenizer_v3.json',
        'n_img': 197,
        'sys_prompt': PRO_SYS,
        'tok_vocab': 16384,
        'tok_min_freq': 2,
        'data_dir': 'v3',
        'sft_dir': 'sft',
        'vision_dir': 'vision',
        'vision': 'clip_b16',
        'hp': dict(
            steps=40000, bs=16, lr=4e-4, warmup=300,
            sft_epochs=1, sft_bs=10, sft_lr=1e-5, sft_warmup=500,
            align_bs=32, align_lr=1e-3, align_warmup=50, align_max=1200,
            mm_bs=24, mm_lr_proj=5e-4, mm_lr_lm=1e-5, mm_max=1500, text_ratio=0.23,
        ),
    },
    'flash': {
        'model_name': 'LingJieLV-Chat-v2-Flash',
        'tokenizer': 'flash/tokenizer.json',
        'n_img': 49,
        'sys_prompt': FLASH_SYS,
        'tok_vocab': 8192,
        'tok_min_freq': 3,
        'data_dir': 'flash',
        'sft_dir': 'flash',
        'vision_dir': 'vision',
        'vision': 'tinyclip',
        'hp': dict(
            steps=14800, bs=32, lr=4e-4, warmup=300,
            sft_epochs=2, sft_bs=32, sft_lr=1e-5, sft_warmup=50,
            align_bs=128, align_lr=1e-3, align_warmup=50, align_max=1968,
            mm_bs=32, mm_lr_proj=5e-4, mm_lr_lm=1e-5, mm_max=2438, text_ratio=0.23,
        ),
    },
}


def get_preset(variant):
    if variant not in PRESETS:
        raise ValueError(f'unknown variant {variant!r}; choose pro|flash')
    return PRESETS[variant]
