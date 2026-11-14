# LingJieLV-Chat-v3-Pro（凌界）开源复现版

一个小型中文多模态 MoE 对话模型的**训练与推理代码复现仓库**，内含 **Pro（满血）** 与 **Flash（快速复现）** 两个变体，各自对应一条独立的一键流水线。

> **本仓库不附带任何训练好的模型权重，也不附带第三方训练数据集。** 权重由你按下面的流水线在本机 / 单卡上重新训练得到；数据集与视觉塔初值由脚本在复现时从公开来源拉取，或按环境变量指到本地副本。

- 对外满血型号：**LingJieLV-Chat-v3-Pro**（中文名：凌界）
- 快速复现型号：**LingJieLV-Chat-v2-Flash**（中文名：凌界 Flash）

---

## 1. 两个变体一览

| 维度 | Pro（满血） | Flash（快速复现） |
|---|---|---|
| 型号 | LingJieLV-Chat-v3-Pro | LingJieLV-Chat-v2-Flash |
| 词表 | 16384（BPE，min_freq 2） | 8192（BPE，min_freq 3） |
| LM 层数 / 头数 | 18 层 / 11 头 | 12 层 / 6 头 |
| `d_model` / FFN | 704 / 1408 | 384 / 704 |
| MoE | 4 专家，top-1 路由 | 4 专家，top-1 路由 |
| 视觉塔 | CLIP ViT-B/16（冻结） | TinyCLIP ViT-B/32（冻结） |
| 图像 token | 197（CLS + 196 patch） | 49（196 patch，丢弃 CLS） |
| 投影器 | 768 → 768 → 704 | 512 → 768 → 384 |
| LM 总参 / 激活 | 190.03M / 82.98M | 36.20M / 16.74M |
| 打包总参 | 276.96M（视觉 85.80M + 投影 1.13M） | 76.32M（视觉 39.43M + 投影 0.69M） |
| 预训练 | 40000 步，bs16，lr 4e-4 | 14800 步，bs32，lr 4e-4 |
| 文本 SFT | 1 epoch，bs10，lr 1e-5 | 2 epoch，bs32，lr 1e-5 |
| 视觉对齐 | ≤1200 步，bs32，lr 1e-3，仅投影器 | 1968 步，bs128，lr 1e-3，仅投影器 |
| 多模态 SFT | ≤1500 步，bs24，proj 5e-4 / LM 1e-5 | 2438 步，bs32，proj 5e-4 / LM 1e-5 |

> 上表参数为代码构网实测值；训练步数 / 批次为归档链路的大方向默认值，可在脚本参数中覆盖。

### 归档实测成绩（历史记录，非本仓复现保证值）

| 指标 | Pro（v3f） | Flash（v2 冻结版） |
|---|---|---|
| 综合基准 | 68.0 / 180（V34.0 / T19.0 / W15.0） | — |
| C-Eval（内容打分） | 30.0% | 28.17%（27 科 600 题） |
| VQA | 35.0 | 45.0%（宽松子串匹配） |
| 同图二选一 | — | 74.0%，NLL gap 0.590 |
| 格式合规 | — | 96.0% |
| TTFT | 35.4 ms（峰值显存 1.23 GB） | 30.3 ms 多模态 / 17.9 ms 文本（峰值 0.39 GB） |

复现结果会受采样边界、数据版本与训练轮次影响，与上述记录存在小幅出入属正常现象。

---

## 2. 模型结构说明

### 2.1 语言模型（MoE Transformer）

- 词表嵌入与输出头 **权重共享（tied embedding）**；
- 每个 Block：RMSNorm → 因果自注意力（RoPE 位置编码）→ RMSNorm → MoE 前馈；
- MoE：路由器 softmax 后 top-1 选专家，4 个专家各持 `w1 / w2` 两份权重；训练时按专家负载占比计算辅助负载均衡损失（权重 0.01）；
- 归一化：RMSNorm（eps 1e-5）；位置编码：RoPE（base 10000）；序列长度 768；
- 精度：bf16 训练，AdamW（β 0.9 / 0.95），梯度裁剪 1.0。

### 2.2 视觉塔与投影器

- **Pro**：CLIP ViT-B/16，224×224 输入，patch 16，输出 197 个 token（含 CLS），训练全程冻结；两层 MLP 投影到 LM 维度。
- **Flash**：TinyCLIP ViT-B/32，224×224 输入，patch 32，输出 49 个 patch token（丢弃 CLS），训练全程冻结；两层 MLP 投影到 LM 维度。
- 图文拼接：文本序列中的 `<|image|>` 占位符被替换为视觉特征序列；对齐阶段只训投影器，多模态 SFT 阶段解冻投影器 + LM 首/末层 + 最终 RMSNorm。

### 2.3 特殊 token

`<|pad|>`、`<|end|>`、`<|user|>`、`<|assistant|>`、`<|image|>`、`<|system|>`，由 BPE 词表训练时写入。

---

## 3. 仓库结构

```
LingJieLV-Chat-v3-Pro-oss/
├── LICENSE                       # MIT
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py                  # Pro LM（MoE）+ 旧版 CLIP 基座 / 投影器 / CLIP Layer
│   ├── model_clip.py             # Pro VL 封装：CLIP ViT-B/16（197 token）
│   ├── model_flash.py            # Flash 变体：小 MoE + TinyCLIP ViT-B/32（49 token）
│   ├── presets.py                # 两变体身份、路径、视觉塔、超参预设
│   ├── build.py                  # 按变体选择配置 / LM / VL / 视觉塔 / 词表
│   ├── data.py                   # 词表封装、预训练/SFT/图文数据集、collate、图像预处理
│   ├── tool_runtime.py           # 安全表达式求值、取时工具、Agent 工具循环
│   └── infer.py                  # 命令行推理入口（--variant pro|flash）
└── scripts/
    ├── download_data.py          # 按变体下载公开语料 / 视觉塔 / C-Eval
    ├── prep_tokenizer.py         # BPE 词表训练 + 编码 train/val bin
    ├── prep_sft.py               # Pro 清洗去重 / Flash Alpaca 精简链路
    ├── prep_vision.py            # COCO/Flickr/VQA parquet → 224 图 + 图文 jsonl
    ├── train_pretrain.py         # 阶段一：纯文本预训练（--variant）
    ├── train_sft.py              # 阶段二：文本 SFT（--variant）
    ├── train_align.py            # 阶段三：视觉对齐，仅投影器（--variant）
    ├── train_mm.py               # 阶段四：多模态 SFT（--variant）
    ├── pack.py                   # 打包自包含 mm_pack.pt（--variant）
    ├── run_pipeline.sh           # Pro 满血一键流水线
    └── run_flash.sh              # Flash 快速复现一键流水线
```

---

## 4. 环境准备

```bash
pip install -r requirements.txt
# 需要一张支持 bf16 的 NVIDIA GPU（原始 Pro 在 RTX 4090 24G 完成；Flash 体量更小）
```

依赖：`torch>=2.1`、`numpy`、`tokenizers`、`pyarrow`、`Pillow`。Flash 加载 TinyCLIP 的 `.safetensors` 时需要 `safetensors`（见 `requirements.txt` 注释）。

---

## 5. 一键复现

### 5.1 Pro 满血

```bash
bash scripts/run_pipeline.sh all
```

### 5.2 Flash 快速复现

```bash
export LJ_ALPACA_JSON=/path/to/alpaca-gpt4-zh.json   # 见 7.2
bash scripts/run_flash.sh all
```

### 5.3 单阶段重跑 / 定位问题

两个脚本都接受阶段参数，断点续跑或单独复现某一步：

```bash
# Pro
bash scripts/run_pipeline.sh download
bash scripts/run_pipeline.sh tokenizer
bash scripts/run_pipeline.sh sft_prep
bash scripts/run_pipeline.sh vision
bash scripts/run_pipeline.sh pretrain
bash scripts/run_pipeline.sh sft
bash scripts/run_pipeline.sh align
bash scripts/run_pipeline.sh mm
bash scripts/run_pipeline.sh pack

# Flash（阶段名相同）
bash scripts/run_flash.sh pretrain
```

数据根目录用环境变量 `LJ_DATA_ROOT` 覆盖，默认仓库内 `./data`；Pro 与 Flash 的中间产物分目录存放（`data/v3`、`data/flash`、`data/vision`），checkpoint 分目录 `ckpt/pro`、`ckpt/flash`。

---

## 6. 训练阶段与超参

### 6.1 词表

| 项 | Pro | Flash |
|---|---|---|
| 词表大小 | 16384 | 8192 |
| min_frequency | 2 | 3 |
| 验证集 | 末 8M token 切为 `val.bin` | 同左 |

### 6.2 阶段一 · 纯文本预训练

| 项 | Pro | Flash |
|---|---|---|
| 可训练 | 全 LM | 全 LM |
| 步数 | 40000 | 14800 |
| 批次 | 16 | 32 |
| 序列 | 768 | 768 |
| 学习率 | 4e-4，余弦衰减 | 4e-4，余弦衰减 |
| warmup | 300 | 300 |
| 优化器 | AdamW(β0.9/0.95)，wd 0.1 | 同左 |
| MoE aux 权重 | 0.01 | 0.01 |

每 2000 步在验证集上打一次 VAL loss；每 200 步打印专家负载，用于观察是否专家坍塌。

### 6.3 阶段二 · 文本 SFT

| 项 | Pro | Flash |
|---|---|---|
| 可训练 | 全 LM | 全 LM |
| epoch | 1 | 2 |
| 批次 | 10 | 32 |
| 学习率 | 1e-5，余弦 | 1e-5，余弦 |
| warmup | 500 | 50 |
| 优化器 | AdamW wd 0.1 | 同左 |

### 6.4 阶段三 · 视觉对齐

| 项 | Pro | Flash |
|---|---|---|
| 可训练 | 仅投影器 | 仅投影器 |
| 批次 | 32 | 128 |
| 学习率 | 1e-3，余弦 | 1e-3，余弦 |
| warmup | 50 | 50 |
| 上限步数 | 1200 | 1968 |
| LM / 视觉塔 | 冻结 | 冻结 |

### 6.5 阶段四 · 多模态 SFT

| 项 | Pro | Flash |
|---|---|---|
| 可训练 | 投影器 + LM 首/末层 + norm_f | 同左 |
| 批次 | 24 | 32 |
| 投影器 lr | 5e-4 | 5e-4 |
| LM lr | 1e-5 | 1e-5 |
| 上限步数 | 1500 | 2438 |
| 文本混入比例 | 0.23 | 0.23 |
| 优化器 | AdamW wd 0.1 | 同左 |

> 说明：对外发布的成品在上述主链路之外，还经过若干轮针对性短补丁（身份、常识门控、工具错误重试等）。本开源脚本只复现主链路大方向，未逐一枚举这些补丁轮次。

---

## 7. 数据来源与署名

训练数据均为第三方公开资源，**版权归原作者与原数据集所有**，使用前请自行核对其许可证。本仓库不重新分发上述资产。

### 7.1 脚本内置下载（Pro / Flash 预训练文本）

| 用途 | 来源（HF 镜像） | 字段 |
|---|---|---|
| 文本预训练语料 | `datasets/jingyaogong/minimind_dataset` · `pretrain_t2t.jsonl` | `text` |
| Pro 文本 SFT 对话 | `datasets/jingyaogong/minimind_dataset` · `sft_t2t.jsonl` | `conversations` |
| Pro 视觉塔初值 | `openai/clip-vit-base-patch16` · `pytorch_model.bin` | CLIP 权重遵循其原始许可 |
| 评测集 | `datasets/ceval/ceval-exam` 各学科 `val` parquet | 仅评测，不参与训练 |

### 7.2 需本地放置的第三方资产（无内置稳定下载地址）

| 变体 | 资产 | 环境变量 / 默认路径 | 说明 |
|---|---|---|---|
| Flash | Alpaca-GPT4-ZH 中文指令数据 | `LJ_ALPACA_JSON` | json 列表或 `{"default": [...]}`，字段 `instruction / input / output`；脚本去重并截取 12000 条 |
| Flash | TinyCLIP ViT-B/32 权重 | `LJ_TINYCLIP_BIN`（默认 `data/models/tinyclip.safetensors`） | MIT TinyCLIP 家族；按 `vision_model.*` 前缀映射到视觉塔 |
| 两者 | COCO Captions / Flickr30k / VQAv2 parquet | `LJ_COCO_GLOB` / `LJ_FLICKR_GLOB` / `LJ_VQA_GLOB` | 图文对与 VQA，见 7.3 |

> 上述资产在原始训练环境中是预放置的，本地暂存里没有可复用的稳定下载链接；为避免编造地址，脚本只提供环境变量入口，请从对应公开仓库获取后放到指定路径。

### 7.3 图文数据集整理

`prep_vision.py` 读取 parquet 分片，统一缩放裁剪到 224×224，输出到 `data/vision/img/`，并生成：

- `align_v3.jsonl`：图文对 `{"img": ..., "cap": ...}`，供视觉对齐阶段使用；
- `mm_vl_train.jsonl`：图文对 + VQA `{"img": ..., "q": ..., "a": ...}`，供多模态 SFT 使用。

```bash
export LJ_COCO_GLOB='data/raw/coco/*.parquet'
export LJ_FLICKR_GLOB='data/raw/flickr/*.parquet'
export LJ_VQA_GLOB='data/raw/vqav2/*.parquet'
python3 scripts/prep_vision.py --variant pro
```

### 7.4 SFT 清洗规则（Pro）

- 对话取首条 user 与末条 assistant；工具样本只保留计算 / 时间两类，剔除天气、联网搜索、浏览、爬虫、HTTP/API、文件读写、代码执行、新闻等；
- 长度区间 4–1250 字符；身份改写：`minimind → 凌界`、`OpenAI → 凌界团队`；
- 精确去重：blake2b(16B)；近似去重：CRC64 三字符 gram 异或 + Hamming 距离 ≤3；
- 带推理链（CoT）的样本同时生成「思考开启 / 思考关闭」两个变体。

---

## 8. 推理

打包得到自包含 `mm_pack.pt`（含 `model / projector / vision / cfg / step / img_tokens`）后：

```bash
# Pro
python3 -m src.infer --variant pro   --pack releases/LingJieLV-Chat-v3-Pro/mm_pack.pt --msg "你好，介绍一下你自己"

# Flash
python3 -m src.infer --variant flash --pack releases/LingJieLV-Chat-v2-Flash/mm_pack.pt --msg "你好"
```

不带 `--msg` 进入交互模式。

### 工具协议

模型输出 `【工具调用】{...}`，由 `tool_runtime.py` 中的工具执行：

- `calculator`：AST 白名单安全求值（仅四则、取模、幂、一元正负），拒绝变量名 / 函数调用 / 导入；
- `get_time`：读取真实时钟（默认 `Asia/Shanghai`），返回 `datetime` 与 `weekday`。

工具结果以 `【工具结果】` 回灌给模型；出错时保持调用态让模型重试，达到调用上限后才进入回答。工具由模型主动决策，脚本层不做话题级硬拦截。

---

## 9. 复现边界与诚实记录

**本仓做了什么**

- 两个变体的完整结构定义、数据集 / collate / 图像预处理、四阶段训练循环、对齐与打包脚本、工具运行时与 CLI 推理；
- 忠实沿用归档链路的结构超参与大方向训练超参；参数规模经构网实测与归档卡一致。

**本仓没做什么**

- 不含任何训练权重、词表成品、图片或第三方数据；
- 未复刻成品后续几十轮短补丁（身份加权、常识门控、工具错误重试、VQA 边界修正等）；
- 内部耦合原环境、依赖第三方基线模型的基准打分脚本未纳入；
- Flash 的 Alpaca 与 TinyCLIP、两者的图文 / VQA 分片未内置下载地址，需按 7.2 / 7.3 自行放置。

### 已知局限（成品实测，复现版同样存在或更明显）

- 语言模型体量小，逻辑推理、复杂应用题能力有限；
- 图文描述偏短，VQA 常以单个 / 少量 token 作答，详细描述能力不足；
- 少量常识题仍会被误路由到计算器；应用题「文字题→表达式」偶发生成畸形表达式；
- 时间类问题路由不稳定；工具结果正确时，最终回答句不一定复述结果数字；
- 评测分数受采样边界 token 波动影响。

---

## 10. 许可证

代码以 MIT 许可证发布，见 [LICENSE](LICENSE)。第三方数据集、CLIP / TinyCLIP 视觉权重遵循其各自原始许可；本仓库不重新分发上述第三方资产。
