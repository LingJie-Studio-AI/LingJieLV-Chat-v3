#!/usr/bin/env bash
# Flash 快速复现流水线（LingJieLV-Chat-v2-Flash，小尺寸 MoE + TinyCLIP 49 token）。
# 用法：bash scripts/run_flash.sh [阶段]
#   阶段：all|download|tokenizer|sft_prep|vision|pretrain|sft|align|mm|pack
# 必备环境变量：
#   LJ_ALPACA_JSON   Alpaca-GPT4-ZH 的本地 json 路径
#   LJ_TINYCLIP_BIN  TinyCLIP ViT-B/32 的 safetensors 路径（脚本内默认 data/models/tinyclip.safetensors）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY_BIN:-python3}"
export LJ_DATA_ROOT="${LJ_DATA_ROOT:-$ROOT/data}"
VARIANT=flash
STAGE="${1:-all}"

run() { echo "==> [$VARIANT] $1"; shift; "$PY" "$@" --variant "$VARIANT"; }

if [[ "$STAGE" == "all" || "$STAGE" == "download" ]]; then
  run download scripts/download_data.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "tokenizer" ]]; then
  run tokenizer scripts/prep_tokenizer.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "sft_prep" ]]; then
  : "${LJ_ALPACA_JSON:?请先设置 LJ_ALPACA_JSON 指向 Alpaca-GPT4-ZH 的 json}"
  run sft_prep scripts/prep_sft.py --format alpaca --src "$LJ_ALPACA_JSON"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "vision" ]]; then
  run vision scripts/prep_vision.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "pretrain" ]]; then
  run pretrain scripts/train_pretrain.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "sft" ]]; then
  run sft scripts/train_sft.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "align" ]]; then
  run align scripts/train_align.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "mm" ]]; then
  run mm scripts/train_mm.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "pack" ]]; then
  run pack scripts/pack.py
fi
echo "PIPELINE_DONE ($VARIANT $STAGE)"
