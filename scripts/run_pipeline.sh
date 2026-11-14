#!/usr/bin/env bash
# Pro 满血复现流水线：下载 → 清洗 → 预训练 → 文本SFT → 视觉对齐 → 多模态SFT → 打包。
# 用法：bash scripts/run_pipeline.sh [阶段]
#   阶段：all|download|tokenizer|sft_prep|vision|pretrain|sft|align|mm|pack
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY_BIN:-python3}"
export LJ_DATA_ROOT="${LJ_DATA_ROOT:-$ROOT/data}"
VARIANT=pro
STAGE="${1:-all}"

run() { echo "==> [$VARIANT] $1"; shift; "$PY" "$@" --variant "$VARIANT"; }

if [[ "$STAGE" == "all" || "$STAGE" == "download" ]]; then
  run download scripts/download_data.py --with-ceval
fi
if [[ "$STAGE" == "all" || "$STAGE" == "tokenizer" ]]; then
  run tokenizer scripts/prep_tokenizer.py
fi
if [[ "$STAGE" == "all" || "$STAGE" == "sft_prep" ]]; then
  run sft_prep scripts/prep_sft.py
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
