#!/usr/bin/env bash
# 两阶段 (默认只训 SID 可学习前缀, 冻结基座):
#   1) 构造 SFT 语料 -> sft/
#   2) SFT -> sid_sft.pt, DPO -> sid_dpo.pt
# 在 proj_6_rqvae 下: bash softprompt/run_train.sh
# 训练长度仅由 MAX_STEPS_SFT / MAX_STEPS_DPO 控制(对应 --max-steps)

set -euo pipefail
ROOT="/nfs5/yhy/tn/proj_6_rqvae"
cd "${ROOT}"

QWEN_BASE="${QWEN_BASE:-/nfs5/yhy/model_hub/Qwen3-8B}"
DPO_JSONL="${DPO_JSONL:-/nfs5/yhy/tn/proj_6_rqvae/softprompt/data/dpo_electronics_generated.jsonl}"
ITEM_JSONL="${ITEM_JSONL:-/nfs5/yhy/tn/proj_6_rqvae/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT}/softprompt/outputs}"
SFT_JSONL="${OUT_DIR}/sft_from_item_title.jsonl"
SFT_DIR="${OUT_DIR}/sft"
DPO_DIR="${OUT_DIR}/dpo"

MAX_STEPS_SFT="${MAX_STEPS_SFT:-500}"
MAX_STEPS_DPO="${MAX_STEPS_DPO:-1000}"
BATCH_SFT="${BATCH_SFT:-2}"
BATCH_DPO="${BATCH_DPO:-1}"
MAX_LEN="${MAX_LEN:-30000}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-0}"
LR_SFT="${LR_SFT:-3e-5}"
LR_DPO="${LR_DPO:-3e-5}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "QWEN_BASE=${QWEN_BASE}  OUT_DIR=${OUT_DIR}  MAX_STEPS_SFT=${MAX_STEPS_SFT}  MAX_STEPS_DPO=${MAX_STEPS_DPO}"
mkdir -p "${OUT_DIR}"

# ---- 1) SFT 语料
python3 softprompt/data/build_sft_from_dpo.py \
  --dpo-jsonl "${DPO_JSONL}" \
  --item-jsonl "${ITEM_JSONL}" \
  --out "${SFT_JSONL}"

# ---- 2) SFT
python3 softprompt/train/train_sft.py \
  --train-jsonl "${SFT_JSONL}" \
  --base-model "${QWEN_BASE}" \
  --output-dir "${SFT_DIR}" \
  --train-batch-size "${BATCH_SFT}" \
  --max-steps "${MAX_STEPS_SFT}" \
  --learning-rate "${LR_SFT}" \
  --max-length "${MAX_LEN}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}"

# ---- 3) DPO
python3 softprompt/train/train_dpo.py \
  --train-jsonl "${DPO_JSONL}" \
  --base-model "${QWEN_BASE}" \
  --sft-ckpt "${SFT_DIR}/sid_sft.pt" \
  --output-dir "${DPO_DIR}" \
  --train-batch-size "${BATCH_DPO}" \
  --max-steps "${MAX_STEPS_DPO}" \
  --learning-rate "${LR_DPO}" \
  --max-length "${MAX_LEN}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}"

echo "Done. SFT: ${SFT_DIR}/sid_sft.pt  DPO: ${DPO_DIR}/sid_dpo.pt"
