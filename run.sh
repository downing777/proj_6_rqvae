#!/usr/bin/env bash
set -e

mkdir -p logs

# ====== 在这里改参数 ======
SEED=42
DEVICE="cuda:1"
CODEBOOK_SIZE=32
MI_WEIGHT=1.0
MI_ALPHA=3.0
MI_BETA=1.0
MI_TAU=0.2
MI_TOPK=4
MI_REG_LAYERS=3

# 小数转成文件名友好格式: 0.2 -> 0p2
MI_WEIGHT_TAG="${MI_WEIGHT//./p}"
MI_ALPHA_TAG="${MI_ALPHA//./p}"
MI_BETA_TAG="${MI_BETA//./p}"
MI_TAU_TAG="${MI_TAU//./p}"
DEVICE_TAG="${DEVICE//:/-}"

BASELINE_RUN_NAME="baseline_cb${CODEBOOK_SIZE}_allusers_s${SEED}"
BASELINE_OUT_DIR="outputs/exp_${BASELINE_RUN_NAME}"
BASELINE_LOG_FILE="logs/train_user_${BASELINE_RUN_NAME}.log"

RUN_NAME="mi_cb${CODEBOOK_SIZE}_w${MI_WEIGHT_TAG}_a${MI_ALPHA_TAG}_b${MI_BETA_TAG}_tau${MI_TAU_TAG}_k${MI_TOPK}_allusers_s${SEED}"
OUT_DIR="outputs/exp_${RUN_NAME}"
LOG_FILE="logs/train_user_${RUN_NAME}.log"

# baseline（不加 MI loss）
# nohup python train_user.py \
#   --sample-by item \
#   --seed "${SEED}" \
#   --device "${DEVICE}" \
#   --codebook-size "${CODEBOOK_SIZE}" \
#   --wandb-logging \
#   --output-dir "${BASELINE_OUT_DIR}" \
#   > "${BASELINE_LOG_FILE}" 2>&1 &
# echo "baseline started"
# echo "output-dir: ${BASELINE_OUT_DIR}"
# echo "log: ${BASELINE_LOG_FILE}"

# mi-loss
nohup python train_user.py \
  --sample-by item \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --enable-item-mi-loss \
  --codebook-size "${CODEBOOK_SIZE}" \
  --mi-weight "${MI_WEIGHT}" \
  --mi-alpha "${MI_ALPHA}" \
  --mi-beta "${MI_BETA}" \
  --mi-tau "${MI_TAU}" \
  --mi-topk "${MI_TOPK}" \
  --mi-reg-layers "${MI_REG_LAYERS}" \
  --wandb-logging \
  --output-dir "${OUT_DIR}" \
  > "${LOG_FILE}" 2>&1 &
echo "mi-loss started"
echo "output-dir: ${OUT_DIR}"
echo "log: ${LOG_FILE}"
