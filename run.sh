#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yuanhanyang.yhy/proj_6_rqvae"
cd "${ROOT}"

# 如果你希望脚本内部自动激活环境，可以取消下面两行注释
# eval "$(conda shell.bash hook)"
# conda activate rqvae

# ====== 在这里改参数 ======
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-32}"
EMBED_DIM="${EMBED_DIM:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
TRAIN_ITERS="${TRAIN_ITERS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
COMMITMENT_WEIGHT=0.1
HIDDEN_DIMS="512 256"

MI_WEIGHT=0.1
MI_ALPHA=1.61
MI_BETA=1.3
MI_TAU=0.5
MI_TOPK=32
MI_REG_LAYERS=3
MI_WARMUP_STEPS=1000

# ====== 输出路径 ======
OUTPUT_ROOT="/home/yuanhanyang.yhy/project_6_outputs/sid"
LOG_ROOT="/home/yuanhanyang.yhy/project_6_outputs/logs/sid"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

# 小数转成文件名友好格式: 0.2 -> 0p2
MI_WEIGHT_TAG="${MI_WEIGHT//./p}"
MI_ALPHA_TAG="${MI_ALPHA//./p}"
MI_BETA_TAG="${MI_BETA//./p}"
MI_TAU_TAG="${MI_TAU//./p}"

# ====== baseline run name/path ======
BASELINE_RUN_NAME="baseline_cb${CODEBOOK_SIZE}_ed${EMBED_DIM}_s${SEED}"
BASELINE_OUT_DIR="${OUTPUT_ROOT}/exp_${BASELINE_RUN_NAME}"
BASELINE_LOG_FILE="${LOG_ROOT}/train_user_${BASELINE_RUN_NAME}.log"

# ====== MI-loss run name/path ======
RUN_NAME="mi_cb${CODEBOOK_SIZE}_ed${EMBED_DIM}_w${MI_WEIGHT_TAG}_a${MI_ALPHA_TAG}_b${MI_BETA_TAG}_tau${MI_TAU_TAG}_k${MI_TOPK}_s${SEED}"
OUT_DIR="${OUTPUT_ROOT}/exp_${RUN_NAME}"
LOG_FILE="${LOG_ROOT}/train_user_${RUN_NAME}.log"

mkdir -p "${OUT_DIR}"

# ============================================================
# Auto-nohup wrapper
# 第一次执行 bash run.sh 时，只负责把脚本挂到后台，然后立刻退出。
# 真正训练发生在 nohup 重新启动后的第二次执行中。
# 因此终端不会显示训练进度，关闭终端也不影响训练。
# ============================================================
if [[ -z "${_RQVAE_NOHUP_WRAPPER:-}" ]]; then
  export _RQVAE_NOHUP_WRAPPER=1

  nohup bash "$0" "$@" >/dev/null 2>&1 < /dev/null &

  echo "============================================="
  echo "  MI-loss training launched in background"
  echo "============================================="
  echo "  Output dir: ${OUT_DIR}"
  echo "  Log: ${LOG_FILE}"
  echo "============================================="

  exit 0
fi

# ============================================================
# 关键：从这里开始已经是后台子进程
# 强制把后续所有输出写入 log，避免继续打印到终端
# ============================================================
exec </dev/null
exec >> "${LOG_FILE}" 2>&1

echo "============================================="
echo "  MI-loss Training Pipeline Started"
echo "============================================="
echo "  Time: $(date)"
echo "  Working dir: ${ROOT}"
echo "  Device: ${DEVICE}"
echo "  Output dir: ${OUT_DIR}"
echo "  Log: ${LOG_FILE}"
echo "============================================="
echo ""

# ============================================================
# 可选：baseline 训练，不加 MI loss
# 如果需要跑 baseline，可以手动取消下面这一整段注释。
# 注意：外层已经由 Auto-nohup 接管，所以 baseline 输出也会写入 log。
# ============================================================
#
# mkdir -p "${BASELINE_OUT_DIR}"
#
# echo "============================================="
# echo "  Baseline Training Started"
# echo "============================================="
# echo "  Time: $(date)"
# echo "  Output dir: ${BASELINE_OUT_DIR}"
# echo "============================================="
#
# python -u train_user.py \
#   --sample-by item \
#   --seed "${SEED}" \
#   --device "${DEVICE}" \
#   --codebook-size "${CODEBOOK_SIZE}" \
#   --embed-dim "${EMBED_DIM}" \
#   --learning-rate "${LEARNING_RATE}" \
#   --train-iterations "${TRAIN_ITERS}" \
#   --train-batch-size "${BATCH_SIZE}" \
#   --commitment-weight "${COMMITMENT_WEIGHT}" \
#   --hidden-dims ${HIDDEN_DIMS} \
#   --output-dir "${BASELINE_OUT_DIR}"
#
# echo ""
# echo "====== Baseline training finished at $(date), analyzing SID distribution ======"
#
# python -u analyze_sid_distribution.py \
#   --input "${BASELINE_OUT_DIR}/user_semantic_ids.jsonl"
#
# echo ""
# echo "====== Baseline analyze finished at $(date) ======"
#
# exit 0

# ============================================================
# MI-loss: 训练 + 自动分析
# 训练结束后自动运行 analyze_sid_distribution.py
# 所有输出都写入 ${LOG_FILE}
# ============================================================

python -u train_user.py \
  --sample-by item \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --codebook-size "${CODEBOOK_SIZE}" \
  --embed-dim "${EMBED_DIM}" \
  --learning-rate "${LEARNING_RATE}" \
  --train-iterations "${TRAIN_ITERS}" \
  --train-batch-size "${BATCH_SIZE}" \
  --commitment-weight "${COMMITMENT_WEIGHT}" \
  --hidden-dims ${HIDDEN_DIMS} \
  --enable-item-mi-loss \
  --mi-weight "${MI_WEIGHT}" \
  --mi-alpha "${MI_ALPHA}" \
  --mi-beta "${MI_BETA}" \
  --mi-tau "${MI_TAU}" \
  --mi-topk "${MI_TOPK}" \
  --mi-reg-layers "${MI_REG_LAYERS}" \
  --mi-warmup-steps "${MI_WARMUP_STEPS}" \
  --output-dir "${OUT_DIR}"

echo ""
echo "====== Training finished at $(date), analyzing SID distribution ======"

python -u analyze_sid_distribution.py \
  --input "${OUT_DIR}/user_semantic_ids.jsonl"

echo ""
echo "====== Evaluating item-internal SID distribution ======"

EVAL_DIR="${OUTPUT_ROOT}/eval"
ITEM_USER_MAP="/home/yuanhanyang.yhy/model_hub/amazon_user/raw/item_to_user_ids.json"
mkdir -p "${EVAL_DIR}"

python -u eval_item_sid_distribution.py \
  --item-user-map-path "${ITEM_USER_MAP}" \
  --exp "${RUN_NAME}=${OUT_DIR}" \
  --sid-mode full \
  --summary-csv-path "${EVAL_DIR}/summary_${RUN_NAME}.csv" \
  --per-item-dir "${EVAL_DIR}/per_item"

echo ""
echo "============================================="
echo "  MI-loss Training + Analysis + Eval Complete"
echo "============================================="
echo "Output dir: ${OUT_DIR}"
echo "SID file: ${OUT_DIR}/user_semantic_ids.jsonl"
echo "Eval summary: ${EVAL_DIR}/summary_${RUN_NAME}.csv"
echo "Per-item CSV: ${EVAL_DIR}/per_item/${RUN_NAME}.per_item.csv"
echo "Log: ${LOG_FILE}"
echo "Finished at: $(date)"
echo "============================================="