#!/usr/bin/env bash
# 三阶段 (默认只训 SID 可学习前缀, 冻结基座):
#   0) 划分 train/test
#   1) 构造 SFT 语料 -> sft/
#   2) SFT -> sid_sft.pt, DPO -> sid_dpo.pt
#
# 使用: 直接 bash softprompt/run_train.sh
#   - 自动 nohup 后台运行，断网/关终端不影响
#   - 日志自动写入 logs 目录
#   - 训练长度仅由 MAX_STEPS_SFT / MAX_STEPS_DPO 控制

set -euo pipefail
ROOT="/home/yuanhanyang.yhy/proj_6_rqvae"
cd "${ROOT}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate softprompt

# ---- 版本标识: 用于区分不同实验, 会作为后缀附加到 log/weight 文件名上 ----
VERSION="${VERSION:-qwen3-30B-farsid-11-sidrebuild-0527}"
QWEN_BASE="${QWEN_BASE:-/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B}"

# ---- Auto-nohup: 如果不是被 nohup 调用的，则自动用 nohup 重启自己 ----
if [[ -z "${_TRAIN_NOHUP_WRAPPER:-}" ]]; then
  export _TRAIN_NOHUP_WRAPPER=1
  _BASE_MODEL_NAME="$(basename "${QWEN_BASE}")"
  _LOG_DIR="/home/yuanhanyang.yhy/project_6_outputs/logs/${_BASE_MODEL_NAME}"
  mkdir -p "${_LOG_DIR}"
  _FULL_LOG="${_LOG_DIR}/train_full_${_BASE_MODEL_NAME}_${VERSION}.log"
  _PIDFILE="${_LOG_DIR}/train_${_BASE_MODEL_NAME}_${VERSION}.pid"

  nohup bash "$0" "$@" >> "${_FULL_LOG}" 2>&1 &
  _PID=$!
  echo "${_PID}" > "${_PIDFILE}"

  echo "============================================="
  echo "  Training launched in background"
  echo "============================================="
  echo "  Base model: ${QWEN_BASE} (${_BASE_MODEL_NAME})"
  echo "  PID: ${_PID}"
  echo "  Log: ${_FULL_LOG}"
  echo "  PID file: ${_PIDFILE}"
  echo ""
  echo "  To monitor: tail -f ${_FULL_LOG}"
  echo "  To stop:    kill \$(cat ${_PIDFILE})"
  echo "============================================="
  exit 0
fi

DPO_JSONL="${DPO_JSONL:-/home/yuanhanyang.yhy/project_6_outputs/data/dpo/dpo_far_sid_1to1_${VERSION}.jsonl}"
ITEM_JSONL="${ITEM_JSONL:-/home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl}"
OUT_DIR="${OUT_DIR:-/home/yuanhanyang.yhy/project_6_outputs}"
SPLIT_DIR="${OUT_DIR}/split/${VERSION}"
SFT_JSONL="${OUT_DIR}/data/sft/sft_from_chosen_title_${VERSION}.jsonl"
SFT_DIR="${OUT_DIR}/weights/${VERSION}/sft"
DPO_DIR="${OUT_DIR}/weights/${VERSION}/dpo"

MAX_STEPS_SFT="${MAX_STEPS_SFT:-1000}"
MAX_STEPS_DPO="${MAX_STEPS_DPO:-1000}"
BATCH_SFT="${BATCH_SFT:-2}"
BATCH_DPO="${BATCH_DPO:-2}"
MAX_LEN="${MAX_LEN:-2048}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-0}"
LR_SFT="${LR_SFT:-3e-6}"
LR_DPO="${LR_DPO:-3e-6}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_SFT_COEF="${DPO_SFT_COEF:-0.2}"   # >0 开启 SFT 正则项, 推荐 0.1~0.5
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"

TRAIN_GPU="${TRAIN_GPU:-1}"
BASE_MODEL_NAME="$(basename "${QWEN_BASE}")"
LOG_DIR="${OUT_DIR}/logs/${BASE_MODEL_NAME}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "============================================="
echo "  Training Pipeline Started"
echo "============================================="
echo "  Base model: ${QWEN_BASE} (${BASE_MODEL_NAME})"
echo "  DPO data: ${DPO_JSONL}"
echo "  Output: ${OUT_DIR}"
echo "  SFT steps: ${MAX_STEPS_SFT}, DPO steps: ${MAX_STEPS_DPO}"
echo "  Log: ${LOG_DIR}/train_full_${BASE_MODEL_NAME}_${VERSION}.log"
echo ""
echo "  To stop: kill $"
echo "============================================="
echo ""

# ---- 0) 划分 train/test (按 (user_id, item_id) pair 级别随机划分)
TRAIN_DPO_JSONL="${SPLIT_DIR}/train.jsonl"
TEST_DPO_JSONL="${SPLIT_DIR}/test.jsonl"
TEST_INFER_JSONL="${SPLIT_DIR}/test_infer.jsonl"

if [[ -s "${TRAIN_DPO_JSONL}" && -s "${TEST_DPO_JSONL}" && -s "${TEST_INFER_JSONL}" \
      && "${FORCE_SPLIT:-0}" != "1" ]]; then
  echo "---- [0/3] Skip split: existing non-empty files detected ----"
  echo "  ${TRAIN_DPO_JSONL}    ($(wc -l < "${TRAIN_DPO_JSONL}") lines)"
  echo "  ${TEST_DPO_JSONL}     ($(wc -l < "${TEST_DPO_JSONL}") lines)"
  echo "  ${TEST_INFER_JSONL}   ($(wc -l < "${TEST_INFER_JSONL}") lines)"
  echo "  (set FORCE_SPLIT=1 to re-split from ${DPO_JSONL})"
else
  echo "---- [0/3] Splitting train/test ----"
  python3 softprompt/data/split_data.py \
    --input-jsonl "${DPO_JSONL}" \
    --output-dir "${SPLIT_DIR}" \
    --test-ratio "${TEST_RATIO}" \
    --seed "${SEED}"
fi

# ---- 1) SFT 语料 (仅使用训练集)
if [[ -s "${SFT_JSONL}" && "${FORCE_REBUILD_SFT_DATA:-0}" != "1" ]]; then
  echo "---- [1/3] Skip SFT data build: existing non-empty file detected ----"
  echo "  ${SFT_JSONL}  ($(wc -l < "${SFT_JSONL}") lines)"
  echo "  (set FORCE_REBUILD_SFT_DATA=1 to rebuild)"
else
  echo "---- [1/3] Building SFT data ----"
  python3 softprompt/data/build_sft_from_dpo.py \
    --dpo-jsonl "${TRAIN_DPO_JSONL}" \
    --item-jsonl "${ITEM_JSONL}" \
    --out "${SFT_JSONL}"
fi

# ---- 2) SFT
echo "---- [2/3] SFT Training ----"
python3 softprompt/train/train_sft.py \
  --train-jsonl "${SFT_JSONL}" \
  --base-model "${QWEN_BASE}" \
  --output-dir "${SFT_DIR}" \
  --train-batch-size "${BATCH_SFT}" \
  --max-steps "${MAX_STEPS_SFT}" \
  --learning-rate "${LR_SFT}" \
  --max-length "${MAX_LEN}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}"

# ---- 3) DPO (仅使用训练集)
echo "---- [3/3] DPO Training ----"
python3 softprompt/train/train_dpo.py \
  --train-jsonl "${TRAIN_DPO_JSONL}" \
  --base-model "${QWEN_BASE}" \
  --sft-ckpt "${SFT_DIR}/sid_sft.pt" \
  --output-dir "${DPO_DIR}" \
  --train-batch-size "${BATCH_DPO}" \
  --max-steps "${MAX_STEPS_DPO}" \
  --learning-rate "${LR_DPO}" \
  --beta "${DPO_BETA}" \
  --sft-coef "${DPO_SFT_COEF}" \
  --max-length "${MAX_LEN}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}"

echo ""
echo "============================================="
echo "  Training Complete"
echo "============================================="
echo "SFT ckpt: ${SFT_DIR}/sid_sft.pt"
echo "DPO ckpt: ${DPO_DIR}/sid_dpo.pt"
echo "SFT loss curve: ${SFT_DIR}/sft_loss_curve.png"
echo "DPO loss curve: ${DPO_DIR}/dpo_loss_curve.png"
echo "Log: ${LOG_DIR}/train_full_${BASE_MODEL_NAME}_${VERSION}.log"
echo "Test set: ${TEST_INFER_JSONL}"
echo ""
echo "To run evaluation: bash softprompt/run_eval.sh"
