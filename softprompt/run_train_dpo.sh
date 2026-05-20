#!/usr/bin/env bash
# DPO from scratch: 不加载任何 SFT checkpoint, sid_prefix 从随机初始化开始,
# 完全靠 DPO loss 把 prefix 从 0 推到一个能区分 chosen/rejected 的状态。
#
# 用途:
#   - 验证"跳过 SFT 直接 DPO"在你这套数据上能不能 work
#   - 避免 SFT-as-copy 退化把 base model 锁死成抄写员
#
# 使用: 直接 bash softprompt/run_train_dpo.sh
#   - 自动 nohup 后台运行, 断网/关终端不影响
#   - 日志写入 logs 目录, 文件名与 run_train.sh 区分 (train_dpo_only_*.log)
#   - 训练长度仅由 MAX_STEPS_DPO 控制
#
# 前置条件:
#   - ${SPLIT_DIR}/train.jsonl 已存在 (DPO 偏好对训练数据)
#
# 注意 (跟以前那个 SFT-init 版本的区别):
#   - 不再要求 ${SFT_DIR}/sid_sft.pt 存在
#   - 不会向 train_dpo.py 传 --sft-ckpt, prefix 用默认随机 init

set -euo pipefail
ROOT="/home/yuanhanyang.yhy/proj_6_rqvae"
cd "${ROOT}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate softprompt

# ---- Auto-nohup: 不是 nohup 调用就重启自己 ----
# 注意: env var 名字和 PID/log 文件名都和 run_train.sh 不同, 避免互相覆盖
if [[ -z "${_DPO_NOHUP_WRAPPER:-}" ]]; then
  export _DPO_NOHUP_WRAPPER=1
  QWEN_BASE="${QWEN_BASE:-/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B}"
  _BASE_MODEL_NAME="$(basename "${QWEN_BASE}")"
  _LOG_DIR="/home/yuanhanyang.yhy/project_6_outputs/logs/${_BASE_MODEL_NAME}"
  mkdir -p "${_LOG_DIR}"
  _FULL_LOG="${_LOG_DIR}/train_dpo_only_${_BASE_MODEL_NAME}.log"
  _PIDFILE="${_LOG_DIR}/train_dpo_only_${_BASE_MODEL_NAME}.pid"

  nohup bash "$0" "$@" >> "${_FULL_LOG}" 2>&1 &
  _PID=$!
  echo "${_PID}" > "${_PIDFILE}"

  echo "============================================="
  echo "  DPO-from-scratch training launched in background"
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

# ---- 路径 ----
QWEN_BASE="${QWEN_BASE:-/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/home/yuanhanyang.yhy/project_6_outputs}"
SPLIT_DIR="${OUT_DIR}/split"
DPO_DIR="${OUT_DIR}/weights/dpo_only"

TRAIN_DPO_JSONL="${SPLIT_DIR}/train.jsonl"

# ---- DPO 超参 (调实验时主要改这几个 env var) ----
MAX_STEPS_DPO="${MAX_STEPS_DPO:-1000}"
BATCH_DPO="${BATCH_DPO:-1}"
LR_DPO="${LR_DPO:-3e-5}"
MAX_LEN="${MAX_LEN:-2048}"
MAX_CONTEXT_CHARS="${MAX_CONTEXT_CHARS:-0}"

TRAIN_GPU="${TRAIN_GPU:-0}"
BASE_MODEL_NAME="$(basename "${QWEN_BASE}")"
LOG_DIR="${OUT_DIR}/logs/${BASE_MODEL_NAME}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${DPO_DIR}"

# ---- 前置条件校验 (注意: 不再校验 SFT ckpt) ----
if [[ ! -f "${TRAIN_DPO_JSONL}" ]]; then
  echo "ERROR: DPO train file not found: ${TRAIN_DPO_JSONL}"
  echo "  请先跑 split (会用 split_data.py 生成):  bash softprompt/run_train.sh"
  exit 1
fi

echo "============================================="
echo "  DPO from scratch (no SFT init)"
echo "============================================="
echo "  Base model:  ${QWEN_BASE} (${BASE_MODEL_NAME})"
echo "  Train data:  ${TRAIN_DPO_JSONL}  ($(wc -l < "${TRAIN_DPO_JSONL}") lines)"
echo "  Output dir:  ${DPO_DIR}"
echo "  Log:         ${LOG_DIR}/train_dpo_only_${BASE_MODEL_NAME}.log"
echo ""
echo "  Hyperparams:"
echo "    MAX_STEPS_DPO     = ${MAX_STEPS_DPO}"
echo "    BATCH_DPO         = ${BATCH_DPO}"
echo "    LR_DPO            = ${LR_DPO}"
echo "    MAX_LEN           = ${MAX_LEN}"
echo "    MAX_CONTEXT_CHARS = ${MAX_CONTEXT_CHARS}"
echo ""
echo "  sid_prefix init   : random (SidPrefixEncoder 默认初始化)"
echo "  ref model init    : 跟 policy 完全相同 (deepcopy 后冻结)"
echo "============================================="
echo ""

# ---- DPO 训练 (不传 --sft-ckpt) ----
echo "---- DPO Training (from scratch) ----"
python3 softprompt/train/train_dpo.py \
  --train-jsonl "${TRAIN_DPO_JSONL}" \
  --base-model "${QWEN_BASE}" \
  --output-dir "${DPO_DIR}" \
  --train-batch-size "${BATCH_DPO}" \
  --max-steps "${MAX_STEPS_DPO}" \
  --learning-rate "${LR_DPO}" \
  --max-length "${MAX_LEN}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}"

echo ""
echo "============================================="
echo "  DPO Training Complete"
echo "============================================="
echo "DPO ckpt:        ${DPO_DIR}/sid_dpo.pt"
echo "DPO loss curve:  ${DPO_DIR}/dpo_loss_curve.png"
echo "Loss history:    ${DPO_DIR}/dpo_loss_history.json"
echo "Log:             ${LOG_DIR}/train_dpo_only_${BASE_MODEL_NAME}.log"
echo ""
echo "WARNING: 这个 ckpt 是 from-scratch 训出来的, 没有任何 SFT 基础。"
echo "         推理 (run_eval.sh) 时仍可加载, 但 base model 可能不会输出"
echo "         良好的标题格式 (因为没人教过它 \\nTitle: 之后该输出短标题)。"
echo ""
echo "To run evaluation: bash softprompt/run_eval.sh"
