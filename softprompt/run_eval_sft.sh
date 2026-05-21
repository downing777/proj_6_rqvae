#!/usr/bin/env bash
# SFT-only 评测脚本: 用 SFT 模型在测试集上生成标题, 再用 LLM-as-judge 对比 original_title。
#
# 流程:
#   1) SFT prediction — 用 sid_sft.pt 在测试集上推理 → predictions_sft_${VERSION}.jsonl
#   2) LLM-as-judge   — 比较 SFT 生成 vs context 里的 Original title, 出胜率
#
# 与 run_eval.sh 的关系:
#   - 完全独立, 不互相覆盖
#   - 共用 predictions 文件 predictions_sft_${VERSION}.jsonl, 已有则跳过推理
#   - 输出 eval/summary 文件名带 _sft, 不会和 DPO 的结果冲突

set -euo pipefail
ROOT="/home/yuanhanyang.yhy/proj_6_rqvae"
cd "${ROOT}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate softprompt

QWEN_BASE="${QWEN_BASE:-/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/home/yuanhanyang.yhy/project_6_outputs}"

# ---- 版本标识: 必须与 run_train.sh 中的 VERSION 一致 ----
VERSION="${VERSION:-sft_chosen_dpo}"

# ---- Auto-nohup: 直接 bash run_eval_sft.sh 即可后台运行 ----
if [[ -z "${_EVAL_SFT_NOHUP_WRAPPER:-}" ]]; then
  export _EVAL_SFT_NOHUP_WRAPPER=1
  _BASE_MODEL_NAME="$(basename "${QWEN_BASE}")"
  _LOG_DIR="${OUT_DIR}/logs/${_BASE_MODEL_NAME}"
  mkdir -p "${_LOG_DIR}"
  _FULL_LOG="${_LOG_DIR}/eval_sft_full_${_BASE_MODEL_NAME}_${VERSION}.log"
  _PIDFILE="${_LOG_DIR}/eval_sft_${_BASE_MODEL_NAME}_${VERSION}.pid"

  nohup bash "$0" "$@" >> "${_FULL_LOG}" 2>&1 &
  _PID=$!
  echo "${_PID}" > "${_PIDFILE}"

  echo "============================================="
  echo "  SFT Evaluation launched in background"
  echo "============================================="
  echo "  PID: ${_PID}"
  echo "  Log: ${_FULL_LOG}"
  echo ""
  echo "  To monitor: tail -f ${_FULL_LOG}"
  echo "  To stop:    kill \$(cat ${_PIDFILE})"
  echo "============================================="
  exit 0
fi

SPLIT_DIR="${OUT_DIR}/split"
EVAL_DIR="${OUT_DIR}/eval"
SFT_DIR="${OUT_DIR}/weights/${VERSION}/sft"

TEST_INFER_JSONL="${SPLIT_DIR}/test_infer.jsonl"
TEST_DPO_JSONL="${SPLIT_DIR}/test.jsonl"

# Prediction & eval output
PRED_SFT="${EVAL_DIR}/predictions_sft_${VERSION}.jsonl"
EVAL_SFT="${EVAL_DIR}/eval_results_sft_${VERSION}.jsonl"
SUMMARY_SFT="${EVAL_DIR}/eval_summary_sft_${VERSION}.json"

# Checkpoint
SFT_CKPT="${SFT_DIR}/sid_sft.pt"

# Inference settings
INFER_GPU="${INFER_GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
TEMPERATURE="${TEMPERATURE:-0.7}"
EVAL_SAMPLES="${EVAL_SAMPLES:-200}"
SEED="${SEED:-42}"

# LLM judge settings
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8002/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-27B}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${INFER_GPU}"

mkdir -p "${EVAL_DIR}"

echo "============================================="
echo "  SFT Evaluation Pipeline"
echo "============================================="
echo "  Base model:   ${QWEN_BASE}"
echo "  Version:      ${VERSION}"
echo "  Test set:     ${TEST_INFER_JSONL}"
echo "  SFT ckpt:     ${SFT_CKPT}"
echo "  Judge:        ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"
echo "  Pred file:    ${PRED_SFT}"
echo "  Eval results: ${EVAL_SFT}"
echo "  Summary:      ${SUMMARY_SFT}"
echo "============================================="
echo ""

# ---- Check prerequisites ----
if [[ ! -f "${TEST_INFER_JSONL}" ]]; then
  echo "ERROR: Test infer data not found: ${TEST_INFER_JSONL}"
  echo "Please run run_train.sh first (which calls split_data.py)."
  exit 1
fi

if [[ ! -f "${SFT_CKPT}" ]]; then
  echo "ERROR: SFT checkpoint not found: ${SFT_CKPT}"
  echo "Please run run_train.sh first to train SFT (VERSION=${VERSION})."
  exit 1
fi

# ======================================================================
# Step 1: SFT Prediction (generate if not exist)
# ======================================================================
echo "---- [1/2] SFT Prediction ----"
if [[ -f "${PRED_SFT}" ]]; then
  echo "  Found existing: ${PRED_SFT} ($(wc -l < "${PRED_SFT}") lines)"
  echo "  Skipping generation. Delete the file to regenerate."
else
  echo "  Generating SFT predictions (sampling ${EVAL_SAMPLES} from test set)..."
  python3 softprompt/infer/generate_title.py \
    --input-jsonl "${TEST_INFER_JSONL}" \
    --base-model "${QWEN_BASE}" \
    --sid-ckpt "${SFT_CKPT}" \
    --output-jsonl "${PRED_SFT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --max-samples "${EVAL_SAMPLES}" \
    --seed "${SEED}"
  echo "  Done: ${PRED_SFT} ($(wc -l < "${PRED_SFT}") lines)"
fi
echo ""

# Quick sample peek
echo "  SFT Sample Outputs (first 5):"
echo "  ────────────────────────────────────────"
head -5 "${PRED_SFT}" | python3 -c "
import sys, json
for i, line in enumerate(sys.stdin, 1):
    row = json.loads(line.strip())
    sid = row.get('sid', [])
    title = row.get('generated_text', '')
    item_id = row.get('item_id', '')
    print(f'  {i:2d}. item={item_id}  sid={sid}')
    print(f'      => {title}')
" 2>/dev/null || head -5 "${PRED_SFT}"
echo "  ────────────────────────────────────────"
echo ""

# ======================================================================
# Step 2: LLM-as-judge evaluation
# ======================================================================
# 注意: judge 对比的是 SFT 生成 vs context 里的 Original title (不是 DPO chosen),
# 跟 run_eval.sh 评 DPO 用的是同一套 judge 口径, 结果之间可以横向对比。
echo "---- [2/2] LLM-as-Judge (SFT vs Original title) ----"
echo "  Judge model: ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"
python3 softprompt/eval/offline_eval.py \
  --pred-jsonl "${PRED_SFT}" \
  --dpo-jsonl "${TEST_DPO_JSONL}" \
  --output-jsonl "${EVAL_SFT}" \
  --summary-json "${SUMMARY_SFT}" \
  --openai-base-url "${JUDGE_BASE_URL}" \
  --openai-api-key "${JUDGE_API_KEY}" \
  --model "${JUDGE_MODEL}" \
  --max-concurrency 8 \
  --extra-body-json '{"top_k": 1, "chat_template_kwargs": {"enable_thinking": false}}'
echo "  Done!"
echo ""

# ======================================================================
# Summary
# ======================================================================
echo "============================================="
echo "  SFT Evaluation Complete"
echo "============================================="
echo "  Predictions: ${PRED_SFT}"
echo "  Eval results: ${EVAL_SFT}"
echo "  Summary: ${SUMMARY_SFT}"
if [[ -f "${SUMMARY_SFT}" ]]; then
  echo ""
  echo "  SFT Summary (vs Original title):"
  python3 -c "
import json
with open('${SUMMARY_SFT}') as f:
    s = json.load(f)
o = s.get('overall', {})
print(f'    Samples: {s.get(\"sample_count\", 0)}')
print(f'    Generated win: {o.get(\"generated_win_rate\", 0):.1%}')
print(f'    Tie: {o.get(\"tie_rate\", 0):.1%}')
print(f'    Original win: {o.get(\"original_win_rate\", 0):.1%}')
print(f'    Strict win rate: {s.get(\"strict_win_rate\", 0):.1%}')
" 2>/dev/null || echo "    (see ${SUMMARY_SFT})"
fi
echo ""
echo "  To view samples: head -5 ${PRED_SFT}"
echo "  To view judge:   head -5 ${EVAL_SFT}"
