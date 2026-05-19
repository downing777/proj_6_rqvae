#!/usr/bin/env bash
# 端到端评测脚本 (全新流程，与 run_eval.sh 独立)
#
# 流程:
#   1) 从原始 user-item 数据中随机采样 N 个 (item, SID) 对，构建 context
#   2) 用 DPO 模型 (sid_dpo.pt) 在采样数据上推理生成标题
#   3) 调用 LLM-as-judge 评测 (生成标题 vs 原商品标题)
#
# 使用方式:
#   cd /home/yuanhanyang.yhy/proj_6_rqvae
#   bash softprompt/run_eval_e2e.sh
#
# 自定义采样数量:
#   EVAL_SAMPLES=500 bash softprompt/run_eval_e2e.sh

set -euo pipefail
ROOT="/home/yuanhanyang.yhy/proj_6_rqvae"
cd "${ROOT}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate softprompt

# ---- Config ----
QWEN_BASE="${QWEN_BASE:-/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/home/yuanhanyang.yhy/project_6_outputs}"
EVAL_DIR="${OUT_DIR}/eval_e2e"

# Data sources
USER_SID="/home/yuanhanyang.yhy/model_hub/amazon_user/user_semantic_ids.jsonl"
ITEM_META="/home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl"
REVIEWS="/home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_target_user_reviews_by_category/final_target_user_reviews_electronics.jsonl"

# Checkpoints
DPO_CKPT="${OUT_DIR}/dpo/sid_dpo.pt"

# Eval parameters
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
TEMPERATURE="${TEMPERATURE:-0.7}"
INFER_GPU="${INFER_GPU:-0}"

# LLM judge
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8002/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-27B}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${INFER_GPU}"

# Output files
EVAL_SAMPLES_JSONL="${EVAL_DIR}/eval_samples_${EVAL_SAMPLES}.jsonl"
PRED_DPO="${EVAL_DIR}/predictions_dpo_${EVAL_SAMPLES}.jsonl"
JUDGE_RESULTS="${EVAL_DIR}/judge_results_${EVAL_SAMPLES}.jsonl"
JUDGE_SUMMARY="${EVAL_DIR}/judge_summary_${EVAL_SAMPLES}.json"

# ---- Auto-nohup ----
if [[ -z "${_EVAL_E2E_NOHUP:-}" ]]; then
  export _EVAL_E2E_NOHUP=1
  _BASE_MODEL_NAME="$(basename "${QWEN_BASE}")"
  _LOG_DIR="${OUT_DIR}/logs/${_BASE_MODEL_NAME}"
  mkdir -p "${_LOG_DIR}"
  _FULL_LOG="${_LOG_DIR}/eval_e2e_${_BASE_MODEL_NAME}.log"
  _PIDFILE="${_LOG_DIR}/eval_e2e_${_BASE_MODEL_NAME}.pid"

  nohup bash "$0" "$@" >> "${_FULL_LOG}" 2>&1 &
  _PID=$!
  echo "${_PID}" > "${_PIDFILE}"

  echo "============================================="
  echo "  E2E Evaluation launched in background"
  echo "============================================="
  echo "  PID: ${_PID}"
  echo "  Log: ${_FULL_LOG}"
  echo "  Samples: ${EVAL_SAMPLES}"
  echo ""
  echo "  To monitor: tail -f ${_FULL_LOG}"
  echo "  To stop:    kill \$(cat ${_PIDFILE})"
  echo "============================================="
  exit 0
fi

mkdir -p "${EVAL_DIR}"

echo "============================================="
echo "  E2E Evaluation Pipeline"
echo "============================================="
echo "  Base model: ${QWEN_BASE}"
echo "  DPO ckpt: ${DPO_CKPT}"
echo "  Eval samples: ${EVAL_SAMPLES}"
echo "  GPU: ${INFER_GPU}"
echo "  Judge: ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"
echo "============================================="
echo ""

# ---- Check prerequisites ----
if [[ ! -f "${DPO_CKPT}" ]]; then
  echo "ERROR: DPO checkpoint not found: ${DPO_CKPT}"
  echo "Please run run_train.sh to train DPO first."
  exit 1
fi

# ======================================================================
# Step 1: Build eval data (sample user-item pairs from raw data)
# ======================================================================
echo "---- [1/3] Building eval data (sampling ${EVAL_SAMPLES} user-item pairs) ----"
if [[ -f "${EVAL_SAMPLES_JSONL}" ]]; then
  echo "  Found existing: ${EVAL_SAMPLES_JSONL} ($(wc -l < "${EVAL_SAMPLES_JSONL}") lines)"
  echo "  Skipping. Delete the file to regenerate."
else
  python3 softprompt/eval/build_eval_data.py \
    --user-sid "${USER_SID}" \
    --item-meta "${ITEM_META}" \
    --reviews "${REVIEWS}" \
    --output "${EVAL_SAMPLES_JSONL}" \
    --num-samples "${EVAL_SAMPLES}" \
    --seed "${SEED}"
fi
echo ""

# ======================================================================
# Step 2: Run inference with DPO model
# ======================================================================
echo "---- [2/3] Generating titles with DPO model ----"
if [[ -f "${PRED_DPO}" ]]; then
  echo "  Found existing: ${PRED_DPO} ($(wc -l < "${PRED_DPO}") lines)"
  echo "  Skipping. Delete the file to regenerate."
else
  python3 softprompt/infer/generate_title.py \
    --input-jsonl "${EVAL_SAMPLES_JSONL}" \
    --base-model "${QWEN_BASE}" \
    --sid-ckpt "${DPO_CKPT}" \
    --output-jsonl "${PRED_DPO}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}"
  echo "  Done: ${PRED_DPO} ($(wc -l < "${PRED_DPO}") lines)"
fi
echo ""

# Show some generated samples
if [[ -f "${PRED_DPO}" ]]; then
  echo "  DPO Generated Samples (first 10):"
  echo "  ────────────────────────────────────────"
  head -10 "${PRED_DPO}" | python3 -c "
import sys, json
for i, line in enumerate(sys.stdin, 1):
    row = json.loads(line.strip())
    sid = row.get('sid', [])
    title = row.get('generated_text', '')
    item_id = row.get('item_id', '')
    print(f'  {i:2d}. item={item_id}  sid={sid}')
    print(f'      => {title}')
" 2>/dev/null || head -10 "${PRED_DPO}"
  echo "  ────────────────────────────────────────"
fi
echo ""

# ======================================================================
# Step 3: LLM-as-judge evaluation
# ======================================================================
echo "---- [3/3] LLM-as-Judge Evaluation ----"
if [[ ! -f "${PRED_DPO}" ]]; then
  echo "  SKIPPED: DPO predictions not available."
else
  echo "  Running LLM judge..."
  echo "  Judge model: ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"
  python3 softprompt/eval/offline_eval.py \
    --pred-jsonl "${PRED_DPO}" \
    --output-jsonl "${JUDGE_RESULTS}" \
    --summary-json "${JUDGE_SUMMARY}" \
    --openai-base-url "${JUDGE_BASE_URL}" \
    --openai-api-key "${JUDGE_API_KEY}" \
    --model "${JUDGE_MODEL}" \
    --max-concurrency 8 \
    --extra-body-json '{"top_k": 1, "chat_template_kwargs": {"enable_thinking": false}}'
  echo "  Done!"
fi

echo ""
echo "============================================="
echo "  E2E Evaluation Complete"
echo "============================================="
echo "  Eval samples: ${EVAL_SAMPLES_JSONL}"
echo "  DPO predictions: ${PRED_DPO}"
echo "  Judge results: ${JUDGE_RESULTS}"
echo "  Judge summary: ${JUDGE_SUMMARY}"
if [[ -f "${JUDGE_SUMMARY}" ]]; then
  echo ""
  echo "  Summary:"
  python3 -c "
import json
with open('${JUDGE_SUMMARY}') as f:
    s = json.load(f)
o = s.get('overall', {})
print(f'    Samples evaluated: {s.get(\"sample_count\", 0)}')
print(f'    Generated win: {o.get(\"generated_win_rate\", 0):.1%}')
print(f'    Tie: {o.get(\"tie_rate\", 0):.1%}')
print(f'    Original win: {o.get(\"original_win_rate\", 0):.1%}')
print(f'    Strict win rate: {s.get(\"strict_win_rate\", 0):.1%}')
" 2>/dev/null || echo "    (see ${JUDGE_SUMMARY})"
fi
echo ""
