#!/usr/bin/env bash
# 评测脚本: 先生成 predictions（如不存在），再调用 LLM-as-judge 评估
#
# 流程:
#   1) SFT prediction — 用 sid_sft.pt 在测试集上推理 → 只展示样本（不做 LLM judge）
#   2) DPO prediction — 用 sid_dpo.pt 在测试集上推理 → 调用 LLM-as-judge 评测
#
# 使用方式:
#   cd /home/yuanhanyang.yhy/proj_6_rqvae
#   bash softprompt/run_eval.sh
#
# 前置条件:
#   - 已运行 run_train.sh 完成训练（sid_sft.pt / sid_dpo.pt 已存在）
#   - vLLM judge 端点可用（用于 DPO 评测）

set -euo pipefail
ROOT="/home/yuanhanyang.yhy/proj_6_rqvae"
cd "${ROOT}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate softprompt

QWEN_BASE="${QWEN_BASE:-/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/home/yuanhanyang.yhy/project_6_outputs}"

# ---- 版本标识: 必须与 run_train.sh 中的 VERSION 一致, 用于定位权重和区分 eval 输出 ----
VERSION="${VERSION:-sft_dpo_farsid_11}"

# ---- Auto-nohup: 直接 bash run_eval.sh 即可后台运行 ----
if [[ -z "${_EVAL_NOHUP_WRAPPER:-}" ]]; then
  export _EVAL_NOHUP_WRAPPER=1
  _BASE_MODEL_NAME="$(basename "${QWEN_BASE}")"
  _LOG_DIR="${OUT_DIR}/logs/${_BASE_MODEL_NAME}"
  mkdir -p "${_LOG_DIR}"
  _FULL_LOG="${_LOG_DIR}/eval_full_${_BASE_MODEL_NAME}_${VERSION}.log"
  _PIDFILE="${_LOG_DIR}/eval_${_BASE_MODEL_NAME}_${VERSION}.pid"

  nohup bash "$0" "$@" >> "${_FULL_LOG}" 2>&1 &
  _PID=$!
  echo "${_PID}" > "${_PIDFILE}"

  echo "============================================="
  echo "  Evaluation launched in background"
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
DPO_DIR="${OUT_DIR}/weights/${VERSION}/dpo"

TEST_INFER_JSONL="${SPLIT_DIR}/test_infer.jsonl"

# Prediction files
PRED_SFT="${EVAL_DIR}/predictions_sft_${VERSION}.jsonl"
PRED_DPO="${EVAL_DIR}/predictions_dpo_${VERSION}.jsonl"

# Checkpoints
SFT_CKPT="${SFT_DIR}/sid_sft.pt"
DPO_CKPT="${DPO_DIR}/sid_dpo.pt"

# Inference settings
INFER_GPU="${INFER_GPU:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
TEMPERATURE="${TEMPERATURE:-0.7}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
SEED="${SEED:-42}"

# LLM judge settings
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8003/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-27B}"

# User-group evidence sources (real Amazon reviews + user->SID mapping)
REVIEWS_JSONL="${REVIEWS_JSONL:-/home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_target_user_reviews_by_category/final_target_user_reviews_electronics.jsonl}"
USER_SID_JSONL="${USER_SID_JSONL:-/home/yuanhanyang.yhy/model_hub/amazon_user/user_semantic_ids.jsonl}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${INFER_GPU}"

mkdir -p "${EVAL_DIR}"

echo "============================================="
echo "  Evaluation Pipeline"
echo "============================================="
echo "  Base model: ${QWEN_BASE}"
echo "  Test set: ${TEST_INFER_JSONL}"
echo "  SFT ckpt: ${SFT_CKPT}"
echo "  DPO ckpt: ${DPO_CKPT}"
echo "  Judge: ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"
echo "  Reviews: ${REVIEWS_JSONL}"
echo "  User->SID: ${USER_SID_JSONL}"
echo "============================================="
echo ""

# ---- Check prerequisites ----
if [[ ! -f "${TEST_INFER_JSONL}" ]]; then
  echo "ERROR: Test infer data not found: ${TEST_INFER_JSONL}"
  echo "Please run run_train.sh first (which calls split_data.py)."
  exit 1
fi

# ======================================================================
# Step 1: SFT Prediction (generate if not exist) — NO LLM judge
# ======================================================================
echo "---- [1/3] SFT Prediction ----"
if [[ -f "${PRED_SFT}" ]]; then
  echo "  Found existing: ${PRED_SFT} ($(wc -l < "${PRED_SFT}") lines)"
  echo "  Skipping generation. Delete the file to regenerate."
elif [[ ! -f "${SFT_CKPT}" ]]; then
  echo "  SKIPPED: SFT checkpoint not found (${SFT_CKPT})"
  echo "  Please run run_train.sh to train SFT first."
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

# Show SFT samples
if [[ -f "${PRED_SFT}" ]]; then
  echo "  SFT Sample Outputs (first 10):"
  echo "  ────────────────────────────────────────"
  head -10 "${PRED_SFT}" | python3 -c "
import sys, json
for i, line in enumerate(sys.stdin, 1):
    row = json.loads(line.strip())
    sid = row.get('sid', [])
    title = row.get('generated_text', '')
    item_id = row.get('item_id', '')
    print(f'  {i:2d}. item={item_id}  sid={sid}')
    print(f'      => {title}')
" 2>/dev/null || head -10 "${PRED_SFT}"
  echo "  ────────────────────────────────────────"
fi
echo ""

# ======================================================================
# Step 2: LLM-as-judge evaluation (SFT)
# ======================================================================
# echo "---- [2/4] LLM-as-Judge (SFT vs Original title) ----"
# if [[ ! -f "${PRED_SFT}" ]]; then
#   echo "  SKIPPED: SFT predictions not available."
#   EVAL_SFT="(not generated)"
# else
#   EVAL_SFT="${EVAL_DIR}/eval_results_sft_${VERSION}.jsonl"
#   SUMMARY_SFT="${EVAL_DIR}/eval_summary_sft_${VERSION}.json"
#   echo "  Running LLM judge on SFT predictions..."
#   echo "  Judge model: ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"
#   python3 softprompt/eval/offline_eval.py \
#     --pred-jsonl "${PRED_SFT}" \
#     --reviews-jsonl "${REVIEWS_JSONL}" \
#     --user-sid "${USER_SID_JSONL}" \
#     --output-jsonl "${EVAL_SFT}" \
#     --summary-json "${SUMMARY_SFT}" \
#     --openai-base-url "${JUDGE_BASE_URL}" \
#     --openai-api-key "${JUDGE_API_KEY}" \
#     --model "${JUDGE_MODEL}" \
#     --max-concurrency 8 \
#     --extra-body-json '{"top_k": 1, "chat_template_kwargs": {"enable_thinking": false}}' \
#   echo "  Done!"
# fi
# echo ""

# ======================================================================
# Step 3: DPO Prediction (generate if not exist)
# ======================================================================
echo "---- [3/4] DPO Prediction ----"
if [[ -f "${PRED_DPO}" ]]; then
  echo "  Found existing: ${PRED_DPO} ($(wc -l < "${PRED_DPO}") lines)"
  echo "  Skipping generation. Delete the file to regenerate."
elif [[ ! -f "${DPO_CKPT}" ]]; then
  echo "  SKIPPED: DPO checkpoint not found (${DPO_CKPT})"
  echo "  Please run run_train.sh to train DPO first."
else
  echo "  Generating DPO predictions (sampling ${EVAL_SAMPLES} from test set)..."
  python3 softprompt/infer/generate_title.py \
    --input-jsonl "${TEST_INFER_JSONL}" \
    --base-model "${QWEN_BASE}" \
    --sid-ckpt "${DPO_CKPT}" \
    --output-jsonl "${PRED_DPO}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --max-samples "${EVAL_SAMPLES}" \
    --seed "${SEED}"
  echo "  Done: ${PRED_DPO} ($(wc -l < "${PRED_DPO}") lines)"
fi
echo ""

# ======================================================================
# Step 3: LLM-as-judge evaluation (DPO only)
# ======================================================================
echo "---- [4/4] LLM-as-Judge (DPO vs Original title) ----"
if [[ ! -f "${PRED_DPO}" ]]; then
  echo "  SKIPPED: DPO predictions not available."
  EVAL_DPO="(not generated)"
else
  EVAL_DPO="${EVAL_DIR}/eval_results_dpo_${VERSION}.jsonl"
  SUMMARY_DPO="${EVAL_DIR}/eval_summary_dpo_${VERSION}.json"
  echo "  Running LLM judge on DPO predictions..."
  echo "  Judge model: ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"
  python3 softprompt/eval/offline_eval.py \
    --pred-jsonl "${PRED_DPO}" \
    --reviews-jsonl "${REVIEWS_JSONL}" \
    --user-sid "${USER_SID_JSONL}" \
    --output-jsonl "${EVAL_DPO}" \
    --summary-json "${SUMMARY_DPO}" \
    --openai-base-url "${JUDGE_BASE_URL}" \
    --openai-api-key "${JUDGE_API_KEY}" \
    --model "${JUDGE_MODEL}" \
    --max-concurrency 8 \
    --extra-body-json '{"top_k": 1, "chat_template_kwargs": {"enable_thinking": false}}' \
    || echo "  [WARN] DPO judge finished with partial errors (non-zero exit code)"
  echo "  Done!"
fi

echo ""
echo "============================================="
echo "  Evaluation Complete"
echo "============================================="
echo "  SFT predictions: ${PRED_SFT}"
echo "  SFT eval results: ${EVAL_SFT:-N/A}"
echo "  DPO predictions: ${PRED_DPO}"
echo "  DPO eval results: ${EVAL_DPO}"
# Pretty-print summary 的字段对齐 offline_eval.py 当前 schema:
#   - personalization: A/B 盲比, 报 generated_better/tie/original_better 三个比例
#   - fluency: 单向 not-worse 检查, 报 not-worse 比例 (越高越好)
#   - hallucination: 单向幻觉检测, 报幻觉率 (越低越好)
#   - strict_win: 无幻觉 + 流畅没退化 + personalization 上 generated 更好
_print_summary() {
  local path="$1"
  python3 - "${path}" 2>/dev/null <<'PY' || echo "    (failed to parse; see ${path})"
import sys, json
with open(sys.argv[1]) as f:
    s = json.load(f)
per = s.get("per_dimension", {}).get("personalization", {})
flu = s.get("fluency", {})
hal = s.get("hallucination", {})
nw_rate = flu.get("not_worse_rate") or 0.0
hal_rate = hal.get("hallucination_rate") or 0.0
print(f'    Samples: {s.get("sample_count", 0)}')
print(f'    Personalization (generated vs original):')
print(f'      generated_better : {per.get("generated_win_rate", 0):.1%}')
print(f'      tie              : {per.get("tie_rate", 0):.1%}')
print(f'      original_better  : {per.get("original_win_rate", 0):.1%}')
print(f'    Fluency not-worse rate : {nw_rate:.1%}  '
      f'({flu.get("not_worse_count", 0)}/{flu.get("checked", 0)})')
print(f'    Hallucination rate     : {hal_rate:.1%}  '
      f'({hal.get("hallucination_count", 0)}/{hal.get("checked", 0)})')
print(f'    Strict win rate        : {s.get("strict_win_rate", 0):.1%}')
PY
}

if [[ -f "${EVAL_DIR}/eval_summary_sft_${VERSION}.json" ]]; then
  echo ""
  echo "  SFT Summary (vs Original title):"
  _print_summary "${EVAL_DIR}/eval_summary_sft_${VERSION}.json"
fi
if [[ -f "${EVAL_DIR}/eval_summary_dpo_${VERSION}.json" ]]; then
  echo ""
  echo "  DPO Summary (vs Original title):"
  _print_summary "${EVAL_DIR}/eval_summary_dpo_${VERSION}.json"
fi
echo ""
echo "  To view SFT judge: head -5 ${EVAL_SFT:-}"
echo "  To view DPO judge: head -5 ${EVAL_DPO}"
