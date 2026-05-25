#!/usr/bin/env bash
# 一键跑 API baseline:
#   1) 用 split_data.py 产出的 test_infer.jsonl 做零样本 API 标题生成
#      (采样口径与 run_eval.sh 完全一致: --max-samples + --seed 走 random.sample)
#   2) 跑同一个 offline_eval.py judge, 输出 personalization 胜率 / fluency not-worse rate /
#      hallucination rate / strict_win_rate
#   3) 末尾打摘要 + 跟 DPO predictions 校验 overlap
#
# 使用:
#   cd /home/yuanhanyang.yhy/proj_6_rqvae
#   bash baseline/run_baseline.sh
#
# 常用环境变量 (跟 run_eval.sh 对齐, 全部可覆盖):
#   ROOT, OUT_DIR, VERSION
#   JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL
#   REVIEWS_JSONL, USER_SID_JSONL
#   EVAL_SAMPLES, SEED
#
# baseline 自己的:
#   BASELINE_MODEL  默认 = ${JUDGE_MODEL}  (直接复用 judge 的 vLLM 端点)
#   BASELINE_TAG    baseline 输出文件后缀,  默认根据 BASELINE_MODEL 自动派生
#   TARGET_WORDS    baseline 目标标题词数, 默认 12 (跟训练目标对齐, 让对比公平)
#   BASELINE_CONC   baseline 推理并发, 默认 8
#   JUDGE_CONC      judge 并发, 默认 8 (实际打到 vLLM 是 3x, 三个 pass 并发)

set -euo pipefail
ROOT="${ROOT:-/home/yuanhanyang.yhy/proj_6_rqvae}"
cd "${ROOT}"

eval "$(conda shell.bash hook)"
conda activate softprompt

QWEN_BASE="${QWEN_BASE:-/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-/home/yuanhanyang.yhy/project_6_outputs}"
VERSION="${VERSION:-sft_dpo_farsid_11}"
BASELINE_MODEL="${BASELINE_MODEL:-Qwen3.5-9B}"

# ---- Auto-nohup: 直接 bash run_baseline.sh 即可后台运行 ----
if [[ -z "${_BASELINE_NOHUP_WRAPPER:-}" ]]; then
  export _BASELINE_NOHUP_WRAPPER=1
  _BASELINE_MODEL_TAG="$(echo "${BASELINE_MODEL}" | sed -E 's![/: ]+!_!g')"
  _LOG_DIR="${OUT_DIR}/logs/baseline"
  mkdir -p "${_LOG_DIR}"
  _FULL_LOG="${_LOG_DIR}/baseline_full_${_BASELINE_MODEL_TAG}_${VERSION}.log"
  _PIDFILE="${_LOG_DIR}/baseline_${_BASELINE_MODEL_TAG}_${VERSION}.pid"

  nohup bash "$0" "$@" >> "${_FULL_LOG}" 2>&1 &
  _PID=$!
  echo "${_PID}" > "${_PIDFILE}"

  echo "============================================="
  echo "  Baseline launched in background"
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
BASELINE_DIR="${OUT_DIR}/baseline"

TEST_INFER_JSONL="${SPLIT_DIR}/test_infer.jsonl"

# 推理 / 评估口径
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
SEED="${SEED:-42}"

# Judge (vLLM)
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8003/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3.5-27B}"
JUDGE_CONC="${JUDGE_CONC:-8}"

# Baseline 生成器 (可独立指定端点和模型, 默认复用 judge)
# BASELINE_BASE_URL="${BASELINE_BASE_URL:-https://idealab.alibaba-inc.com/api/openai/v1/}"
# BASELINE_API_KEY="${BASELINE_API_KEY:-7015f1753e78f3067053c6432a933cb7}"
BASELINE_BASE_URL="${BASELINE_BASE_URL:-http://localhost:8001/v1}"
BASELINE_API_KEY="${BASELINE_API_KEY:-EMPTY}"
BASELINE_CONC="${BASELINE_CONC:-8}"
TARGET_WORDS="${TARGET_WORDS:-12}"

# 把模型名里的斜杠 / 冒号 / 空格转成下划线, 当成文件后缀的默认值
_default_tag() {
  echo "$1" | sed -E 's![/: ]+!_!g'
}
BASELINE_TAG="${BASELINE_TAG:-$(_default_tag "${BASELINE_MODEL}")}"

# 用户画像证据源 (跟 run_eval.sh 同源)
REVIEWS_JSONL="${REVIEWS_JSONL:-/home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_target_user_reviews_by_category/final_target_user_reviews_electronics.jsonl}"
USER_SID_JSONL="${USER_SID_JSONL:-/home/yuanhanyang.yhy/model_hub/amazon_user/user_semantic_ids.jsonl}"

# 输出路径
PRED_BASELINE="${BASELINE_DIR}/predictions_${BASELINE_TAG}_${VERSION}.jsonl"
EVAL_BASELINE="${BASELINE_DIR}/eval_results_${BASELINE_TAG}_${VERSION}.jsonl"
SUMMARY_BASELINE="${BASELINE_DIR}/eval_summary_${BASELINE_TAG}_${VERSION}.json"

# DPO predictions 路径 (用于校验 overlap, 可选)
PRED_DPO="${EVAL_DIR}/predictions_dpo_${VERSION}.jsonl"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${BASELINE_DIR}"

echo "============================================="
echo "  Baseline Pipeline (API zero-shot)"
echo "============================================="
echo "  Test set:       ${TEST_INFER_JSONL}"
echo "  Samples / Seed: ${EVAL_SAMPLES} / ${SEED}"
echo "  Baseline model: ${BASELINE_MODEL}  @ ${BASELINE_BASE_URL}"
echo "  Target words:   ${TARGET_WORDS}"
echo "  Judge model:    ${JUDGE_MODEL}     @ ${JUDGE_BASE_URL}"
echo "  Output tag:     ${BASELINE_TAG}_${VERSION}"
echo "  Reviews:        ${REVIEWS_JSONL}"
echo "  User->SID:      ${USER_SID_JSONL}"
echo "============================================="
echo ""

# ---- Prereq ----
if [[ ! -f "${TEST_INFER_JSONL}" ]]; then
  echo "ERROR: Test infer data not found: ${TEST_INFER_JSONL}"
  echo "Please run run_train.sh first (which calls split_data.py)."
  exit 1
fi

# ======================================================================
# Step 1: Baseline inference (API zero-shot)
# ======================================================================
echo "---- [1/3] Baseline Inference ----"
if [[ -f "${PRED_BASELINE}" ]]; then
  echo "  Found existing: ${PRED_BASELINE} ($(wc -l < "${PRED_BASELINE}") lines)"
  echo "  Will resume / skip duplicates via --skip-existing."
fi

# 备注: api_title_gen.py 内部 --max-samples + --random-sample + --seed 走 random.sample,
# 跟 generate_title.py 是同一套 numpy/random 调用, 同 seed 应选出同一批 500 条
# (前提是 test_infer.jsonl 没空行/缺字段, split_data.py 出来的都符合)。
python3 -m baseline.api_title_gen \
  --input-jsonl     "${TEST_INFER_JSONL}" \
  --output-jsonl    "${PRED_BASELINE}" \
  --openai-base-url "${BASELINE_BASE_URL}" \
  --openai-api-key  "${BASELINE_API_KEY}" \
  --model           "${BASELINE_MODEL}" \
  --target-words    "${TARGET_WORDS}" \
  --max-samples     "${EVAL_SAMPLES}" \
  --random-sample \
  --seed            "${SEED}" \
  --max-concurrency "${BASELINE_CONC}" \
  --extra-body-json '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --skip-existing

echo "  Done: ${PRED_BASELINE} ($(wc -l < "${PRED_BASELINE}") lines)"
echo ""

# 抽 5 条肉眼看下
if [[ -f "${PRED_BASELINE}" ]]; then
  echo "  Baseline Sample Outputs (first 5):"
  echo "  ────────────────────────────────────────"
  head -5 "${PRED_BASELINE}" | python3 -c "
import sys, json
for i, line in enumerate(sys.stdin, 1):
    r = json.loads(line)
    print(f'  {i}. item={r[\"item_id\"]} sid={r[\"sid\"]}')
    print(f'     orig : {r.get(\"original_title\", \"\")[:120]}')
    print(f'     gen  : {r.get(\"generated_text\", \"\")[:120]}')
" 2>/dev/null || head -5 "${PRED_BASELINE}"
  echo "  ────────────────────────────────────────"
fi
echo ""

# ======================================================================
# Step 2: LLM-as-judge on baseline predictions
# ======================================================================
echo "---- [2/3] LLM-as-Judge (Baseline vs Original title) ----"
echo "  Judge model: ${JUDGE_MODEL} @ ${JUDGE_BASE_URL}"

python3 softprompt/eval/offline_eval.py \
  --pred-jsonl      "${PRED_BASELINE}" \
  --reviews-jsonl   "${REVIEWS_JSONL}" \
  --user-sid        "${USER_SID_JSONL}" \
  --output-jsonl    "${EVAL_BASELINE}" \
  --summary-json    "${SUMMARY_BASELINE}" \
  --openai-base-url "${JUDGE_BASE_URL}" \
  --openai-api-key  "${JUDGE_API_KEY}" \
  --model           "${JUDGE_MODEL}" \
  --max-concurrency "${JUDGE_CONC}" \
  --extra-body-json '{"top_k": 1, "chat_template_kwargs": {"enable_thinking": false}}' \
  || echo "  [WARN] Baseline judge finished with partial errors (non-zero exit code)"

echo "  Done!"
echo ""

# ======================================================================
# Step 3: Summary + overlap check vs DPO
# ======================================================================
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

echo "============================================="
echo "  Baseline Pipeline Complete"
echo "============================================="
echo "  Predictions: ${PRED_BASELINE}"
echo "  Eval rows  : ${EVAL_BASELINE}"
echo "  Summary    : ${SUMMARY_BASELINE}"
echo ""

if [[ -f "${SUMMARY_BASELINE}" ]]; then
  echo "  Baseline Summary (${BASELINE_TAG}, vs Original title):"
  _print_summary "${SUMMARY_BASELINE}"
  echo ""
fi

echo ""
echo "  To re-judge baseline only (after editing prompts): "
echo "    rm -f ${EVAL_BASELINE} ${EVAL_BASELINE%.jsonl}.usage.jsonl ${SUMMARY_BASELINE}"
echo "    bash baseline/run_baseline.sh   # 推理已缓存, 只会重跑 judge"
echo ""
