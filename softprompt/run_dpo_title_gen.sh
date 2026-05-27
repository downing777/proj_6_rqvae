#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# MODEL_NAME="claude-opus-4-6"
# API_URL="https://idealab.alibaba-inc.com/api/openai/v1/"
# API_KEY="7015f1753e78f3067053c6432a933cb7"

MODEL_NAME="Qwen3-30B"
API_URLS="http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1,http://localhost:8008/v1"
API_KEY="EMPTY"

USER_SID="/home/yuanhanyang.yhy/project_6_outputs/sid/exp_mi_cb32_ed128_w0p1_a1p4_b1p5_tau0p5_k32_s42/user_semantic_ids.jsonl"

OUTPUT_DIR="/home/yuanhanyang.yhy/project_6_outputs/data/dpo"
LOG_DIR="$/home/yuanhanyang.yhy/project_6_outputs/logs/dpo_title_gen"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

OUTPUT_JSONL="${OUTPUT_DIR}/dpo_electronics_generated_${MODEL_NAME}.jsonl"
LOG="${LOG_DIR}/dpo_title_gen_${MODEL_NAME}.log"
PIDFILE="${LOG_DIR}/dpo_title_gen_${MODEL_NAME}.pid"

nohup python3 dpo_title_gen.py \
  --user-sid "${USER_SID}" \
  --item-jsonl /home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl \
  --reviews-jsonl /home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_target_user_reviews_by_category/final_target_user_reviews_electronics.jsonl \
  --output-jsonl "${OUTPUT_JSONL}" \
  --openai-base-urls "${API_URLS}" \
  --openai-api-key "${API_KEY}" \
  --model "${MODEL_NAME}" \
  --max-tokens 2048 \
  --max-concurrency 8 \
  --extra-body-json '{"chat_template_kwargs": {"enable_thinking": false}}' \
  >> "${LOG}" 2>&1 &

echo $! | tee "${PIDFILE}"
echo ""
echo "=========================================="
echo "  DPO Title Gen Started"
echo "=========================================="
echo "  Model: ${MODEL_NAME}"
echo "  PID: $(cat "${PIDFILE}")"
echo "  Log: ${LOG}"
echo "  Output: ${OUTPUT_JSONL}"
echo "  Usage: ${OUTPUT_DIR}/dpo_electronics_generated_${MODEL_NAME}.usage.jsonl"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG}"
echo ""
echo "To stop this task:"
echo "  kill \$(cat ${PIDFILE})"
echo "  # or: pkill -f 'dpo_title_gen.py'"
echo "=========================================="
echo "=========================================="