#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_NAME="Qwen3.5-27B"
ENDPOINTS="http://localhost:8002/v1,http://localhost:8003/v1,http://localhost:8004/v1,http://localhost:8005/v1,http://localhost:8006/v1,http://localhost:8007/v1,http://localhost:8008/v1"

OUTPUT_DIR="/home/yuanhanyang.yhy/project_6_outputs/data"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

OUTPUT_JSONL="${OUTPUT_DIR}/dpo_electronics_generated_${MODEL_NAME}.jsonl"
LOG="${LOG_DIR}/dpo_title_gen_${MODEL_NAME}.log"
PIDFILE="${LOG_DIR}/dpo_title_gen_${MODEL_NAME}.pid"

nohup python3 dpo_title_gen.py \
  --user-sid /home/yuanhanyang.yhy/model_hub/amazon_user/user_semantic_ids.jsonl \
  --item-jsonl /home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl \
  --reviews-jsonl /home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_target_user_reviews_by_category/final_target_user_reviews_electronics.jsonl \
  --output-jsonl "${OUTPUT_JSONL}" \
  --openai-base-urls "${ENDPOINTS}" \
  --openai-api-key "EMPTY" \
  --model "${MODEL_NAME}" \
  --max-tokens 2048 \
  --max-concurrency 32 \
  --num-hard-negatives 3 \
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