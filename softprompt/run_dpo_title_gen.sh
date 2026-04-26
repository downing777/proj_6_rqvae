#!/usr/bin/env bash
# DPO 标题生成：后台运行 + 日志重定向
# 用法: bash run_dpo_title_gen_nohup.sh
# 查看: tail -f dpo_electronics_generated.nohup.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG="${SCRIPT_DIR}/logs/dpo_electronics_generated.log"
PIDFILE="${SCRIPT_DIR}/logs/dpo_title_gen.pid"
mkdir -p "${SCRIPT_DIR}/data"

# 如未设置，可选（脚本内已有默认 http://localhost:8000/v1）
# export OPENAI_BASE_URL="http://localhost:8000/v1"
# export OPENAI_API_KEY="EMPTY"

nohup python3 dpo_title_gen.py \
  --user-sid /nfs5/yhy/tn/proj_6_rqvae/amazon_user/sid/user_semantic_ids.jsonl \
  --item-jsonl /nfs5/yhy/tn/proj_6_rqvae/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl \
  --reviews-jsonl /nfs5/yhy/tn/proj_6_rqvae/amazon_user/raw/step4/final_target_user_reviews_by_category/final_target_user_reviews_electronics.jsonl \
  --output-jsonl /nfs5/yhy/tn/proj_6_rqvae/softprompt/data/dpo_electronics_generated.jsonl \
  --model "/nfs5/yhy/model_hub/Qwen2.5-14B-Instruct" \
  --max-tokens 2048 \
  --max-concurrency 4 \
  >> "${LOG}" 2>&1 &

echo $! | tee "${PIDFILE}"
echo "Started in background. Log: ${LOG}  PID: $(cat "${PIDFILE}")"
echo "tail -f ${LOG}"
