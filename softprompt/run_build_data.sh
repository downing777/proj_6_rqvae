#!/usr/bin/env bash
# 数据构建流水线:
#   1) 用 dpo_title_gen.py 的输出 (每个 item×SID 的个性化标题) 构造 far-SID DPO pairs
#   2) 划分 train/test
#   3) 从 train 集构造 SFT 语料
#
# 前置条件:
#   - 已运行 run_dpo_title_gen.sh 完成标题生成
#
# 使用:
#   cd /home/yuanhanyang.yhy/proj_6_rqvae
#   bash softprompt/run_build_data.sh
#
# 可覆盖的环境变量:
#   TITLE_GEN_JSONL   dpo_title_gen.py 的输出文件
#   OUT_DIR           输出根目录
#   VERSION           版本标识 (区分不同实验)
#   USER_NPZ          user embeddings npz 路径
#   USER_SID_JSONL    user -> SID 映射文件
#   ITEM_JSONL        商品元数据 (build_sft_from_dpo 可选参数)
#   TEST_RATIO        测试集比例
#   DISTANCE          far-SID 距离度量 (l2 / cosine)
#   SEED              随机种子

set -euo pipefail
ROOT="${ROOT:-/home/yuanhanyang.yhy/proj_6_rqvae}"
cd "${ROOT}"

eval "$(conda shell.bash hook)"
conda activate softprompt

# ---- 版本和路径 ----
VERSION="${VERSION:-qwen3-30B-farsid_11}"
OUT_DIR="${OUT_DIR:-/home/yuanhanyang.yhy/project_6_outputs}"

# dpo_title_gen.py 的输出 (Phase 1 产出, 每行 {item_id, sid, title_chosen, context, user_id})
TITLE_GEN_JSONL="${TITLE_GEN_JSONL:-${OUT_DIR}/data/dpo_electronics_generated_Qwen3-30B_xlength.jsonl}"

# User embedding 和 SID 映射
USER_NPZ="${USER_NPZ:-/home/yuanhanyang.yhy/model_hub/amazon_user/amazon_user_item_dataset.user.npz}"
USER_SID_JSONL="${USER_SID_JSONL:-/home/yuanhanyang.yhy/model_hub/amazon_user/user_semantic_ids.jsonl}"

# 商品元数据 (build_sft_from_dpo 的可选参数)
ITEM_JSONL="${ITEM_JSONL:-/home/yuanhanyang.yhy/model_hub/amazon_user/raw/step4/final_filtered_item_meta_electronics.jsonl}"

# 参数
TEST_RATIO="${TEST_RATIO:-0.1}"
DISTANCE="${DISTANCE:-l2}"
SEED="${SEED:-42}"

# 输出路径
DPO_JSONL="${OUT_DIR}/data/dpo_far_sid_1to1_${VERSION}.jsonl"
SPLIT_DIR="${OUT_DIR}/split_${VERSION}"
SFT_JSONL="${OUT_DIR}/sft_from_chosen_title_${VERSION}.jsonl"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "============================================="
echo "  Data Build Pipeline"
echo "============================================="
echo "  Title gen input: ${TITLE_GEN_JSONL}"
echo "  User NPZ:        ${USER_NPZ}"
echo "  User SID:        ${USER_SID_JSONL}"
echo "  Distance:        ${DISTANCE}"
echo "  Version:         ${VERSION}"
echo "  Output DPO:      ${DPO_JSONL}"
echo "  Output Split:    ${SPLIT_DIR}/"
echo "  Output SFT:      ${SFT_JSONL}"
echo "============================================="
echo ""

# ---- Prereq check ----
if [[ ! -f "${TITLE_GEN_JSONL}" ]]; then
  echo "ERROR: Title gen output not found: ${TITLE_GEN_JSONL}"
  echo "Please run run_dpo_title_gen.sh first."
  exit 1
fi

# ======================================================================
# Step 1: Build far-SID DPO pairs (1:1)
# ======================================================================
echo "---- [1/3] Build far-SID DPO pairs ----"
if [[ -f "${DPO_JSONL}" && "${FORCE_REBUILD_DPO:-0}" != "1" ]]; then
  echo "  Found existing: ${DPO_JSONL} ($(wc -l < "${DPO_JSONL}") lines)"
  echo "  Skipping. Set FORCE_REBUILD_DPO=1 to rebuild."
else
  python3 softprompt/data/build_dpo_far_sid_1to1.py \
    --input-jsonl   "${TITLE_GEN_JSONL}" \
    --user-npz      "${USER_NPZ}" \
    --user-sid-jsonl "${USER_SID_JSONL}" \
    --distance      "${DISTANCE}" \
    --out           "${DPO_JSONL}"
  echo "  Done: ${DPO_JSONL} ($(wc -l < "${DPO_JSONL}") lines)"
fi
echo ""

# ======================================================================
# Step 2: Split train/test
# ======================================================================
echo "---- [2/3] Split train/test ----"
TRAIN_JSONL="${SPLIT_DIR}/train.jsonl"
TEST_JSONL="${SPLIT_DIR}/test.jsonl"
TEST_INFER_JSONL="${SPLIT_DIR}/test_infer.jsonl"

if [[ -s "${TRAIN_JSONL}" && -s "${TEST_JSONL}" && "${FORCE_SPLIT:-0}" != "1" ]]; then
  echo "  Found existing split:"
  echo "    ${TRAIN_JSONL}       ($(wc -l < "${TRAIN_JSONL}") lines)"
  echo "    ${TEST_JSONL}        ($(wc -l < "${TEST_JSONL}") lines)"
  echo "    ${TEST_INFER_JSONL}  ($(wc -l < "${TEST_INFER_JSONL}") lines)"
  echo "  Skipping. Set FORCE_SPLIT=1 to re-split."
else
  mkdir -p "${SPLIT_DIR}"
  python3 softprompt/data/split_data.py \
    --input-jsonl "${DPO_JSONL}" \
    --output-dir "${SPLIT_DIR}" \
    --test-ratio "${TEST_RATIO}" \
    --seed "${SEED}"
  echo "  Done:"
  echo "    train: $(wc -l < "${TRAIN_JSONL}") lines"
  echo "    test:  $(wc -l < "${TEST_JSONL}") lines"
  echo "    test_infer: $(wc -l < "${TEST_INFER_JSONL}") lines"
fi
echo ""

# ======================================================================
# Step 3: Build SFT data from train split
# ======================================================================
echo "---- [3/3] Build SFT data ----"
if [[ -f "${SFT_JSONL}" && "${FORCE_REBUILD_SFT:-0}" != "1" ]]; then
  echo "  Found existing: ${SFT_JSONL} ($(wc -l < "${SFT_JSONL}") lines)"
  echo "  Skipping. Set FORCE_REBUILD_SFT=1 to rebuild."
else
  python3 softprompt/data/build_sft_from_dpo.py \
    --dpo-jsonl "${TRAIN_JSONL}" \
    --item-jsonl "${ITEM_JSONL}" \
    --out "${SFT_JSONL}"
  echo "  Done: ${SFT_JSONL} ($(wc -l < "${SFT_JSONL}") lines)"
fi
echo ""

# ======================================================================
# Summary
# ======================================================================
echo "============================================="
echo "  Data Build Complete"
echo "============================================="
echo "  DPO pairs:    ${DPO_JSONL}"
echo "  Train split:  ${TRAIN_JSONL}"
echo "  Test split:   ${TEST_JSONL}"
echo "  Test infer:   ${TEST_INFER_JSONL}"
echo "  SFT data:     ${SFT_JSONL}"
echo ""
echo "  Next: update run_train.sh to use these paths, then:"
echo "    DPO_JSONL=${DPO_JSONL} VERSION=${VERSION} bash softprompt/run_train.sh"
echo "============================================="
