#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build SFT training data from DPO JSONL.

SFT 目标 (P0 修复后): 给定 SID softprompt + 商品 context, 让模型学会输出
**该 SID 群体的偏好标题 (title_chosen)**, 而**不是**复制 context 里的 Original title。

为什么不再用 original_title 做 target:
  Context 里已经有一行 "Original title: BUFFALO ...", 如果 target 也是同一字符串,
  SFT 退化成 attention 指针拷贝, loss 几步就降到 0, 但模型只学会"找+抄"这个
  平凡任务。base model 被锁死成抄写员, 后续 DPO 怎么训也撼不动。

改成 title_chosen 之后:
  - target 不在 prompt 里出现, 模型必须真做"读 features + 理解 SID prefix → 生成"
  - SFT loss 不会降到 0, 健康值在 1.5-2.5 之间
  - 模型保留生成能力 + 学会在标题末尾吐 EOS, DPO 才有可优化基础

输入:
  --dpo-jsonl: DPO 数据 (含 user_id, item_id, sid, context, title_chosen)
  --item-jsonl: (可选, 旧版本兼容用) 现已不再使用, 仅保留参数避免 run_train.sh 改动

输出:
  每行: {"item_id": "...", "sid": [x,y,z], "context": "...", "target_title": "<chosen>",
         "user_id": "<同 (item,sid) 下首个见到的 user_id, 仅作参考>"}

去重: (item_id, sid) 唯一。同一 (item, sid) 下不同 user 共享同一个 chosen,
      去重避免该样本被重复采样、训练数据被高频 (item, sid) 主导。
      user_id 用第一个见到的, 训练用不上, 仅作 trace 参考。

Usage:
  python3 softprompt/data/build_sft_from_dpo.py \
    --dpo-jsonl data/split/train.jsonl \
    --out /path/to/sft_from_item_title.jsonl
  # --item-jsonl 已不需要传, 传了也不会用
"""

import argparse
import json
import os
from typing import Any, Dict, List, Set, Tuple


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SFT data: SID prefix + context -> SID-personalized chosen title."
    )
    parser.add_argument("--dpo-jsonl", type=str, required=True, help="DPO 数据文件")
    parser.add_argument(
        "--item-jsonl",
        type=str,
        default=None,
        help="(legacy) 已不使用; 旧的 SFT target=original 版本需要从这里查 original_title。"
             "保留参数仅为了 run_train.sh 不用改。",
    )
    parser.add_argument("--out", type=str, required=True, help="输出 SFT JSONL 路径")
    args = parser.parse_args()

    if args.item_jsonl:
        print(f"NOTE: --item-jsonl 已不再使用 (target 改为 title_chosen)。"
              f"传入路径 {args.item_jsonl} 将被忽略。")

    # Load DPO data
    dpo_rows = load_jsonl(args.dpo_jsonl)
    print(f"Loaded {len(dpo_rows)} DPO rows from {args.dpo_jsonl}.")

    # Build SFT data: deduplicate by (item_id, sid)
    # 同一 (item_id, sid) 在 DPO 数据里会因不同 user_id 出现多次, 但 chosen 一样,
    # 在 SFT 里没必要重复学。
    seen_keys: Set[Tuple[str, Tuple[int, ...]]] = set()
    sft_rows: List[Dict[str, Any]] = []

    skip_no_chosen = 0
    skip_bad_fields = 0

    for row in dpo_rows:
        item_id = str(row.get("item_id", ""))
        sid = row.get("sid")
        context = row.get("context", "")
        chosen = (row.get("title_chosen") or "").strip()
        user_id = str(row.get("user_id", ""))

        if not item_id or not sid or not context:
            skip_bad_fields += 1
            continue
        if not chosen:
            skip_no_chosen += 1
            continue

        # 去重 key: (item_id, tuple(sid))
        try:
            sid_tuple = tuple(int(x) for x in sid)
        except (TypeError, ValueError):
            skip_bad_fields += 1
            continue
        key = (item_id, sid_tuple)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        sft_rows.append({
            "item_id": item_id,
            "sid": list(sid_tuple),
            "context": context,
            "target_title": chosen,
            "user_id": user_id,   # 同 (item, sid) 下首个见到的 user_id, 仅作 trace 参考, 训练不用
        })

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSFT data generated (target = title_chosen):")
    print(f"  Output: {args.out}")
    print(f"  Total rows: {len(sft_rows)}")
    print(f"  Skipped (missing title_chosen): {skip_no_chosen}")
    print(f"  Skipped (bad item_id/sid/context): {skip_bad_fields}")
    print(f"  Unique (item_id, sid) pairs: {len(seen_keys)}")
    if dpo_rows:
        print(f"  Dedup ratio: {len(sft_rows)} / {len(dpo_rows)} = "
              f"{len(sft_rows)/len(dpo_rows):.1%}")


if __name__ == "__main__":
    main()
