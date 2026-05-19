#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build SFT training data from DPO JSONL + item metadata.

SFT 目标：给定 SID softprompt + 商品基本信息(context)，让模型学会输出商品的原标题。
这让模型先掌握"理解商品信息并生成标题"的基础能力，后续 DPO 再学习个性化。

输入:
  --dpo-jsonl: DPO 数据(提供 user_id, item_id, sid, context)
  --item-jsonl: 商品元数据(提供 parent_asin -> title 的映射)

输出:
  每行: {"user_id": "...", "sid": [x,y,z], "context": "...", "target_title": "<商品原标题>", "item_id": "..."}

去重: 同一个 (user_id, item_id) 只保留一条。

Usage:
  python3 softprompt/data/build_sft_from_dpo.py \
    --dpo-jsonl data/split/train.jsonl \
    --item-jsonl /path/to/item_meta.jsonl \
    --out /path/to/sft_from_item_title.jsonl
"""

import argparse
import json
import os
from typing import Any, Dict, List, Set

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_item_titles(path: str) -> Dict[str, str]:
    """Load item_id -> original title mapping from item metadata."""
    titles: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parent_asin = row.get("parent_asin", "")
            title = (row.get("title") or "").strip()
            if parent_asin and title:
                titles[str(parent_asin)] = title
    return titles

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SFT data: SID + context -> original item title."
    )
    parser.add_argument("--dpo-jsonl", type=str, required=True, help="DPO 数据文件")
    parser.add_argument("--item-jsonl", type=str, required=True, help="商品元数据文件")
    parser.add_argument("--out", type=str, required=True, help="输出 SFT JSONL 路径")
    args = parser.parse_args()

    # Load item original titles
    print(f"Loading item titles from {args.item_jsonl}...")
    item_titles = load_item_titles(args.item_jsonl)
    print(f"Loaded {len(item_titles)} item titles.")

    # Load DPO data
    dpo_rows = load_jsonl(args.dpo_jsonl)
    print(f"Loaded {len(dpo_rows)} DPO rows from {args.dpo_jsonl}.")

    # Build SFT data: deduplicate by (user_id, item_id)
    seen_keys: Set[str] = set()
    sft_rows: List[Dict[str, Any]] = []

    missing_title_count = 0
    for row in dpo_rows:
        user_id = str(row.get("user_id", ""))
        item_id = str(row.get("item_id", ""))
        sid = row.get("sid")
        context = row.get("context", "")

        if not user_id or not item_id or not sid or not context:
            continue

        # Deduplicate by (user_id, item_id)
        key = f"{user_id}::{item_id}"
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Get original title from item metadata
        original_title = item_titles.get(item_id, "")
        if not original_title:
            missing_title_count += 1
            continue

        sft_rows.append({
            "user_id": user_id,
            "sid": sid,
            "context": context,
            "target_title": original_title,
            "item_id": item_id,
        })

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSFT data generated:")
    print(f"  Output: {args.out}")
    print(f"  Total rows: {len(sft_rows)}")
    print(f"  Skipped (missing title): {missing_title_count}")
    print(f"  Unique (user_id, item_id) pairs: {len(seen_keys)}")

if __name__ == "__main__":
    main()