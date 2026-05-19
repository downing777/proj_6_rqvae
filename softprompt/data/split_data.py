#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split DPO JSONL data into train/test sets.

划分策略: 按 (user_id, item_id) pair 级别随机划分。
同一个 user 或同一个 item 可以同时出现在 train 和 test 中，
但同一个 (user, item) 交互只属于其中一个集合。

Usage:
  python3 softprompt/data/split_data.py \
    --input-jsonl data/dpo_electronics_generated.jsonl \
    --output-dir data/split \
    --test-ratio 0.1 \
    --seed 42
"""

import argparse
import json
import os
import random
from collections import defaultdict
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

def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def make_user_item_key(row: Dict[str, Any]) -> str:
    """Generate a unique key for (user_id, item_id) pair."""
    user_id = str(row.get("user_id", ""))
    item_id = str(row.get("item_id", ""))
    return f"{user_id}::{item_id}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Split DPO data into train/test by (user_id, item_id) pair.")
    parser.add_argument("--input-jsonl", type=str, required=True, help="DPO 数据文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例 (按 unique (user, item) pair 数量)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(rows)} rows from {args.input_jsonl}")

    # Group rows by (user_id, item_id) pair
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = make_user_item_key(row)
        by_pair[key].append(row)

    pair_keys = sorted(by_pair.keys())
    print(f"Total unique (user_id, item_id) pairs: {len(pair_keys)}")

    # Shuffle and split by pair
    rng = random.Random(args.seed)
    rng.shuffle(pair_keys)

    num_test_pairs = max(1, int(len(pair_keys) * args.test_ratio))
    test_pair_keys: Set[str] = set(pair_keys[:num_test_pairs])
    train_pair_keys: Set[str] = set(pair_keys[num_test_pairs:])

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for key in train_pair_keys:
        train_rows.extend(by_pair[key])
    for key in test_pair_keys:
        test_rows.extend(by_pair[key])

    # Shuffle within each split
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    # Save outputs
    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    save_jsonl(train_rows, train_path)
    save_jsonl(test_rows, test_path)

    # Count unique users/items in each split
    train_users = set(str(r.get("user_id", "")) for r in train_rows)
    train_items = set(str(r.get("item_id", "")) for r in train_rows)
    test_users = set(str(r.get("user_id", "")) for r in test_rows)
    test_items = set(str(r.get("item_id", "")) for r in test_rows)

    print(f"\nSplit result:")
    print(f"  Train: {len(train_rows)} rows ({len(train_pair_keys)} pairs, "
          f"{len(train_users)} users, {len(train_items)} items) -> {train_path}")
    print(f"  Test:  {len(test_rows)} rows ({len(test_pair_keys)} pairs, "
          f"{len(test_users)} users, {len(test_items)} items) -> {test_path}")

    # Also save a test subset suitable for generate_title.py input
    # (deduplicate by (user_id, item_id), keep one row per unique pair)
    seen_keys: Set[str] = set()
    test_infer_rows: List[Dict[str, Any]] = []
    for row in test_rows:
        key = make_user_item_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        test_infer_rows.append({
            "user_id": row["user_id"],
            "item_id": row["item_id"],
            "sid": row["sid"],
            "context": row["context"],
            "title_chosen": row.get("title_chosen", ""),
        })

    test_infer_path = os.path.join(args.output_dir, "test_infer.jsonl")
    save_jsonl(test_infer_rows, test_infer_path)
    print(f"  Test (infer, dedup by user+item): {len(test_infer_rows)} rows -> {test_infer_path}")

if __name__ == "__main__":
    main()