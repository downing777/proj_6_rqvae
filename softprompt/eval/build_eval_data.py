#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始 user-item review 数据中随机采样 user-item pairs，
通过 user_id 查 SID，构建 context，输出为可用于 inference 的 JSONL。

核心思路 (正确流程):
  1) 加载 user_semantic_ids.jsonl，得到 user_id -> SID 映射
  2) 流式扫描 reviews，收集有 SID 映射的 user-item pairs
  3) 随机选 N 个 user-item pairs
  4) 对选中的 pairs，加载对应 item metadata，聚合同 SID 用户对该 item 的评论证据
  5) 构建完整 context 并输出

输出格式: {item_id, user_id, sid, context, original_title}
- item_id: parent_asin
- user_id: 触发此样本的原始用户
- sid: [rqid_0, rqid_1, rqid_2] (该用户的 SID)
- context: 完整商品+评论上下文（与 dpo_title_gen.py 格式一致）
- original_title: 原商品英文标题（用于 LLM judge 对比）

Usage:
    python3 softprompt/eval/build_eval_data.py \
        --user-sid /path/to/user_semantic_ids.jsonl \
        --item-meta /path/to/final_filtered_item_meta_electronics.jsonl \
        --reviews /path/to/final_target_user_reviews_electronics.jsonl \
        --output /path/to/eval_samples.jsonl \
        --num-samples 200 \
        --seed 42
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set, Tuple

SidTuple = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

def load_user_sid_map(path: str) -> Dict[str, SidTuple]:
    """Load user_id_raw -> SID mapping from JSONL."""
    out: Dict[str, SidTuple] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = (rec.get("user_id_raw") or "").strip()
            if not uid:
                continue
            rid = rec.get("rqvae_id")
            if isinstance(rid, list) and len(rid) == 3:
                try:
                    sid = (int(rid[0]), int(rid[1]), int(rid[2]))
                except (TypeError, ValueError):
                    continue
                out[uid] = sid
    return out


def load_item_meta(path: str) -> Dict[str, Dict[str, Any]]:
    """Load item metadata indexed by parent_asin."""
    items: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            parent = row.get("parent_asin")
            if parent:
                items[str(parent)] = row
    return items


# --------------------------------------------------------------------------- #
#  Phase 1: Scan reviews to find valid user-item pairs
# --------------------------------------------------------------------------- #

def scan_user_item_pairs(
    reviews_path: str,
    user_to_sid: Dict[str, SidTuple],
    item_set: Set[str],
    max_scan_lines: int = 0,
) -> List[Tuple[str, str]]:
    """
    Scan reviews and collect (user_id, parent_asin) pairs where:
    - user has a SID mapping
    - item exists in item_meta
    Returns deduplicated list of (user_id, item_id) pairs.
    """
    seen_pairs: Set[Tuple[str, str]] = set()
    pairs: List[Tuple[str, str]] = []
    scanned = 0

    print(f"  Scanning reviews to find valid user-item pairs...")
    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_scan_lines > 0 and scanned >= max_scan_lines:
                break
            scanned += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("user_id", "")
            asin = rec.get("parent_asin", "")
            if not uid or not asin:
                continue
            if uid not in user_to_sid:
                continue
            if asin not in item_set:
                continue
            pair = (uid, asin)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                pairs.append(pair)

            if scanned % 1000000 == 0:
                print(f"    Scanned {scanned/1e6:.1f}M lines, found {len(pairs)} valid pairs...")

    print(f"  Scanned {scanned} lines total, found {len(pairs)} valid user-item pairs.")
    return pairs


# --------------------------------------------------------------------------- #
#  Phase 2: Collect reviews for selected items
# --------------------------------------------------------------------------- #

def collect_reviews_for_items(
    reviews_path: str,
    target_asins: Set[str],
    valid_users: Set[str],
) -> DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]]:
    """Stream reviews and index by asin -> user_id -> [reviews] for target items."""
    by_asin: DefaultDict[str, DefaultDict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("parent_asin")
            if not asin or asin not in target_asins:
                continue
            uid = rec.get("user_id")
            if not uid or uid not in valid_users:
                continue
            text = (rec.get("text") or "").replace("<br />", " ").replace("<br/>", " ")
            text = re.sub(r"\s+", " ", text).strip()
            by_asin[asin][str(uid)].append({
                "rating": rec.get("rating"),
                "text": text,
            })
    return by_asin


# --------------------------------------------------------------------------- #
#  Context building (same logic as dpo_title_gen.py)
# --------------------------------------------------------------------------- #

def clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def features_bullets(item: Dict[str, Any], max_bullets: int = 3) -> str:
    feats = item.get("features") or []
    if not isinstance(feats, list):
        return ""
    out: List[str] = []
    for f in feats[:max_bullets]:
        if isinstance(f, str) and f.strip():
            out.append(clip(f, 200))
    return " | ".join(out)


def aggregate_reviews_for_sid(
    reviews_by_user: Dict[str, List[Dict[str, Any]]],
    user_ids: Set[str],
    max_chars: int = 2000,
) -> str:
    """Aggregate review texts from users in the same SID group for a given item."""
    chunks: List[str] = []
    for uid in user_ids:
        for r in reviews_by_user.get(uid, []):
            t = (r.get("text") or "").strip()
            if t:
                rt = r.get("rating")
                prefix = f"[评分{rt}] " if isinstance(rt, (int, float)) else ""
                chunks.append(f"{prefix}{t}")

    def score(c: str) -> Tuple[float, int]:
        m = re.match(r"^\[评分([0-9.]+)\]", c)
        r = float(m.group(1)) if m else 0.0
        return (-r, -len(c))

    chunks.sort(key=score)
    out = " ".join(chunks)
    return clip(out, max_chars) if len(out) > max_chars else out


def build_context(
    item: Dict[str, Any],
    review_evidence: str,
    sid: SidTuple,
) -> str:
    main_cat = item.get("main_category") or ""
    store = item.get("store") or ""
    otitle = (item.get("title") or "")[:300]
    price = item.get("price")
    price_s = f"{price}" if price is not None else "未知"
    cats = item.get("categories")
    cat_s = " > ".join(str(c) for c in cats[:4]) if isinstance(cats, list) and cats else str(main_cat)
    parts = [
        f"站点: 美国站; 主类: {main_cat}",
        f"父 ASIN: {item.get('parent_asin', '')}; 品牌/店铺: {store}",
        f"原商品标题(英文): {otitle}",
        f"价格: {price_s}; 浏览类目: {cat_s}",
        f"商品要点(英文摘录): {features_bullets(item)}",
        f"目标 SID (rqid_0, rqid_1, rqid_2): {list(sid)}",
        f"该 SID 用户群在本商品上的评论证据(可有多条, 已截断): {review_evidence}",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval data by sampling user-item pairs from raw reviews.")
    parser.add_argument("--user-sid", type=str, required=True,
                        help="User SID mapping JSONL (user_semantic_ids.jsonl)")
    parser.add_argument("--item-meta", type=str, required=True,
                        help="Item metadata JSONL (final_filtered_item_meta_electronics.jsonl)")
    parser.add_argument("--reviews", type=str, required=True,
                        help="User reviews JSONL (final_target_user_reviews_electronics.jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path for eval samples")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of user-item pairs to sample for eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-review-chars", type=int, default=50,
                        help="Minimum review evidence length to include a sample")
    parser.add_argument("--max-review-chars", type=int, default=2000,
                        help="Maximum chars for review evidence per sample")
    parser.add_argument("--max-scan-lines", type=int, default=0,
                        help="Max lines to scan from reviews (0=all, for debugging)")
    args = parser.parse_args()

    random.seed(args.seed)

    # 1) Load user-SID mapping
    print(f"[1/5] Loading user-SID mapping from {args.user_sid}...")
    user_to_sid = load_user_sid_map(args.user_sid)
    print(f"  Loaded {len(user_to_sid)} users with SID.")

    # 2) Load item metadata
    print(f"[2/5] Loading item metadata from {args.item_meta}...")
    items = load_item_meta(args.item_meta)
    item_set = set(items.keys())
    print(f"  Loaded {len(items)} items.")

    # 3) Scan reviews to find valid (user, item) pairs
    print(f"[3/5] Scanning reviews for valid user-item pairs...")
    all_pairs = scan_user_item_pairs(
        args.reviews, user_to_sid, item_set, max_scan_lines=args.max_scan_lines
    )

    if not all_pairs:
        print("ERROR: No valid user-item pairs found!", file=sys.stderr)
        sys.exit(1)

    # 4) Randomly sample pairs
    sample_count = min(len(all_pairs), args.num_samples * 3)  # oversample for filtering
    sampled_pairs = random.sample(all_pairs, sample_count)
    print(f"  Sampled {len(sampled_pairs)} pairs (will filter to ~{args.num_samples}).")

    # Collect item ASINs we need reviews for
    target_asins = set(asin for _, asin in sampled_pairs)

    # 5) Second pass: collect all reviews for target items (from users with SID)
    print(f"[4/5] Collecting reviews for {len(target_asins)} target items...")
    reviews_by_asin = collect_reviews_for_items(
        args.reviews, target_asins, set(user_to_sid.keys())
    )
    print(f"  Collected reviews for {len(reviews_by_asin)} items.")

    # 6) Build eval samples
    print(f"[5/5] Building eval samples...")
    candidates: List[Dict[str, Any]] = []
    seen_item_sid: Set[Tuple[str, SidTuple]] = set()

    for user_id, item_id in sampled_pairs:
        if len(candidates) >= args.num_samples:
            break

        sid = user_to_sid[user_id]
        # Deduplicate by (item_id, sid) to avoid duplicate contexts
        key = (item_id, sid)
        if key in seen_item_sid:
            continue
        seen_item_sid.add(key)

        item = items.get(item_id)
        if item is None:
            continue

        asin_reviews = reviews_by_asin.get(item_id)
        if not asin_reviews:
            continue

        # Find all users with the same SID who reviewed this item
        same_sid_users: Set[str] = set()
        for uid in asin_reviews:
            if user_to_sid.get(uid) == sid:
                same_sid_users.add(uid)

        if not same_sid_users:
            continue

        # Aggregate review evidence from same-SID users
        review_evidence = aggregate_reviews_for_sid(
            asin_reviews, same_sid_users, args.max_review_chars
        )
        if len(review_evidence.strip()) < args.min_review_chars:
            continue

        context = build_context(item, review_evidence, sid)
        original_title = (item.get("title") or "")[:300]

        candidates.append({
            "item_id": item_id,
            "user_id": user_id,
            "sid": list(sid),
            "context": context,
            "original_title": original_title,
        })

    # 7) Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(candidates)} eval samples to: {args.output}")
    if len(candidates) < args.num_samples:
        print(f"  (Requested {args.num_samples}, got {len(candidates)} after filtering)")

    # Print some stats
    unique_users = set(r["user_id"] for r in candidates)
    unique_items = set(r["item_id"] for r in candidates)
    unique_sids = set(tuple(r["sid"]) for r in candidates)
    print(f"  Unique users: {len(unique_users)}")
    print(f"  Unique items: {len(unique_items)}")
    print(f"  Unique SIDs: {len(unique_sids)}")


if __name__ == "__main__":
    main()
