#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接读取已有 DPO 数据, 重写 title_rejected, 产出 1:1 的新 DPO 数据集。

规则:
  - 正例 (title_chosen)   保持不变, 仍是原 chosen 标题
  - 负例 (title_rejected) 重置为: 同一 item 下, SID 嵌入距离当前 SID 最远的
    另一个 SID 所对应的 title_chosen
  - 严格 1:1: 一条输入 → 一条输出 (找不到合法负例的行会被跳过, 写入统计)

SID 嵌入来源 (RQ-VAE):
  - --user-npz:       user.npz 里 'user_embeddings' [N_users, D] 矩阵
  - --user-sid-jsonl: 每行 {user_id_raw, user_id(int), rqvae_id(SID)}
  - --user-ids-json:  (可选) 给出 npz 行序对应的 user_id_raw 列表
                      不传则默认 jsonl 里 user_id == npz 行索引

每个 unique SID 的嵌入 = 该 SID 下所有用户 embedding 的均值。

输出格式:
  和输入一致, 只是 title_rejected 被替换; meta 里加 {far_sid, far_dist,
  neg_source} 方便溯源; schema 不变, 直接喂给 train_dpo.py。

Usage:
  python3 softprompt/data/build_dpo_far_sid_1to1.py \\
    --input-jsonl    /path/to/old_train.jsonl \\
    --user-npz       /home/.../amazon_user_item_dataset.user.npz \\
    --user-sid-jsonl /home/.../user_semantic_ids.jsonl \\
    --out            /path/to/new_train.jsonl
  # 如果 user_id 整数列和 npz 行序不一致, 再加 --user-ids-json /path/to/.user_ids.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_user_id_order(path: Optional[str]) -> Optional[List[str]]:
    """读 user_ids.json, 返回 npz 行序对应的 user_id_raw 列表 (或 None 表示文件没给)。

    尝试兼容两种结构:
      a) 直接是 list[str]
      b) dict 包了一层, 常见 key: users / user_ids / order
    """
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("users", "user_ids", "order", "user_id_raw"):
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]
    raise ValueError(
        f"无法解析 {path}: 期待 list[str] 或 dict[users/user_ids/order/user_id_raw=list]。"
    )


def build_sid_embeddings(
    user_npz_path: str,
    user_sid_jsonl_path: str,
    user_ids_json_path: Optional[str],
    needed_sids: Optional[set] = None,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """聚合每个 SID 的用户 embedding 均值, 返回 dict[SID tuple] -> ndarray[D]。

    needed_sids: 若提供, 只算这些 SID (省内存); None 表示算全部。
    """
    # 1) 加载 npz
    npz = np.load(user_npz_path)
    if "user_embeddings" not in npz:
        raise KeyError(f"{user_npz_path} 里没有 user_embeddings 数组, "
                       f"现有 keys={list(npz.keys())}")
    emb = npz["user_embeddings"]  # 通常 float16
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)  # 算均值/距离用 float32 更稳
    n_users, dim = emb.shape
    print(f"  user_embeddings: shape={emb.shape}, dtype={emb.dtype}")

    # 2) 加载 jsonl: user_id_raw -> SID, user_id (int) -> SID
    raw_to_sid: Dict[str, Tuple[int, ...]] = {}
    int_to_sid: Dict[int, Tuple[int, ...]] = {}
    skip_jsonl = 0
    for r in load_jsonl(user_sid_jsonl_path):
        try:
            sid = tuple(int(x) for x in r["rqvae_id"])
        except (TypeError, ValueError, KeyError):
            skip_jsonl += 1
            continue
        raw = r.get("user_id_raw")
        if raw:
            raw_to_sid[str(raw)] = sid
        uid = r.get("user_id")
        if isinstance(uid, int):
            int_to_sid[uid] = sid
    print(f"  semantic_ids jsonl: raw_to_sid={len(raw_to_sid)}, "
          f"int_to_sid={len(int_to_sid)}, skipped_lines={skip_jsonl}")

    # 3) 建立 npz 行 -> SID 的映射
    sid_to_indices: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    if user_ids_json_path:
        order = load_user_id_order(user_ids_json_path)
        assert order is not None
        if len(order) != n_users:
            raise ValueError(
                f"user_ids.json 长度 {len(order)} 与 npz 行数 {n_users} 不一致"
            )
        for i, raw in enumerate(order):
            sid = raw_to_sid.get(raw)
            if sid is None:
                continue
            sid_to_indices[sid].append(i)
        print(f"  alignment: via user_ids.json (raw_id order)")
    else:
        # 默认假设: jsonl 里的 user_id (int) 就是 npz 行索引
        for uid, sid in int_to_sid.items():
            if 0 <= uid < n_users:
                sid_to_indices[sid].append(uid)
        print(f"  alignment: default (jsonl user_id == npz row index)")

    print(f"  unique SIDs found: {len(sid_to_indices)}")

    # 4) 按需聚合
    out: Dict[Tuple[int, ...], np.ndarray] = {}
    iter_sids = sid_to_indices.items()
    if needed_sids is not None:
        iter_sids = [(s, sid_to_indices[s]) for s in needed_sids if s in sid_to_indices]
    for sid, indices in iter_sids:
        if not indices:
            continue
        out[sid] = emb[indices].mean(axis=0)
    print(f"  SID embeddings aggregated: {len(out)} "
          f"(needed={'all' if needed_sids is None else len(needed_sids)})")
    return out


def _distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    if metric == "cosine":
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    raise ValueError(f"unknown distance metric: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite DPO data with far-SID negatives (1:1).")
    parser.add_argument("--input-jsonl", required=True, help="已有 DPO 数据 (如 split/train.jsonl)")
    parser.add_argument("--out", required=True, help="输出新 DPO 数据路径")
    parser.add_argument("--user-npz", required=True,
                        help="user.npz, 含 'user_embeddings' [N_users, D]")
    parser.add_argument("--user-sid-jsonl", required=True,
                        help="user_semantic_ids.jsonl, 每行 user_id_raw/user_id/rqvae_id")
    parser.add_argument("--user-ids-json", default=None,
                        help="(可选) user_ids.json, npz 行序对应的 user_id_raw 列表; "
                             "不传则默认 jsonl 里 user_id(int) 就是 npz 行索引")
    parser.add_argument("--distance", choices=["l2", "cosine"], default="l2")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(rows)} rows from {args.input_jsonl}")

    # 1) 索引: (item, sid) -> chosen; item -> {sid}
    pair_chosen: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    item_to_sids: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
    for r in rows:
        item = str(r.get("item_id", ""))
        try:
            sid = tuple(int(x) for x in r["sid"])
        except (TypeError, ValueError, KeyError):
            continue
        chosen = (r.get("title_chosen") or "").strip()
        if not item or not chosen:
            continue
        key = (item, sid)
        if key in pair_chosen:
            continue
        pair_chosen[key] = chosen
        item_to_sids[item].append(sid)

    print(f"Unique (item, sid) pairs: {len(pair_chosen)}")
    print(f"Items: {len(item_to_sids)}")

    # 2) 只为出现过的 SID 聚合嵌入 (省时省内存)
    needed_sids = {s for sids in item_to_sids.values() for s in sids}
    print(f"\nBuilding SID embeddings from user data...")
    sid_emb = build_sid_embeddings(
        user_npz_path=args.user_npz,
        user_sid_jsonl_path=args.user_sid_jsonl,
        user_ids_json_path=args.user_ids_json,
        needed_sids=needed_sids,
    )

    missing = needed_sids - set(sid_emb.keys())
    if missing:
        print(f"WARN: {len(missing)}/{len(needed_sids)} SID 在 user 数据里没找到 → "
              f"这些 SID 的行将退化为 skip 处理。")

    # 3) 逐行重写 title_rejected
    out_rows: List[Dict[str, Any]] = []
    skip_no_other_sid = 0
    skip_no_emb = 0
    skip_dup_chosen = 0
    skip_bad_fields = 0

    for r in rows:
        item = str(r.get("item_id", ""))
        try:
            sid = tuple(int(x) for x in r["sid"])
        except (TypeError, ValueError, KeyError):
            skip_bad_fields += 1
            continue
        chosen = (r.get("title_chosen") or "").strip()
        if not item or not chosen:
            skip_bad_fields += 1
            continue

        others = [s for s in item_to_sids[item] if s != sid]
        if not others:
            skip_no_other_sid += 1
            continue

        anchor = sid_emb.get(sid)
        if anchor is None:
            skip_no_emb += 1
            continue
        # 只在两端都有 embedding 的候选里挑
        candidates = [s for s in others if s in sid_emb]
        if not candidates:
            skip_no_emb += 1
            continue

        far_dist, far_sid = max((_distance(anchor, sid_emb[s], args.distance), s) for s in candidates)
        far_chosen = pair_chosen[(item, far_sid)]
        if far_chosen == chosen:
            skip_dup_chosen += 1
            continue

        new_row = dict(r)
        new_row["title_rejected"] = far_chosen
        new_row["negative_type"] = "far_sid_in_item"
        prev_meta = new_row.get("meta") if isinstance(new_row.get("meta"), dict) else {}
        new_row["meta"] = {
            **prev_meta,
            "far_sid": list(far_sid),
            "far_dist": far_dist,
            "neg_source": "build_dpo_far_sid_1to1(rqvae_user_emb)",
        }
        out_rows.append(new_row)

    # 4) 写出
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_rows)} rows to {args.out}")
    print(f"  input rows:            {len(rows)}")
    print(f"  output (1:1) rows:     {len(out_rows)}")
    print(f"  skipped no-other-sid:  {skip_no_other_sid}  (同 item 下只有 1 个 SID)")
    print(f"  skipped no-embedding:  {skip_no_emb}    (锚 SID 或所有候选 SID 在 user 数据里没出现)")
    print(f"  skipped dup-chosen:    {skip_dup_chosen}    (最远 SID 的 chosen 和正例字面相同)")
    print(f"  skipped bad-fields:    {skip_bad_fields}")
    print(f"  distance metric:       {args.distance}")


if __name__ == "__main__":
    main()
