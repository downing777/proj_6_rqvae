#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 DPO 数据构造更"硬"的负例, 替换原始 title_rejected。

两种负例策略:
  1) cross_item: 从其它 item 的 chosen 标题里随机抽一条。主题/品类差异大,
     给模型一个"anything but this item"的简单对比信号, 防止 collapse.

  2) far_sid: 同一个 item 下, 找 SID 嵌入距离当前 SID 最远的另一个 SID,
     用它对应的 chosen 标题作为负例。SID 嵌入从 --sft-ckpt 里加载
     SidPrefixEncoder 的 sid_embeddings + sid_projector 算出。

输入:  原始 DPO jsonl (含 item_id, sid, context, title_chosen, ...)
输出:  新 DPO jsonl, title_rejected 被替换为新负例, 加 negative_type 字段。
       默认每条输入产生 2 条输出 (两种负例各一); 通过 --neg-type 控制。

Usage:
  python3 softprompt/data/build_dpo_negatives.py \\
    --dpo-jsonl train.jsonl \\
    --sft-ckpt /path/to/sid_sft.pt \\
    --out dpo_hard_neg.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from softprompt.models import SidPrefixConfig, SidPrefixEncoder


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_sid_embeddings(
    sft_ckpt: str,
    sid_dims: List[int],
    sid_embed_dim: int,
    num_virtual_tokens: int,
    num_basis_tokens: int,
    unique_sids: List[Tuple[int, ...]],
) -> Dict[Tuple[int, ...], torch.Tensor]:
    """加载 SFT ckpt 中的 SidPrefixEncoder, 给每个 unique SID 算 sid_hidden 表征。"""
    state = torch.load(sft_ckpt, map_location="cpu")
    sid_state = state["sid_prefix"]
    # hidden_size 从 token_basis 形状里反推, 免得让用户手动传 base model hidden_size
    hidden_size = sid_state["token_basis"].shape[1]
    cfg = SidPrefixConfig(
        sid_dims=sid_dims,
        sid_embed_dim=sid_embed_dim,
        num_virtual_tokens=num_virtual_tokens,
        hidden_size=hidden_size,
        num_basis_tokens=num_basis_tokens,
    )
    encoder = SidPrefixEncoder(cfg)
    encoder.load_state_dict(sid_state, strict=True)
    encoder.eval()

    sid_tensor = torch.tensor(unique_sids, dtype=torch.long)  # [N, L]
    with torch.no_grad():
        parts = [emb(sid_tensor[:, level]) for level, emb in enumerate(encoder.sid_embeddings)]
        sid_hidden = torch.cat(parts, dim=-1)
        sid_hidden = encoder.sid_projector(sid_hidden)  # [N, joined_dim]

    return {s: sid_hidden[i] for i, s in enumerate(unique_sids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build harder DPO negatives (cross-item / far-SID).")
    parser.add_argument("--dpo-jsonl", required=True, help="原始 DPO jsonl")
    parser.add_argument("--out", required=True, help="输出 jsonl 路径")
    parser.add_argument(
        "--sft-ckpt",
        default=None,
        help="SFT 检查点; 用其 sid_prefix 算 SID 嵌入。far_sid 模式必填。",
    )
    parser.add_argument("--sid-dims", default="32,32,32",
                        help="必须与 SFT 训练时一致")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument(
        "--neg-type",
        choices=["cross_item", "far_sid", "both"],
        default="both",
        help="cross_item: 跨 item 随机; far_sid: 同 item 最远 SID; both: 各产一条 (输出量 x2)",
    )
    parser.add_argument("--distance", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = load_jsonl(args.dpo_jsonl)
    print(f"Loaded {len(rows)} rows from {args.dpo_jsonl}")

    # 去重 (item, sid) -> chosen
    pair_chosen: Dict[Tuple[str, Tuple[int, ...]], str] = {}
    item_to_sids: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
    all_chosen: List[Tuple[str, str]] = []  # (item_id, chosen_title) for cross-item pool

    for r in rows:
        item = str(r["item_id"])
        try:
            sid_t = tuple(int(x) for x in r["sid"])
        except (TypeError, ValueError):
            continue
        chosen = (r.get("title_chosen") or "").strip()
        if not chosen:
            continue
        key = (item, sid_t)
        if key in pair_chosen:
            continue
        pair_chosen[key] = chosen
        item_to_sids[item].append(sid_t)
        all_chosen.append((item, chosen))

    print(f"Unique (item, sid) pairs: {len(pair_chosen)}")
    print(f"Items: {len(item_to_sids)}")

    # SID 嵌入 (far_sid 模式才需要)
    sid_emb: Dict[Tuple[int, ...], torch.Tensor] = {}
    if args.neg_type in ("far_sid", "both"):
        if not args.sft_ckpt:
            raise ValueError("--sft-ckpt is required for neg-type in {far_sid, both}.")
        unique_sids = sorted({s for sids in item_to_sids.values() for s in sids})
        sid_emb = build_sid_embeddings(
            sft_ckpt=args.sft_ckpt,
            sid_dims=parse_int_list(args.sid_dims),
            sid_embed_dim=args.sid_embed_dim,
            num_virtual_tokens=args.num_virtual_tokens,
            num_basis_tokens=args.num_basis_tokens,
            unique_sids=unique_sids,
        )
        print(f"Built SID embeddings for {len(unique_sids)} unique SIDs from {args.sft_ckpt}")

    def _distance(a: torch.Tensor, b: torch.Tensor) -> float:
        if args.distance == "l2":
            return torch.dist(a, b).item()
        # cosine distance = 1 - cos_sim
        return float(1.0 - torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    out_rows: List[Dict[str, Any]] = []
    skip_no_other_item = 0
    skip_no_other_sid = 0

    for r in rows:
        item = str(r["item_id"])
        try:
            sid_t = tuple(int(x) for x in r["sid"])
        except (TypeError, ValueError):
            continue
        chosen = (r.get("title_chosen") or "").strip()
        if not chosen:
            continue

        base = {k: v for k, v in r.items() if k not in ("title_rejected", "negative_type", "meta")}

        # (1) cross_item: random title from a different item
        if args.neg_type in ("cross_item", "both"):
            cand_title = None
            for _ in range(20):
                ci, ct = random.choice(all_chosen)
                if ci != item and ct != chosen:
                    cand_title = ct
                    break
            if cand_title is None:
                skip_no_other_item += 1
            else:
                out_rows.append({
                    **base,
                    "title_rejected": cand_title,
                    "negative_type": "cross_item",
                })

        # (2) far_sid: same item, SID with max embedding distance
        if args.neg_type in ("far_sid", "both"):
            sids_here = item_to_sids.get(item, [])
            others = [s for s in sids_here if s != sid_t]
            if not others:
                skip_no_other_sid += 1
            else:
                anchor = sid_emb[sid_t]
                far_dist, far_sid = max((_distance(anchor, sid_emb[s]), s) for s in others)
                far_title = pair_chosen[(item, far_sid)]
                if far_title == chosen:
                    # 同 item 多个 SID 共用了相同 chosen, 退化, 跳过
                    skip_no_other_sid += 1
                else:
                    out_rows.append({
                        **base,
                        "title_rejected": far_title,
                        "negative_type": "far_sid_in_item",
                        "meta": {"far_sid": list(far_sid), "far_dist": far_dist},
                    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_cross = sum(1 for r in out_rows if r["negative_type"] == "cross_item")
    n_far = sum(1 for r in out_rows if r["negative_type"] == "far_sid_in_item")
    print(f"\nWrote {len(out_rows)} rows to {args.out}")
    print(f"  cross_item:        {n_cross}")
    print(f"  far_sid_in_item:   {n_far}")
    print(f"  skipped (no cross-item candidate): {skip_no_other_item}")
    print(f"  skipped (no other sid in same item or dup chosen): {skip_no_other_sid}")


if __name__ == "__main__":
    main()
