#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick demo: 给定用户SID和商品信息，查看SFT/DPO模型分别生成什么标题。

Usage:
  python3 softprompt/infer/quick_demo.py \
    --base-model /path/to/Qwen3-8B \
    --sft-ckpt softprompt/outputs/sft/sid_sft.pt \
    --dpo-ckpt softprompt/outputs/dpo/sid_dpo.pt \
    --test-jsonl softprompt/outputs/split/test_infer.jsonl \
    --num-samples 10

会打印每个样本的:
  - 商品信息(context截断)
  - 目标SID
  - DPO训练集中的chosen标题(参考)
  - SFT模型生成的标题
  - DPO模型生成的标题
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from softprompt.models import SidModelLoadConfig, build_sid_model
from softprompt.train.common import render_prompt


def parse_sid_dims(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_title(model, tokenizer, row: Dict, device: str, max_new_tokens: int, temperature: float) -> str:
    sid = torch.tensor([row["sid"]], dtype=torch.long, device=device)
    prompt = render_prompt(row["context"])
    inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=256).to(device)
    generated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        sid=sid,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        temperature=temperature,
    )
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    # Strip the prompt prefix from generated text
    prompt_text = render_prompt(row["context"])
    if text.startswith(prompt_text):
        text = text[len(prompt_text):]
    # Clean up common prefixes
    for prefix in ["\n标题：", "标题：", "\n"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick demo: compare SFT vs DPO title generation.")
    parser.add_argument("--test-jsonl", type=str, required=True, help="测试集路径 (test_infer.jsonl)")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sft-ckpt", type=str, required=True)
    parser.add_argument("--dpo-ckpt", type=str, default="", help="DPO checkpoint; 留空则只展示SFT结果")
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-samples", type=int, default=10, help="展示样本数量")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_jsonl(args.test_jsonl)
    if args.num_samples > 0:
        rows = rows[:args.num_samples]

    print(f"Loaded {len(rows)} test samples")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = SidModelLoadConfig(
        base_model_name_or_path=args.base_model,
        sid_dims=tuple(parse_sid_dims(args.sid_dims)),
        sid_embed_dim=args.sid_embed_dim,
        num_virtual_tokens=args.num_virtual_tokens,
        num_basis_tokens=args.num_basis_tokens,
    )

    # Load SFT model
    print("Loading SFT model...")
    sft_model = build_sid_model(cfg, device=args.device)
    sft_state = torch.load(args.sft_ckpt, map_location=args.device)
    sft_model.sid_prefix.load_state_dict(sft_state["sid_prefix"], strict=True)
    sft_model.eval()

    # Load DPO model (if provided)
    dpo_model = None
    if args.dpo_ckpt and os.path.isfile(args.dpo_ckpt):
        print("Loading DPO model...")
        dpo_model = build_sid_model(cfg, device=args.device)
        dpo_state = torch.load(args.dpo_ckpt, map_location=args.device)
        dpo_model.sid_prefix.load_state_dict(dpo_state["sid_prefix"], strict=True)
        dpo_model.eval()

    # Generate and display results
    print("\n" + "=" * 80)
    print("  Title Generation Demo: SFT vs DPO")
    print("=" * 80)

    for i, row in enumerate(rows):
        context = row.get("context", "")
        context_preview = context[:200] + "..." if len(context) > 200 else context
        sid = row.get("sid", [])
        reference = row.get("title_chosen", "N/A")

        print(f"\n{'─' * 80}")
        print(f"  Sample {i + 1}/{len(rows)}")
        print(f"{'─' * 80}")
        print(f"  Item ID:   {row.get('item_id', 'N/A')}")
        print(f"  SID:       {sid}")
        print(f"  Context:   {context_preview}")
        print(f"  Reference: {reference}")

        with torch.no_grad():
            sft_title = generate_title(sft_model, tokenizer, row, args.device, args.max_new_tokens, args.temperature)
            print(f"  SFT生成:   {sft_title}")

            if dpo_model is not None:
                dpo_title = generate_title(dpo_model, tokenizer, row, args.device, args.max_new_tokens, args.temperature)
                print(f"  DPO生成:   {dpo_title}")

    print(f"\n{'=' * 80}")
    print(f"  Demo finished. Total samples: {len(rows)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
