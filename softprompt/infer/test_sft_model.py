#!/usr/bin/env python3
"""
快速测试 SFT/DPO 模型能否正常输出标题。
直接在终端运行：
  cd /home/yuanhanyang.yhy/proj_6_rqvae
  conda activate softprompt
  CUDA_VISIBLE_DEVICES=0 python3 softprompt/infer/test_sft_model.py --ckpt sft
  CUDA_VISIBLE_DEVICES=0 python3 softprompt/infer/test_sft_model.py --ckpt dpo
"""
import argparse
import json
import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from softprompt.models import SidModelLoadConfig, build_sid_model
from softprompt.train.common import render_prompt, strip_thinking_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="sft", choices=["sft", "dpo"],
                        help="Which checkpoint to load: sft or dpo")
    parser.add_argument("--base-model", type=str,
                        default="/home/yuanhanyang.yhy/model_hub/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to test")
    parser.add_argument("--output", type=str, default="",
                        help="If set, write results to this JSONL file")
    args = parser.parse_args()

    ckpt_map = {
        "sft": "/home/yuanhanyang.yhy/project_6_outputs/sft/sid_sft.pt",
        "dpo": "/home/yuanhanyang.yhy/project_6_outputs/dpo/sid_dpo.pt",
    }
    ckpt_path = ckpt_map[args.ckpt]

    print(f"=== Testing {args.ckpt.upper()} model ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Base model: {args.base_model}")
    print()

    # Load model
    print("Loading model...")
    model = build_sid_model(
        SidModelLoadConfig(
            base_model_name_or_path=args.base_model,
            sid_dims=(32, 32, 32),
        ),
        device="cuda:0",
    )
    state = torch.load(ckpt_path, map_location="cuda:0")
    model.sid_prefix.load_state_dict(state["sid_prefix"], strict=True)
    model.eval()
    print("Model loaded!\n")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load real eval samples
    eval_path = "/home/yuanhanyang.yhy/project_6_outputs/eval_e2e/eval_samples_5000.jsonl"
    if not os.path.isfile(eval_path):
        eval_path = "/home/yuanhanyang.yhy/project_6_outputs/split/test_infer.jsonl"
    if not os.path.isfile(eval_path):
        print(f"ERROR: No eval data found!")
        return

    samples = []
    with open(eval_path) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            samples.append(json.loads(line.strip()))

    print(f"Testing with {len(samples)} real samples from: {eval_path}")
    print("=" * 60)

    results = []
    for i, row in enumerate(samples, 1):
        sid = torch.tensor([row["sid"]], dtype=torch.long, device="cuda:0")
        prompt = render_prompt(row["context"])
        inputs = tokenizer(
            [prompt], return_tensors="pt", truncation=True, max_length=args.max_input_length
        ).to("cuda:0")
        input_len = inputs["input_ids"].shape[1]

        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            sid=sid,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        new_tokens = generated[:, input_len:]
        text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        text = strip_thinking_tags(text)

        # Extract original title from context
        original_title = ""
        for ctx_line in row["context"].split("\n"):
            if "原商品标题" in ctx_line:
                original_title = ctx_line.split(": ", 1)[-1] if ": " in ctx_line else ctx_line
                break

        result = {
            "item_id": row["item_id"],
            "sid": row["sid"],
            "original_title": original_title,
            "generated_text": text,
            "input_tokens": input_len,
            "new_tokens": new_tokens.shape[1],
            "context": row["context"],
        }
        results.append(result)

        if i % 10 == 0 or i <= 5:
            print(f"  [{i}/{len(samples)}] item={row['item_id']} sid={row['sid']}")
            print(f"    原标题: {original_title[:60]}")
            print(f"    生成: {text[:80]}")
            print()

    # Write to file if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(results)} results to: {args.output}")
    else:
        # Print all to stdout
        print("\n" + "=" * 60)
        for i, r in enumerate(results, 1):
            print(f"{i:3d}. [{r['sid']}] {r['original_title'][:50]}")
            print(f"     => {r['generated_text'][:80]}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
