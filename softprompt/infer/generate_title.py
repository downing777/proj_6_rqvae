import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from softprompt.models import SidModelLoadConfig, build_sid_model
from softprompt.train.common import render_prompt, strip_thinking_tags


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one best title for each (item, sid).")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sid-ckpt", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default="softprompt/outputs/predictions.jsonl")
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-input-length", type=int, default=2048,
                        help="Max tokenizer input length (truncate context if exceeds)")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Randomly sample N rows for inference (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import random
    rows = load_jsonl(args.input_jsonl)
    if args.max_samples is not None and args.max_samples < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.max_samples)
        print(f"Sampled {len(rows)} rows from {args.input_jsonl}")
    else:
        print(f"Using all {len(rows)} rows from {args.input_jsonl}")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_sid_model(
        SidModelLoadConfig(
            base_model_name_or_path=args.base_model,
            sid_dims=tuple(parse_sid_dims(args.sid_dims)),
            sid_embed_dim=args.sid_embed_dim,
            num_virtual_tokens=args.num_virtual_tokens,
            num_basis_tokens=args.num_basis_tokens,
        ),
        device=args.device,
    )
    state = torch.load(args.sid_ckpt, map_location=args.device)
    model.sid_prefix.load_state_dict(state["sid_prefix"], strict=True)
    model.eval()

    # 注: 不再做 suppress_tokens —
    #   1) EOS 大概率被 suppress 列表撞到 (Qwen 系 <|endoftext|> 就是 EOS),
    #      撞到后模型永远停不下来, 只能写满 max_new_tokens;
    #   2) 解码端的 skip_special_tokens=True 已经能把 special token 从字符串里剔除,
    #      不需要在 logits 层再压一道。

    total = len(rows)
    print(f"Starting inference: {total} samples, max_new_tokens={args.max_new_tokens}, "
          f"max_input_length={args.max_input_length}, temperature={args.temperature}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        pbar = tqdm(
            enumerate(rows, 1),
            total=total,
            desc="generate",
            mininterval=2.0,
            dynamic_ncols=True,
        )
        for idx, row in pbar:
            sid = torch.tensor([row["sid"]], dtype=torch.long, device=args.device)
            # 注: 必须与训练时 prompt_only 的拼接保持一致 (common.py:build_prompt_target_tensors),
            # prompt 截止在 ":" (无尾空格)。
            # 原因: Qwen BPE 把 "Title: BUFFALO" 的空格和 ' BUFFALO' 合并成单 token,
            # 但 "Title: " (末尾孤独空格) 会被编成独立 token (id=220), 训练里这种位置
            # 模型一次都没见过 → 推理 OOD, top-1 容易跳到 EOS 上, 标题被一刀切空。
            # 训练 texts 里的 "Title: " 空格不动 (它会和 target 首词合并被监督)。
            prompt = render_prompt(row["context"]) + "\nTitle:"
            inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=args.max_input_length).to(args.device)
            input_len = inputs["input_ids"].shape[1]
            gen_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sid=sid,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature,
                # 关键: 必须用 tokenizer 的 EOS, 跟训练 common.py:106 拼的 `tokenizer.eos_token`
                # 对齐。Qwen 系 model.config.eos_token_id 常常和 tokenizer.eos_token_id 不一致
                # (base EOS vs chat EOS), 用错了 break 永远不触发, 模型只能写满 max_new_tokens。
                eos_token_id=tokenizer.eos_token_id,
            )
            generated = model.generate(**gen_kwargs)
            # Only decode newly generated tokens (exclude the input prompt)
            new_tokens = generated[:, input_len:]
            text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            text = strip_thinking_tags(text)
            f.write(
                json.dumps(
                    {
                        "item_id": row["item_id"],
                        "sid": row["sid"],
                        "context": row["context"],
                        "generated_text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pbar.set_postfix_str(f"{text[:40]}", refresh=False)
            if idx % 10 == 0 or idx == total:
                tqdm.write(f"  [{idx}/{total}] item={row['item_id']} => {text[:60]}")
        pbar.close()

    print(f"Saved predictions to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
