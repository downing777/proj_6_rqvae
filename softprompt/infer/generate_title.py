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

    # Suppress special tokens that should not appear in generated titles
    suppress_token_ids = []
    for token_str in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        suppress_token_ids.extend(ids)
    suppress_token_ids = list(set(suppress_token_ids)) if suppress_token_ids else None

    total = len(rows)
    print(f"Starting inference: {total} samples, max_new_tokens={args.max_new_tokens}, "
          f"max_input_length={args.max_input_length}, temperature={args.temperature}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, 1):
            sid = torch.tensor([row["sid"]], dtype=torch.long, device=args.device)
            prompt = render_prompt(row["context"])
            inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=args.max_input_length).to(args.device)
            input_len = inputs["input_ids"].shape[1]
            gen_kwargs = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sid=sid,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature,
            )
            if suppress_token_ids:
                gen_kwargs["suppress_tokens"] = suppress_token_ids
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
            if idx % 10 == 0 or idx == total:
                print(f"  [{idx}/{total}] item={row['item_id']} => {text[:60]}")

    print(f"Saved predictions to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
