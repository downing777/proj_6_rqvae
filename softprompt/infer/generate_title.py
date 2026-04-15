import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoTokenizer

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
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

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

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            sid = torch.tensor([row["sid"]], dtype=torch.long, device=args.device)
            prompt = render_prompt(row["context"])
            inputs = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=256).to(args.device)
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                sid=sid,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                temperature=args.temperature,
            )
            text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
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

    print(f"Saved predictions to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
