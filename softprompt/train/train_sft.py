import argparse
import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from softprompt.models import SidModelLoadConfig, build_sid_model
from softprompt.train.common import (
    SFTJsonlDataset,
    build_prompt_target_tensors,
    collated_sid_to_tensor,
    load_jsonl,
    truncate_context_in_rows,
)


def parse_sid_dims(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT: SID 软前缀, 基座可冻结。")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="softprompt/outputs/sft")
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=30000)
    parser.add_argument("--max-context-chars", type=int, default=0)
    parser.add_argument("--max-steps", type=int, required=True, help="优化器步数, 到即停(数据可多轮扫)。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help="不冻结基座(显存/算力大); 默认只训软前缀。",
    )
    args = parser.parse_args()
    freeze = not args.no_freeze_backbone

    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    rows = truncate_context_in_rows(
        load_jsonl(args.train_jsonl), max_context_chars=args.max_context_chars
    )
    dataset = SFTJsonlDataset(rows)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

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
        device=str(device),
    )
    if freeze:
        model.freeze_backbone()
        print("Backbone frozen: SidPrefix only.")
    else:
        print("Backbone not frozen: full finetune + SidPrefix.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)

    model.train()
    global_step = 0
    _grad_clip = 1.0

    while global_step < args.max_steps:
        for batch in loader:
            if global_step >= args.max_steps:
                break
            prompts = batch["prompt"]
            targets = batch["target"]
            sid = collated_sid_to_tensor(batch["sid"], device=str(device))

            input_ids, attention_mask, labels = build_prompt_target_tensors(
                tokenizer=tokenizer,
                prompts=prompts,
                targets=targets,
                max_length=args.max_length,
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sid=sid,
                labels=labels,
            )
            loss = out.loss
            if not torch.isfinite(loss):
                print("[sft] skip non-finite loss (调 max-length / max-context-chars).")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if _grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, _grad_clip)
            optimizer.step()
            global_step += 1
            if global_step % 20 == 0 or global_step >= args.max_steps:
                print(f"[sft] step={global_step}/{args.max_steps} loss={loss.item():.4f}")
            if global_step >= args.max_steps:
                break

    ckpt_path = os.path.join(args.output_dir, "sid_sft.pt")
    torch.save({"sid_prefix": model.sid_prefix.state_dict()}, ckpt_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved: {ckpt_path}  {model.trainable_parameters_report()}")


if __name__ == "__main__":
    main()
