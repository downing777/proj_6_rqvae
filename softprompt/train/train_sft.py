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
)


def parse_sid_dims(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for SID-conditioned title generation.")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="softprompt/outputs/sft")
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_jsonl(args.train_jsonl)
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
        device=args.device,
    )
    if args.freeze_backbone:
        model.freeze_backbone()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            prompts = batch["prompt"]
            targets = batch["target"]
            sid = collated_sid_to_tensor(batch["sid"], device=args.device)

            input_ids, attention_mask, labels = build_prompt_target_tensors(
                tokenizer=tokenizer,
                prompts=prompts,
                targets=targets,
                max_length=args.max_length,
            )
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sid=sid,
                labels=labels,
            )
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 20 == 0:
                print(f"[sft] epoch={epoch+1} step={global_step} loss={loss.item():.4f}")

    ckpt_path = os.path.join(args.output_dir, "sid_sft.pt")
    torch.save({"sid_prefix": model.sid_prefix.state_dict()}, ckpt_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved SFT SID adapter to: {ckpt_path}")
    print(f"Trainable params: {model.trainable_parameters_report()}")


if __name__ == "__main__":
    main()
