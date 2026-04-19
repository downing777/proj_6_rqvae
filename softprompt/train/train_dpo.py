import argparse
import copy
import os
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from softprompt.models import SidModelLoadConfig, build_sid_model
from softprompt.train.common import (
    DPOJsonlDataset,
    collated_sid_to_tensor,
    load_jsonl,
    sequence_logp,
)


def parse_sid_dims(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def dpo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    pi_logratio = pi_chosen - pi_rejected
    ref_logratio = ref_chosen - ref_rejected
    logits = beta * (pi_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training for SID-conditioned title generation.")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sft-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="softprompt/outputs/dpo")
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl-to-nosid-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_jsonl(args.train_jsonl)
    dataset = DPOJsonlDataset(rows)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

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
    model = build_sid_model(cfg, device=args.device)
    sft_state = torch.load(args.sft_ckpt, map_location=args.device)
    model.sid_prefix.load_state_dict(sft_state["sid_prefix"], strict=True)
    if args.freeze_backbone:
        model.freeze_backbone()
    model.train()

    ref_model = build_sid_model(cfg, device=args.device)
    ref_model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            sid = collated_sid_to_tensor(batch["sid"], device=args.device)
            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            pi_chosen, out_chosen, chosen_pack = sequence_logp(
                model=model,
                tokenizer=tokenizer,
                sid=sid,
                prompts=prompts,
                targets=chosen,
                max_length=args.max_length,
            )
            pi_rejected, _, _ = sequence_logp(
                model=model,
                tokenizer=tokenizer,
                sid=sid,
                prompts=prompts,
                targets=rejected,
                max_length=args.max_length,
            )
            with torch.no_grad():
                ref_chosen, _, _ = sequence_logp(
                    model=ref_model,
                    tokenizer=tokenizer,
                    sid=sid,
                    prompts=prompts,
                    targets=chosen,
                    max_length=args.max_length,
                )
                ref_rejected, _, _ = sequence_logp(
                    model=ref_model,
                    tokenizer=tokenizer,
                    sid=sid,
                    prompts=prompts,
                    targets=rejected,
                    max_length=args.max_length,
                )

            loss = dpo_loss(
                pi_chosen=pi_chosen,
                pi_rejected=pi_rejected,
                ref_chosen=ref_chosen,
                ref_rejected=ref_rejected,
                beta=args.beta,
            )

            if args.kl_to_nosid_weight > 0:
                zero_sid = torch.zeros_like(sid)
                no_sid_out = model(
                    input_ids=chosen_pack["input_ids"],
                    attention_mask=chosen_pack["attention_mask"],
                    sid=zero_sid,
                )
                sid_logits = out_chosen.logits
                no_sid_logits = no_sid_out.logits.detach()
                kl = F.kl_div(
                    sid_logits.log_softmax(-1),
                    no_sid_logits.softmax(-1),
                    reduction="batchmean",
                )
                loss = loss + args.kl_to_nosid_weight * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % 10 == 0:
                print(f"[dpo] epoch={epoch+1} step={global_step} loss={loss.item():.4f}")

    ckpt_path = os.path.join(args.output_dir, "sid_dpo.pt")
    torch.save({"sid_prefix": model.sid_prefix.state_dict()}, ckpt_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DPO SID adapter to: {ckpt_path}")


if __name__ == "__main__":
    main()
