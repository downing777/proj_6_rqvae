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
    truncate_context_in_rows,
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
    parser = argparse.ArgumentParser(description="DPO: 只更新 SID 软前缀(默认冻结基座)。")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sft-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="softprompt/outputs/dpo")
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=30000)
    parser.add_argument("--max-context-chars", type=int, default=0)
    parser.add_argument("--max-steps", type=int, required=True, help="优化器步数, 到即停(数据可多轮扫)。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help="不冻结基座(显存大); 默认只训软前缀。",
    )
    args = parser.parse_args()
    freeze = not args.no_freeze_backbone

    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    rows = truncate_context_in_rows(
        load_jsonl(args.train_jsonl), max_context_chars=args.max_context_chars
    )
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
    model = build_sid_model(cfg, device=str(device))
    sft_state = torch.load(args.sft_ckpt, map_location="cpu")
    model.sid_prefix.load_state_dict(sft_state["sid_prefix"], strict=True)
    if freeze:
        model.freeze_backbone()
        print("Backbone frozen: DPO on SidPrefix; ref 同步。")
    else:
        print("DPO: policy+ref 全参(显存大)。")
    model.train()

    ref_model = build_sid_model(cfg, device=str(device))
    ref_model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01)
    _grad_clip = 1.0

    global_step = 0
    while global_step < args.max_steps:
        for batch in loader:
            if global_step >= args.max_steps:
                break
            sid = collated_sid_to_tensor(batch["sid"], device=str(device))
            prompts = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            pi_chosen, _, _ = sequence_logp(
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
                beta=0.1,
            )
            if not torch.isfinite(loss):
                print("[dpo] skip non-finite loss (调 max-length / max-context-chars).")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if _grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, _grad_clip)
            optimizer.step()
            global_step += 1
            if global_step % 10 == 0 or global_step >= args.max_steps:
                print(f"[dpo] step={global_step}/{args.max_steps} loss={loss.item():.4f}")
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    ckpt_path = os.path.join(args.output_dir, "sid_dpo.pt")
    torch.save({"sid_prefix": model.sid_prefix.state_dict()}, ckpt_path)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
