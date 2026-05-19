import argparse
import copy
import json
import os
import sys
from typing import Dict, List

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
    loss_history: List[Dict[str, float]] = []

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
            loss_val = loss.item()
            loss_history.append({"step": global_step, "loss": loss_val})
            if global_step % 10 == 0 or global_step >= args.max_steps:
                print(f"[dpo] step={global_step}/{args.max_steps} loss={loss_val:.4f}")
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    # Save loss history and plot
    loss_json_path = os.path.join(args.output_dir, "dpo_loss_history.json")
    with open(loss_json_path, "w", encoding="utf-8") as f:
        json.dump(loss_history, f)
    print(f"Loss history saved to: {loss_json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [h["step"] for h in loss_history]
        losses = [h["loss"] for h in loss_history]
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, linewidth=0.8, alpha=0.6, label="loss")
        # Smoothed curve (moving average)
        window = max(1, len(losses) // 50)
        if window > 1 and len(losses) > window:
            smoothed = [sum(losses[max(0, i - window):i + 1]) / min(i + 1, window + 1) for i in range(len(losses))]
            plt.plot(steps, smoothed, linewidth=2, color="red", label=f"smoothed (window={window})")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("DPO Training Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "dpo_loss_curve.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Loss curve saved to: {fig_path}")
    except ImportError:
        print("matplotlib not installed; skipping loss curve plot.")

    ckpt_path = os.path.join(args.output_dir, "sid_dpo.pt")
    if freeze:
        # Only save SID prefix weights
        torch.save({"sid_prefix": model.sid_prefix.state_dict()}, ckpt_path)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved (prefix only): {ckpt_path}")
    else:
        # Full finetune: save both backbone and SID prefix
        full_ckpt_dir = os.path.join(args.output_dir, "full_model")
        os.makedirs(full_ckpt_dir, exist_ok=True)
        torch.save(
            {
                "sid_prefix": model.sid_prefix.state_dict(),
                "full_model": model.state_dict(),
            },
            ckpt_path,
        )
        # Also save the backbone in HuggingFace format for easy loading/serving
        model.backbone.save_pretrained(full_ckpt_dir)
        tokenizer.save_pretrained(full_ckpt_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved (full model): {ckpt_path}")
        print(f"Saved HF format backbone: {full_ckpt_dir}")


if __name__ == "__main__":
    main()
