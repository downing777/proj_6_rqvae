import argparse
import json
import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from softprompt.models import SidModelLoadConfig, build_sid_model
from softprompt.train.common import load_jsonl, render_prompt, sequence_logp


def parse_sid_dims(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def distinct_n(texts: Sequence[str], n: int) -> float:
    all_ngrams = []
    for text in texts:
        chars = list(text.strip())
        if len(chars) < n:
            continue
        all_ngrams.extend(tuple(chars[i : i + n]) for i in range(len(chars) - n + 1))
    if not all_ngrams:
        return 0.0
    uniq = len(set(all_ngrams))
    return uniq / len(all_ngrams)


def keyword_recall(text: str, context: str) -> float:
    raw = context.replace(";", " ").replace(":", " ")
    candidates = [x.strip() for x in raw.split() if len(x.strip()) >= 2]
    keys = list(dict.fromkeys(candidates))[:8]
    if not keys:
        return 0.0
    hits = sum(1 for k in keys if k in text)
    return hits / len(keys)


def build_margin_gating(
    chosen_score: float, rejected_score: float, generated: str, fallback_title: str, min_margin: float
) -> Tuple[str, bool]:
    if (chosen_score - rejected_score) < min_margin:
        return fallback_title, True
    return generated, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval for SID title generation.")
    parser.add_argument("--dpo-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--sid-ckpt", type=str, required=True)
    parser.add_argument("--sid-dims", type=str, default="32,32,32")
    parser.add_argument("--sid-embed-dim", type=int, default=64)
    parser.add_argument("--num-virtual-tokens", type=int, default=16)
    parser.add_argument("--num-basis-tokens", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-margin", type=float, default=0.2)
    parser.add_argument("--output-json", type=str, default="softprompt/outputs/offline_eval.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dpo_rows = load_jsonl(args.dpo_jsonl)
    pred_rows = load_jsonl(args.pred_jsonl)
    pred_map = {(r["item_id"], tuple(r["sid"])): r for r in pred_rows}

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

    wins = 0
    total = 0
    recalls: List[float] = []
    generated_texts: List[str] = []
    fallback_count = 0
    sid_buckets = Counter()

    for row in dpo_rows:
        key = (row["item_id"], tuple(row["sid"]))
        pred = pred_map.get(key)
        if pred is None:
            continue
        sid = torch.tensor([row["sid"]], dtype=torch.long, device=args.device)
        prompt = render_prompt(row["context"])
        chosen_logp, _, _ = sequence_logp(
            model=model,
            tokenizer=tokenizer,
            sid=sid,
            prompts=[prompt],
            targets=[row["title_chosen"]],
            max_length=args.max_length,
        )
        rejected_logp, _, _ = sequence_logp(
            model=model,
            tokenizer=tokenizer,
            sid=sid,
            prompts=[prompt],
            targets=[row["title_rejected"]],
            max_length=args.max_length,
        )
        chosen_score = chosen_logp.item()
        rejected_score = rejected_logp.item()
        total += 1
        wins += int(chosen_score > rejected_score)

        final_title, fallback = build_margin_gating(
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            generated=pred["generated_text"],
            fallback_title=row["title_chosen"],
            min_margin=args.min_margin,
        )
        fallback_count += int(fallback)
        generated_texts.append(final_title)
        recalls.append(keyword_recall(final_title, row["context"]))
        sid_buckets[str(tuple(row["sid"]))] += 1

    result = {
        "pair_win_rate": (wins / total) if total > 0 else 0.0,
        "keyword_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "distinct_1": distinct_n(generated_texts, 1),
        "distinct_2": distinct_n(generated_texts, 2),
        "fallback_ratio": (fallback_count / total) if total > 0 else 0.0,
        "sid_bucket_count": len(sid_buckets),
        "sample_count": total,
        "confidence_interval_95": (
            1.96 * math.sqrt(max((wins / total) * (1 - wins / total), 1e-9) / max(total, 1))
            if total > 0
            else None
        ),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
