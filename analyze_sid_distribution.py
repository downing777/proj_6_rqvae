"""Analyze SID distribution from RQ-VAE training output."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Analyze user semantic ID distribution")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/exp_mi_cb32_ed128_w1p0_a1p0_b1p0_tau0p5_k16_s42/user_semantic_ids.jsonl",
        help="Path to user_semantic_ids.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Show top-k most frequent SIDs")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    # Load all SIDs
    sid_tuples = []
    per_layer_ids = [[], [], []]
    with open(input_path, "r") as f:
        for line in f:
            record = json.loads(line)
            rqvae_id = record["rqvae_id"]
            sid_tuples.append(tuple(rqvae_id))
            for layer_idx, code_id in enumerate(rqvae_id):
                per_layer_ids[layer_idx].append(code_id)

    total_users = len(sid_tuples)
    unique_sids = len(set(sid_tuples))

    print("=" * 60)
    print(f"  SID Distribution Analysis")
    print(f"  Input: {input_path}")
    print("=" * 60)
    print(f"\n  Total users:      {total_users:,}")
    print(f"  Unique SIDs:      {unique_sids:,}")
    print(f"  Theoretical max:  {32**3:,} (32^3)")
    print(f"  Utilization:      {unique_sids / 32**3 * 100:.2f}%")
    print()

    # Per-layer codebook utilization
    print("--- Per-Layer Codebook Utilization ---")
    for layer_idx, ids in enumerate(per_layer_ids):
        unique_codes = len(set(ids))
        counter = Counter(ids)
        most_common = counter.most_common(5)
        least_common = counter.most_common()[-3:]
        print(f"  Layer {layer_idx}: {unique_codes}/32 codes active")
        top_str = ", ".join(f"code{c}={n}" for c, n in most_common)
        print(f"    Top-5:    {top_str}")
        bot_str = ", ".join(f"code{c}={n}" for c, n in least_common)
        print(f"    Bottom-3: {bot_str}")
    print()

    # SID frequency distribution
    sid_counter = Counter(sid_tuples)
    print(f"--- Top-{args.top_k} Most Frequent SIDs ---")
    print(f"  {'SID':<20} {'Count':>8} {'Percent':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for sid, count in sid_counter.most_common(args.top_k):
        pct = count / total_users * 100
        print(f"  {str(sid):<20} {count:>8,} {pct:>7.2f}%")
    print()

    # Distribution summary
    counts = sorted(sid_counter.values(), reverse=True)
    print("--- Distribution Summary ---")
    print(f"  Max users per SID:     {counts[0]:,}")
    print(f"  Min users per SID:     {counts[-1]:,}")
    print(f"  Median users per SID:  {counts[len(counts)//2]:,}")
    print(f"  Mean users per SID:    {total_users / unique_sids:,.1f}")

    # Entropy
    import math
    entropy = 0.0
    for count in counts:
        p = count / total_users
        entropy -= p * math.log2(p)
    max_entropy = math.log2(unique_sids) if unique_sids > 1 else 1.0
    print(f"\n  SID entropy:           {entropy:.2f} bits")
    print(f"  Max possible entropy:  {max_entropy:.2f} bits (uniform over {unique_sids} SIDs)")
    print(f"  Normalized entropy:    {entropy / max_entropy:.3f} (1.0 = perfectly uniform)")
    print()

    # Bucket distribution
    print("--- SID Size Buckets ---")
    buckets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 500), (501, 1000), (1001, float("inf"))]
    for lo, hi in buckets:
        in_bucket = sum(1 for c in counts if lo <= c <= hi)
        if in_bucket > 0:
            hi_str = f"{int(hi)}" if hi != float("inf") else "∞"
            print(f"  [{lo:>5} - {hi_str:>5}] users: {in_bucket:>5} SIDs")


if __name__ == "__main__":
    main()
