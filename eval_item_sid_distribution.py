import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


def _parse_experiment_arg(raw: str) -> Tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --exp format: {raw}. Expected name=path")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid --exp format: {raw}. Expected non-empty name and path")
    return name, path


def _resolve_sid_csv(path: str) -> str:
    if os.path.isdir(path):
        csv_path = os.path.join(path, "user_semantic_ids.csv")
    else:
        csv_path = path
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"user_semantic_ids.csv not found: {csv_path}")
    return csv_path


def _build_sid_string(row: Dict[str, str], sid_mode: str) -> str:
    if sid_mode == "full":
        return f"{row['rqid_0']}-{row['rqid_1']}-{row['rqid_2']}"
    if sid_mode == "rqid_0":
        return row["rqid_0"]
    if sid_mode == "rqid_1":
        return row["rqid_1"]
    if sid_mode == "rqid_2":
        return row["rqid_2"]
    raise ValueError(f"Unsupported sid_mode: {sid_mode}")


def _load_user_sid_map(csv_path: str, sid_mode: str) -> Dict[str, str]:
    user_to_sid: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"user_id_raw", "rqid_0", "rqid_1", "rqid_2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
        for row in reader:
            uid = row["user_id_raw"]
            if uid:
                user_to_sid[uid] = _build_sid_string(row, sid_mode=sid_mode)
    if not user_to_sid:
        raise ValueError(f"No user SID rows loaded from {csv_path}")
    return user_to_sid


def _entropy_from_counts(counts: Iterable[int]) -> float:
    counts = [c for c in counts if c > 0]
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def _quantiles(values: Sequence[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    def at(q: float) -> float:
        idx = int(round((n - 1) * q))
        return float(s[idx])
    return at(0.25), at(0.5), at(0.75)


def _safe_mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _evaluate_experiment(
    exp_name: str,
    user_to_sid: Dict[str, str],
    item_to_users: Dict[str, List[str]],
    min_users_per_item: int,
    per_item_csv_path: str,
) -> Dict[str, float]:
    kept_item_count = 0
    dropped_item_count = 0
    total_user_links = 0
    matched_user_links = 0

    entropy_values: List[float] = []
    normalized_entropy_values: List[float] = []
    top1_ratio_values: List[float] = []
    unique_sid_values: List[float] = []
    item_size_values: List[float] = []

    out_rows: List[List[object]] = []

    for item_id, users in item_to_users.items():
        if not isinstance(users, list):
            dropped_item_count += 1
            continue

        total_user_links += len(users)
        sids = [user_to_sid[u] for u in users if u in user_to_sid]
        matched_user_links += len(sids)
        if len(sids) < min_users_per_item:
            dropped_item_count += 1
            continue

        freq = Counter(sids)
        counts = list(freq.values())
        n = len(sids)
        k = len(freq)

        entropy = _entropy_from_counts(counts)
        top1_ratio = max(counts) / float(n)
        unique_sid = float(k)
        normalized_entropy = 0.0 if k <= 1 else entropy / math.log(k + 1e-12)

        kept_item_count += 1
        entropy_values.append(entropy)
        normalized_entropy_values.append(normalized_entropy)
        top1_ratio_values.append(top1_ratio)
        unique_sid_values.append(unique_sid)
        item_size_values.append(float(n))

        if per_item_csv_path:
            out_rows.append(
                [
                    item_id,
                    n,
                    k,
                    entropy,
                    normalized_entropy,
                    top1_ratio,
                ]
            )

    if per_item_csv_path:
        os.makedirs(os.path.dirname(per_item_csv_path), exist_ok=True)
        with open(per_item_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "item_id",
                    "item_user_count",
                    "unique_sid_count",
                    "sid_entropy",
                    "sid_entropy_norm",
                    "sid_top1_ratio",
                ]
            )
            writer.writerows(out_rows)

    p25_ent, p50_ent, p75_ent = _quantiles(entropy_values)
    p25_top1, p50_top1, p75_top1 = _quantiles(top1_ratio_values)
    p25_unique, p50_unique, p75_unique = _quantiles(unique_sid_values)

    weighted_entropy_num = 0.0
    weight_den = 0.0
    for ent, n in zip(entropy_values, item_size_values):
        weighted_entropy_num += ent * n
        weight_den += n

    return {
        "experiment": exp_name,
        "items_total": float(len(item_to_users)),
        "items_kept": float(kept_item_count),
        "items_dropped": float(dropped_item_count),
        "raw_user_links": float(total_user_links),
        "matched_user_links": float(matched_user_links),
        "matched_link_coverage": float(matched_user_links / max(total_user_links, 1)),
        "entropy_mean": _safe_mean(entropy_values),
        "entropy_weighted_mean": float(weighted_entropy_num / max(weight_den, 1.0)),
        "entropy_p25": p25_ent,
        "entropy_p50": p50_ent,
        "entropy_p75": p75_ent,
        "entropy_norm_mean": _safe_mean(normalized_entropy_values),
        "top1_ratio_mean": _safe_mean(top1_ratio_values),
        "top1_ratio_p25": p25_top1,
        "top1_ratio_p50": p50_top1,
        "top1_ratio_p75": p75_top1,
        "unique_sid_mean": _safe_mean(unique_sid_values),
        "unique_sid_p25": p25_unique,
        "unique_sid_p50": p50_unique,
        "unique_sid_p75": p75_unique,
    }


def _print_summary(rows: List[Dict[str, float]]) -> None:
    headers = [
        "experiment",
        "items_kept",
        "matched_link_coverage",
        "entropy_p50",
        "top1_ratio_p50",
        "unique_sid_p50",
    ]
    print("\n=== Item-internal SID distribution summary ===")
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["experiment"]),
                    f"{int(row['items_kept'])}",
                    f"{row['matched_link_coverage']:.4f}",
                    f"{row['entropy_p50']:.4f}",
                    f"{row['top1_ratio_p50']:.4f}",
                    f"{row['unique_sid_p50']:.2f}",
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate item-internal SID distributions across experiments."
    )
    parser.add_argument(
        "--item-user-map-path",
        type=str,
        default="/data/tangning/proj_6/amazon_raw/item_to_user_ids.json",
        help="Path to JSON mapping item_id -> [user_id].",
    )
    parser.add_argument(
        "--exp",
        action="append",
        required=True,
        help="Experiment spec in format name=path_to_output_dir_or_csv. "
        "If a directory is given, script reads <dir>/user_semantic_ids.csv",
    )
    parser.add_argument(
        "--sid-mode",
        type=str,
        choices=["full", "rqid_0", "rqid_1", "rqid_2"],
        default="full",
        help="Use full 3-layer SID or a single layer for analysis.",
    )
    parser.add_argument(
        "--min-users-per-item",
        type=int,
        default=2,
        help="Drop items with fewer matched users than this threshold.",
    )
    parser.add_argument(
        "--summary-csv-path",
        type=str,
        default="outputs/sid_eval/summary.csv",
        help="Where to write experiment-level summary CSV.",
    )
    parser.add_argument(
        "--per-item-dir",
        type=str,
        default="",
        help="Optional output dir for per-item metrics CSV per experiment.",
    )
    args = parser.parse_args()

    with open(args.item_user_map_path, "r", encoding="utf-8") as f:
        item_to_users = json.load(f)
    if not isinstance(item_to_users, dict):
        raise ValueError(f"Invalid item-user map json: {args.item_user_map_path}")

    summaries: List[Dict[str, float]] = []
    for raw_exp in args.exp:
        exp_name, exp_path = _parse_experiment_arg(raw_exp)
        sid_csv = _resolve_sid_csv(exp_path)
        user_to_sid = _load_user_sid_map(sid_csv, sid_mode=args.sid_mode)
        per_item_csv_path = (
            os.path.join(args.per_item_dir, f"{exp_name}.per_item.csv")
            if args.per_item_dir
            else ""
        )
        summary = _evaluate_experiment(
            exp_name=exp_name,
            user_to_sid=user_to_sid,
            item_to_users=item_to_users,
            min_users_per_item=args.min_users_per_item,
            per_item_csv_path=per_item_csv_path,
        )
        summaries.append(summary)

    os.makedirs(os.path.dirname(args.summary_csv_path), exist_ok=True)
    summary_headers = [
        "experiment",
        "items_total",
        "items_kept",
        "items_dropped",
        "raw_user_links",
        "matched_user_links",
        "matched_link_coverage",
        "entropy_mean",
        "entropy_weighted_mean",
        "entropy_p25",
        "entropy_p50",
        "entropy_p75",
        "entropy_norm_mean",
        "top1_ratio_mean",
        "top1_ratio_p25",
        "top1_ratio_p50",
        "top1_ratio_p75",
        "unique_sid_mean",
        "unique_sid_p25",
        "unique_sid_p50",
        "unique_sid_p75",
    ]
    with open(args.summary_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(summaries)

    _print_summary(summaries)
    print(f"\nSummary saved to: {args.summary_csv_path}")
    if args.per_item_dir:
        print(f"Per-item csv dir: {args.per_item_dir}")


if __name__ == "__main__":
    main()
