import argparse
import ast
import csv
import datetime as dt
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


@dataclass
class ReviewRecord:
    user_id: str
    item_id: str
    text: str
    timestamp: int


def _iter_json_lines(path: str) -> Iterable[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def _extract_record(row: dict) -> Optional[ReviewRecord]:
    # New format
    if "user_id" in row:
        user_id = str(row.get("user_id") or "")
        item_id = str(row.get("parent_asin") or row.get("asin") or "")
        timestamp = int(row.get("timestamp", 0) or 0)
        text = (str(row.get("title", "") or "") + " " + str(row.get("text", "") or "")).strip()
    else:
        # Legacy format
        user_id = str(row.get("reviewerID") or "")
        item_id = str(row.get("asin") or "")
        timestamp = int(row.get("unixReviewTime", 0) or 0)
        text = (str(row.get("summary", "") or "") + " " + str(row.get("reviewText", "") or "")).strip()

    if not user_id or not item_id or not text:
        return None

    return ReviewRecord(
        user_id=user_id,
        item_id=item_id,
        text=text,
        timestamp=timestamp,
    )


def _discover_files(reviews_path: str, category_glob: str) -> List[str]:
    if os.path.isfile(reviews_path):
        return [reviews_path]
    if not os.path.isdir(reviews_path):
        raise FileNotFoundError(f"reviews_path not found: {reviews_path}")

    out = []
    for name in sorted(os.listdir(reviews_path)):
        if not (name.endswith(".jsonl") or name.endswith(".jsonl.gz")):
            continue
        if category_glob and category_glob not in name:
            continue
        out.append(os.path.join(reviews_path, name))
    if not out:
        raise RuntimeError(f"No review jsonl files found in {reviews_path}")
    return out


def _load_and_filter_reviews(
    reviews_path: str,
    category_glob: str,
    max_reviews_per_user: int,
    min_reviews_per_user: int,
    max_files: int,
    max_rows: int,
    max_users: int,
) -> List[ReviewRecord]:
    per_user: Dict[str, List[ReviewRecord]] = defaultdict(list)
    files = _discover_files(reviews_path, category_glob)
    if max_files > 0:
        files = files[:max_files]
        print(f"[debug] Limiting to first {len(files)} files")

    seen_rows = 0
    kept_rows = 0
    for fp in tqdm(files, desc="Loading review files", unit="file"):
        for row in _iter_json_lines(fp):
            if max_rows > 0 and seen_rows >= max_rows:
                break
            seen_rows += 1
            rec = _extract_record(row)
            if rec is not None:
                if max_users > 0 and rec.user_id not in per_user and len(per_user) >= max_users:
                    continue
                per_user[rec.user_id].append(rec)
                kept_rows += 1
        tqdm.write(
            f"[load] {os.path.basename(fp)} | rows={seen_rows} kept={kept_rows} users={len(per_user)}"
        )
        if max_rows > 0 and seen_rows >= max_rows:
            print(f"[debug] Reached --max-rows={max_rows}, stop loading more files.")
            break

    records: List[ReviewRecord] = []
    for _, user_records in tqdm(per_user.items(), total=len(per_user), desc="Filtering users", unit="user"):
        user_records = sorted(user_records, key=lambda r: r.timestamp)
        if len(user_records) > max_reviews_per_user:
            user_records = user_records[-max_reviews_per_user:]
        if len(user_records) < min_reviews_per_user:
            continue
        records.extend(user_records)

    if not records:
        raise RuntimeError("No valid records after filtering.")
    return records


def _encode_reviews(
    records: List[ReviewRecord],
    embedding_model: str,
    batch_size: int,
    device: str,
    gpu_ids: str,
    output_dir: str,
    resume_embeddings: bool,
    checkpoint_interval: int,
) -> torch.Tensor:
    texts = [r.text for r in records]
    if not texts:
        return torch.empty((0, 0), dtype=torch.float32)

    available_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    parsed_gpu_ids: List[int] = []
    if gpu_ids.strip():
        parsed_gpu_ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
    elif available_gpu_count > 0:
        parsed_gpu_ids = list(range(available_gpu_count))

    valid_gpu_ids = [gid for gid in parsed_gpu_ids if 0 <= gid < available_gpu_count]
    if gpu_ids.strip() and not valid_gpu_ids:
        raise ValueError(f"--gpu-ids has no valid gpu index: {gpu_ids}, available=0..{available_gpu_count - 1}")

    cache_dir = os.path.join(output_dir, "review_embedding_cache")
    os.makedirs(cache_dir, exist_ok=True)
    mmap_path = os.path.join(cache_dir, "review_embeddings.float32.mmap")
    meta_path = os.path.join(cache_dir, "meta.json")

    checkpoint_interval = max(1, int(checkpoint_interval))

    def _write_meta(next_idx: int, embedding_dim: int, completed: bool = False) -> None:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_reviews": len(texts),
                    "embedding_dim": int(embedding_dim),
                    "next_idx": int(next_idx),
                    "embedding_model": embedding_model,
                    "batch_size": int(batch_size),
                    "gpu_ids": gpu_ids,
                    "completed": completed,
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    start_idx = 0
    cached_dim: Optional[int] = None
    if resume_embeddings and os.path.isfile(meta_path) and os.path.isfile(mmap_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("total_reviews") == len(texts) and meta.get("embedding_model") == embedding_model:
            start_idx = int(meta.get("next_idx", 0))
            cached_dim = int(meta.get("embedding_dim", 0) or 0) or None
            if start_idx > 0:
                print(f"[encode] Resume cache found: {start_idx}/{len(texts)}")
        else:
            print("[encode] Existing cache incompatible, restarting embedding.")

    if start_idx <= 0 and os.path.isfile(mmap_path):
        os.remove(mmap_path)

    use_multi_gpu = device.startswith("cuda") and len(valid_gpu_ids) > 1
    if use_multi_gpu:
        target_devices = [f"cuda:{gid}" for gid in valid_gpu_ids]
        print(f"[encode] Multi-GPU enabled: {target_devices}")
        # Keep coordinator on CPU and run workers on selected GPUs.
        model = SentenceTransformer(embedding_model, device="cpu")
        pool = model.start_multi_process_pool(target_devices=target_devices)
        # One chunk feeds roughly one batch per GPU.
        chunk_size = max(batch_size * len(target_devices), batch_size)
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        mmap = (
            np.memmap(mmap_path, mode="r+", dtype=np.float32, shape=(len(texts), cached_dim))
            if cached_dim is not None and start_idx > 0
            else None
        )
        try:
            steps_since_checkpoint = 0
            for i in tqdm(
                range(start_idx, len(texts), chunk_size),
                total=total_chunks,
                initial=start_idx // chunk_size,
                desc="Embedding reviews (multi-gpu)",
                unit="chunk",
            ):
                text_chunk = texts[i : i + chunk_size]
                chunk_np = model.encode_multi_process(text_chunk, pool=pool, batch_size=batch_size)
                if mmap is None:
                    mmap = np.memmap(
                        mmap_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(texts), int(chunk_np.shape[1])),
                    )
                end = i + chunk_np.shape[0]
                mmap[i:end] = chunk_np
                steps_since_checkpoint += 1
                if steps_since_checkpoint >= checkpoint_interval:
                    mmap.flush()
                    _write_meta(next_idx=end, embedding_dim=int(chunk_np.shape[1]), completed=False)
                    steps_since_checkpoint = 0
        finally:
            model.stop_multi_process_pool(pool)
        if mmap is None:
            raise RuntimeError("Embedding produced no outputs.")
        mmap.flush()
        _write_meta(next_idx=len(texts), embedding_dim=int(mmap.shape[1]), completed=True)
        return torch.from_numpy(np.asarray(mmap))

    single_device = device
    if device.startswith("cuda") and valid_gpu_ids:
        single_device = f"cuda:{valid_gpu_ids[0]}"
    print(f"[encode] Single-GPU/CPU mode on: {single_device}")
    model = SentenceTransformer(embedding_model, device=single_device)
    mmap = (
        np.memmap(mmap_path, mode="r+", dtype=np.float32, shape=(len(texts), cached_dim))
        if cached_dim is not None and start_idx > 0
        else None
    )
    total_batches = (len(texts) + batch_size - 1) // batch_size
    steps_since_checkpoint = 0
    for i in tqdm(
        range(start_idx, len(texts), batch_size),
        total=total_batches,
        initial=start_idx // batch_size,
        desc="Embedding reviews",
        unit="batch",
    ):
        batch_texts = texts[i : i + batch_size]
        batch_np = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if mmap is None:
            mmap = np.memmap(
                mmap_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(texts), int(batch_np.shape[1])),
            )
        end = i + batch_np.shape[0]
        mmap[i:end] = batch_np
        steps_since_checkpoint += 1
        if steps_since_checkpoint >= checkpoint_interval:
            mmap.flush()
            _write_meta(next_idx=end, embedding_dim=int(batch_np.shape[1]), completed=False)
            steps_since_checkpoint = 0

    if mmap is None:
        raise RuntimeError("Embedding produced no outputs.")
    mmap.flush()
    _write_meta(next_idx=len(texts), embedding_dim=int(mmap.shape[1]), completed=True)
    return torch.from_numpy(np.asarray(mmap))


def _build_mappings(
    records: List[ReviewRecord],
    review_embeddings: torch.Tensor,
) -> dict:
    review_user_ids = [r.user_id for r in records]
    review_item_ids = [r.item_id for r in records]
    review_timestamps = torch.tensor([r.timestamp for r in records], dtype=torch.long)

    user_to_review_indices: Dict[str, List[int]] = defaultdict(list)
    item_to_user_set: Dict[str, set] = defaultdict(set)
    for ridx, (uid, iid) in enumerate(zip(review_user_ids, review_item_ids)):
        user_to_review_indices[uid].append(ridx)
        item_to_user_set[iid].add(uid)

    user_ids = sorted(user_to_review_indices.keys())
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    user_review_indices = [user_to_review_indices[uid] for uid in user_ids]
    user_embeddings = torch.stack(
        [review_embeddings[idxs].mean(dim=0) for idxs in user_review_indices],
        dim=0,
    ).float()

    item_ids = sorted(item_to_user_set.keys())
    item_user_indices = [
        sorted(user_id_to_idx[uid] for uid in item_to_user_set[item_id])
        for item_id in item_ids
    ]
    item_embeddings = torch.stack(
        [user_embeddings[idxs].mean(dim=0) for idxs in item_user_indices],
        dim=0,
    ).float()

    item_user_lengths = [len(idxs) for idxs in item_user_indices]
    item_user_offsets = [0]
    for n in item_user_lengths:
        item_user_offsets.append(item_user_offsets[-1] + n)
    item_user_indices_flat = [uidx for idxs in item_user_indices for uidx in idxs]

    return {
        "review_embeddings": review_embeddings.half(),
        "review_user_ids": review_user_ids,
        "review_item_ids": review_item_ids,
        "review_timestamps": review_timestamps,
        "user_ids": user_ids,
        "user_review_indices": user_review_indices,
        "user_embeddings": user_embeddings.half(),
        "item_ids": item_ids,
        "item_user_indices": item_user_indices,
        "item_embeddings": item_embeddings.half(),
        "item_user_offsets": item_user_offsets,
        "item_user_indices_flat": item_user_indices_flat,
    }


def _save_stats(output_dir: str, state: dict) -> None:
    user_stats_path = os.path.join(output_dir, "user_review_stats.csv")
    with open(user_stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id_raw", "num_reviews"])
        for uid, ridxs in zip(state["user_ids"], state["user_review_indices"]):
            writer.writerow([uid, len(ridxs)])

    item_stats_path = os.path.join(output_dir, "item_user_stats.csv")
    with open(item_stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "num_users"])
        for iid, uidxs in zip(state["item_ids"], state["item_user_indices"]):
            writer.writerow([iid, len(uidxs)])

    print(f"Saved stats:\n- {user_stats_path}\n- {item_stats_path}")


def _derive_bundle_paths(npz_path: str) -> Tuple[str, str, str, str]:
    stem, ext = os.path.splitext(npz_path)
    if ext.lower() != ".npz":
        raise ValueError(f"--output-path must end with .npz, got: {npz_path}")
    return (
        f"{stem}.user.npz",
        f"{stem}.item.npz",
        f"{stem}.user_ids.json",
        f"{stem}.item_ids.json",
    )


def _save_npz_bundle(npz_path: str, state: dict) -> None:
    user_npz_path, item_npz_path, user_ids_path, item_ids_path = _derive_bundle_paths(npz_path)
    np.savez_compressed(
        user_npz_path,
        user_embeddings=state["user_embeddings"].numpy(),
    )
    np.savez_compressed(
        item_npz_path,
        item_embeddings=state["item_embeddings"].numpy(),
        item_user_offsets=np.asarray(state["item_user_offsets"], dtype=np.int64),
        item_user_indices_flat=np.asarray(state["item_user_indices_flat"], dtype=np.int32),
    )
    with open(user_ids_path, "w", encoding="utf-8") as f:
        json.dump(state["user_ids"], f, ensure_ascii=False)
    with open(item_ids_path, "w", encoding="utf-8") as f:
        json.dump(state["item_ids"], f, ensure_ascii=False)
    print(f"Saved bundle:\n- {user_npz_path}\n- {item_npz_path}\n- {user_ids_path}\n- {item_ids_path}")


def _save_dataset_schema(output_dir: str, npz_path: str, state: dict) -> None:
    user_npz_path, item_npz_path, user_ids_path, item_ids_path = _derive_bundle_paths(npz_path)
    schema_path = os.path.join(output_dir, "dataset_schema.json")
    schema = {
        "format": "split-npz-with-id-lists",
        "files": {
            "user_npz": {
                "path": os.path.basename(user_npz_path),
                "arrays": {
                    "user_embeddings": {
                        "shape": list(state["user_embeddings"].shape),
                        "dtype": str(state["user_embeddings"].dtype),
                        "description": "User embedding matrix aligned with user_ids.json order.",
                    }
                },
            },
            "item_npz": {
                "path": os.path.basename(item_npz_path),
                "arrays": {
                    "item_embeddings": {
                        "shape": list(state["item_embeddings"].shape),
                        "dtype": str(state["item_embeddings"].dtype),
                        "description": "Item embedding matrix aligned with item_ids.json order.",
                    },
                    "item_user_offsets": {
                        "shape": [len(state["item_user_offsets"])],
                        "dtype": "int64",
                        "description": "CSR-style offsets to reconstruct item->user indices.",
                    },
                    "item_user_indices_flat": {
                        "shape": [len(state["item_user_indices_flat"])],
                        "dtype": "int32",
                        "description": "Flattened user indices for each item segment.",
                    },
                },
            },
            "user_ids_json": {
                "path": os.path.basename(user_ids_path),
                "count": len(state["user_ids"]),
                "description": "List[str], row-aligned with user_embeddings.",
            },
            "item_ids_json": {
                "path": os.path.basename(item_ids_path),
                "count": len(state["item_ids"]),
                "description": "List[str], row-aligned with item_embeddings.",
            },
            "user_review_stats_csv": {
                "path": "user_review_stats.csv",
                "description": "Per-user review counts.",
            },
            "item_user_stats_csv": {
                "path": "item_user_stats.csv",
                "description": "Per-item user counts.",
            },
        },
        "reconstruct_item_user_indices": "users = item_user_indices_flat[item_user_offsets[i]:item_user_offsets[i+1]]",
        "meta": state.get("meta", {}),
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"Saved schema:\n- {schema_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached Amazon user/item datasets for RQ-VAE training.")
    parser.add_argument(
        "--reviews-path",
        type=str,
        default="/data/tangning/proj_6/amazon_raw/step4/final_target_user_reviews_by_category",
        help="Input file or directory with jsonl/jsonl.gz review files.",
    )
    parser.add_argument("--category-glob", type=str, default="", help="Optional filename substring filter.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/tangning/proj_6/amazon_emb",
        help="Directory to store generated npz/json artifacts.",
    )
    parser.add_argument("--embedding-model", type=str, default="/data/tangning/model_hub/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="Comma-separated CUDA IDs, e.g. '0,1,3'. Empty means all visible GPUs.",
    )
    parser.add_argument("--max-reviews-per-user", type=int, default=1000)
    parser.add_argument("--min-reviews-per-user", type=int, default=1)
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="For smoke test: limit number of input files. 0 means no limit.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="For smoke test: limit raw loaded rows across files. 0 means no limit.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="For smoke test: cap distinct users kept while loading. 0 means no limit.",
    )
    parser.add_argument(
        "--resume-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint embedding chunks to disk and resume from cache when available.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Flush embedding cache every N chunks/batches.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bundle_base_path = os.path.join(args.output_dir, "amazon_user_item_dataset.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1/3] Loading raw reviews...")
    records = _load_and_filter_reviews(
        reviews_path=args.reviews_path,
        category_glob=args.category_glob,
        max_reviews_per_user=args.max_reviews_per_user,
        min_reviews_per_user=args.min_reviews_per_user,
        max_files=args.max_files,
        max_rows=args.max_rows,
        max_users=args.max_users,
    )
    print(f"Reviews kept: {len(records)}")

    print("[2/3] Encoding all review texts...")
    review_embeddings = _encode_reviews(
        records=records,
        embedding_model=args.embedding_model,
        batch_size=args.embed_batch_size,
        device=device,
        gpu_ids=args.gpu_ids,
        output_dir=args.output_dir,
        resume_embeddings=args.resume_embeddings,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(f"Review embedding shape: {tuple(review_embeddings.shape)}")

    print("[3/3] Building user/item datasets and saving...")
    state = _build_mappings(records=records, review_embeddings=review_embeddings)
    state["meta"] = {
        "reviews_path": args.reviews_path,
        "category_glob": args.category_glob,
        "embedding_model": args.embedding_model,
    }
    _save_npz_bundle(bundle_base_path, state)
    _save_stats(args.output_dir, state)
    _save_dataset_schema(args.output_dir, bundle_base_path, state)
    print(f"Users: {len(state['user_ids'])} | Items: {len(state['item_ids'])}")


if __name__ == "__main__":
    main()
