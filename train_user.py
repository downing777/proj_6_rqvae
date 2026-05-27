import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class UserDataset(Dataset):
    user_ids: List[str]
    user_embeddings: torch.Tensor
    user_review_indices: List[List[int]]

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {"embedding": self.user_embeddings[idx]}


@dataclass
class ItemDataset(Dataset):
    item_ids: List[str]
    item_user_indices: List[List[int]]
    user_embeddings: torch.Tensor

    def __len__(self) -> int:
        return len(self.item_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {"user_indices": self.item_user_indices[idx]}


def _collate_user(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {"embeddings": torch.stack([row["embedding"] for row in batch], dim=0)}


def _collate_item(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {"item_user_indices": [row["user_indices"] for row in batch]}


def _add_local_rqvae_to_path() -> None:
    proj_root = os.path.dirname(os.path.abspath(__file__))
    rqvae_root = os.path.join(proj_root, "rqvae")
    if rqvae_root not in sys.path:
        sys.path.insert(0, rqvae_root)


def _disable_torch_compile() -> None:
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.suppress_errors = True
    if hasattr(torch, "compile"):
        def _identity_compile(fn=None, *args, **kwargs):
            if fn is None:
                return lambda f: f
            return fn
        torch.compile = _identity_compile


def _derive_bundle_paths(npz_path: str) -> Tuple[str, str, str, str]:
    if npz_path.lower().endswith(".user.npz"):
        stem = npz_path[: -len(".user.npz")]
    elif npz_path.lower().endswith(".item.npz"):
        stem = npz_path[: -len(".item.npz")]
    elif npz_path.lower().endswith(".npz"):
        stem = npz_path[: -len(".npz")]
    else:
        raise ValueError(
            f"NPZ dataset path must end with .npz/.user.npz/.item.npz, got: {npz_path}"
        )
    return (
        f"{stem}.user.npz",
        f"{stem}.item.npz",
        f"{stem}.user_ids.json",
        f"{stem}.item_ids.json",
    )


def _torch_load_compat(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_item_user_indices_from_json(
    item_user_map_path: str,
    user_ids: List[str],
) -> Tuple[List[str], List[List[int]], int]:
    with open(item_user_map_path, "r", encoding="utf-8") as f:
        raw_item_to_users = json.load(f)
    if not isinstance(raw_item_to_users, dict):
        raise ValueError(f"Expected object mapping in {item_user_map_path}")

    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_ids: List[str] = []
    item_user_indices: List[List[int]] = []
    total_raw_links = 0
    for item_id, user_list in raw_item_to_users.items():
        if not isinstance(user_list, list):
            continue
        total_raw_links += len(user_list)

        mapped: List[int] = []
        seen = set()
        for uid in user_list:
            uidx = user_id_to_idx.get(uid)
            if uidx is None or uidx in seen:
                continue
            seen.add(uidx)
            mapped.append(uidx)

        if mapped:
            item_ids.append(item_id)
            item_user_indices.append(mapped)

    if not item_ids:
        raise ValueError(
            "No valid item-user mappings remain after aligning with user_ids. "
            f"Please check: {item_user_map_path}"
        )
    return item_ids, item_user_indices, total_raw_links


def _load_built_dataset_npz(path: str, item_user_map_path: str) -> Tuple[UserDataset, ItemDataset]:
    user_npz_path, _, user_ids_path, _ = _derive_bundle_paths(path)
    required = [user_ids_path, item_user_map_path]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset files:\n- " + "\n- ".join(missing)
        )

    with open(user_ids_path, "r", encoding="utf-8") as f:
        user_ids = json.load(f)

    if os.path.isfile(user_npz_path):
        user_data = np.load(user_npz_path, allow_pickle=False)
    elif path.lower().endswith(".npz") and os.path.isfile(path):
        # Backward compatibility: old single-file npz bundle.
        user_data = np.load(path, allow_pickle=False)
    else:
        raise FileNotFoundError(
            "Missing user embedding npz. Expected one of:\n"
            f"- {user_npz_path}\n"
            f"- {path} (legacy single-file bundle)"
        )

    user_embeddings = torch.from_numpy(user_data["user_embeddings"]).float()
    item_ids, item_user_indices, total_raw_links = _load_item_user_indices_from_json(
        item_user_map_path, user_ids
    )
    total_mapped_links = sum(len(v) for v in item_user_indices)
    coverage = total_mapped_links / max(total_raw_links, 1)
    print(
        f"[data] Item-user map loaded: items={len(item_ids)} "
        f"mapped_links={total_mapped_links} raw_links={total_raw_links} "
        f"coverage={coverage:.4f} user_vocab={len(user_ids)}"
    )

    user_ds = UserDataset(
        user_ids=user_ids,
        user_embeddings=user_embeddings,
        user_review_indices=[],
    )
    item_ds = ItemDataset(
        item_ids=item_ids,
        item_user_indices=item_user_indices,
        user_embeddings=user_embeddings,
    )
    return user_ds, item_ds


def _load_built_dataset_pt(path: str) -> Tuple[UserDataset, ItemDataset]:
    state = _torch_load_compat(path, map_location="cpu")
    user_embeddings = state["user_embeddings"].float()

    user_ds = UserDataset(
        user_ids=state["user_ids"],
        user_embeddings=user_embeddings,
        user_review_indices=state["user_review_indices"],
    )
    item_ds = ItemDataset(
        item_ids=state["item_ids"],
        item_user_indices=state["item_user_indices"],
        user_embeddings=user_embeddings,
    )
    return user_ds, item_ds


def _load_built_dataset(path: str, item_user_map_path: str) -> Tuple[UserDataset, ItemDataset]:
    if path.lower().endswith(".npz"):
        return _load_built_dataset_npz(path, item_user_map_path=item_user_map_path)
    return _load_built_dataset_pt(path)


def _as_seq_batch(x: torch.Tensor, device: str):
    from data.schemas import SeqBatch

    b = x.shape[0]
    ids = torch.arange(b, device=device)
    return SeqBatch(
        user_ids=-torch.ones(b, dtype=torch.long, device=device),
        ids=ids,
        ids_fut=-torch.ones(b, dtype=torch.long, device=device),
        x=x,
        x_fut=-torch.ones_like(x),
        seq_mask=torch.ones((b, 1), dtype=torch.bool, device=device),
    )


def _sample_users_from_items(
    item_user_indices: List[List[int]],
    user_embeddings: torch.Tensor,
    rng: random.Random,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    sampled_indices_per_item: List[List[int]] = []
    for users in item_user_indices:
        if not users:
            continue
        sampled_indices_per_item.append(rng.sample(users, len(users)))
    if not sampled_indices_per_item:
        raise RuntimeError("No users sampled from item batch.")
    flat_indices = [idx for picked in sampled_indices_per_item for idx in picked]
    sampled = user_embeddings[flat_indices]

    item_groups = []
    offset = 0
    for picked in sampled_indices_per_item:
        item_groups.append(torch.arange(offset, offset + len(picked), dtype=torch.long))
        offset += len(picked)
    return sampled, item_groups


def _sample_users_from_items_with_dedup(
    item_user_indices: List[List[int]],
    user_embeddings: torch.Tensor,
    rng: random.Random,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    sampled_indices_per_item: List[List[int]] = []
    for users in item_user_indices:
        if not users:
            continue
        sampled_indices_per_item.append(rng.sample(users, len(users)))
    if not sampled_indices_per_item:
        raise RuntimeError("No users sampled from item batch.")

    unique_user_indices: List[int] = []
    global_to_unique: Dict[int, int] = {}
    item_groups: List[torch.Tensor] = []
    for picked in sampled_indices_per_item:
        local_group = []
        for user_idx in picked:
            if user_idx not in global_to_unique:
                global_to_unique[user_idx] = len(unique_user_indices)
                unique_user_indices.append(user_idx)
            local_group.append(global_to_unique[user_idx])
        item_groups.append(torch.tensor(local_group, dtype=torch.long))

    sampled = user_embeddings[unique_user_indices]
    return sampled, item_groups


def _compute_item_mi_regularizer(
    model,
    residuals: torch.Tensor,
    item_groups: List[torch.Tensor],
    mi_alpha: float,
    mi_beta: float,
    mi_tau: float,
    mi_topk: int,
    mi_reg_layers: int,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not item_groups:
        zero = residuals.new_zeros(())
        return zero, {
            "mi_h_global": zero,
            "mi_h_cond": zero,
            "mi_layers_used": zero,
        }

    n_layers = min(mi_reg_layers, residuals.shape[-1], len(model.layers))
    if n_layers <= 0:
        zero = residuals.new_zeros(())
        return zero, {
            "mi_h_global": zero,
            "mi_h_cond": zero,
            "mi_layers_used": zero,
        }

    total_memberships = float(sum(group.numel() for group in item_groups))
    if total_memberships <= 0:
        zero = residuals.new_zeros(())
        return zero, {
            "mi_h_global": zero,
            "mi_h_cond": zero,
            "mi_layers_used": zero,
        }

    reg_total = residuals.new_zeros(())
    h_global_total = residuals.new_zeros(())
    h_cond_total = residuals.new_zeros(())

    for layer_idx in range(n_layers):
        residual_k = residuals[:, :, layer_idx]
        layer = model.layers[layer_idx]
        codebook = layer.out_proj(layer.embedding.weight)
        dist = (
            (residual_k**2).sum(dim=1, keepdim=True)
            + (codebook**2).sum(dim=1).unsqueeze(0)
            - 2 * residual_k @ codebook.T
        )

        if mi_topk > 0 and mi_topk < codebook.shape[0]:
            top_dist, top_idx = torch.topk(dist, k=mi_topk, dim=1, largest=False)
            top_prob = torch.softmax(-top_dist / mi_tau, dim=1)
            pi = torch.zeros_like(dist).scatter(1, top_idx, top_prob)
        else:
            pi = torch.softmax(-dist / mi_tau, dim=1)

        p_global = pi.mean(dim=0)
        h_global = -(p_global * torch.log(p_global + eps)).sum()

        h_cond = residuals.new_zeros(())
        for group in item_groups:
            if group.numel() == 0:
                continue
            p_item = pi.index_select(0, group).mean(dim=0)
            h_item = -(p_item * torch.log(p_item + eps)).sum()
            h_cond = h_cond + (float(group.numel()) / total_memberships) * h_item

        reg_total = reg_total + (mi_alpha * h_cond - mi_beta * h_global)
        h_global_total = h_global_total + h_global
        h_cond_total = h_cond_total + h_cond

    layers_denom = float(n_layers)
    return reg_total, {
        "mi_h_global": h_global_total / layers_denom,
        "mi_h_cond": h_cond_total / layers_denom,
        "mi_layers_used": residuals.new_tensor(layers_denom),
    }


def _train_rqvae(
    user_ds: UserDataset,
    item_ds: ItemDataset,
    sample_by: str,
    iterations: int,
    batch_size: int,
    learning_rate: float,
    hidden_dims: List[int],
    embed_dim: int,
    codebook_size: int,
    n_layers: int,
    commitment_weight: float,
    device: str,
    seed: int,
    wandb_logging: bool,
    enable_item_mi_loss: bool,
    dedup_users_in_item_batch: bool,
    mi_alpha: float,
    mi_beta: float,
    mi_weight: float,
    mi_tau: float,
    mi_topk: int,
    mi_reg_layers: int,
    mi_warmup_steps: int,
):
    from modules.quantize import QuantizeForwardMode
    from modules.rqvae import RqVae
    from modules.normalize import l2norm

    model = RqVae(
        input_dim=user_ds.user_embeddings.shape[1],
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        codebook_size=codebook_size,
        codebook_kmeans_init=True,
        codebook_normalize=False,
        codebook_sim_vq=False,
        codebook_mode=QuantizeForwardMode.ROTATION_TRICK,
        n_layers=n_layers,
        commitment_weight=commitment_weight,
        n_cat_features=0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    rng = random.Random(seed)
    if sample_by == "user":
        loader = DataLoader(user_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_user)
    else:
        loader = DataLoader(item_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_item)
    it = iter(loader)

    if wandb_logging:
        if wandb is None:
            raise ImportError("wandb is not installed. Install it or disable --wandb-logging.")
        wandb.login()
        wandb.init(
            project="rq-vae-user",
            config={
                "sample_by": sample_by,
                "train_iterations": iterations,
                "train_batch_size": batch_size,
                "learning_rate": learning_rate,
                "hidden_dims": hidden_dims,
                "embed_dim": embed_dim,
                "codebook_size": codebook_size,
                "n_layers": n_layers,
                "commitment_weight": commitment_weight,
                "seed": seed,
                "enable_item_mi_loss": enable_item_mi_loss,
                "dedup_users_in_item_batch": dedup_users_in_item_batch,
                "mi_alpha": mi_alpha,
                "mi_beta": mi_beta,
                "mi_weight": mi_weight,
                "mi_tau": mi_tau,
                "mi_topk": mi_topk,
                "mi_reg_layers": mi_reg_layers,
                "mi_warmup_steps": mi_warmup_steps,
            },
        )

    use_item_mi_loss = enable_item_mi_loss and sample_by == "item"
    if enable_item_mi_loss and sample_by != "item":
        print("[warn] --enable-item-mi-loss only works with --sample-by=item. Disabled for this run.")

    for step in range(1, iterations + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        item_groups: Optional[List[torch.Tensor]] = None
        if sample_by == "user":
            x = batch["embeddings"]
        else:
            if dedup_users_in_item_batch:
                x, item_groups = _sample_users_from_items_with_dedup(
                    item_user_indices=batch["item_user_indices"],
                    user_embeddings=item_ds.user_embeddings,
                    rng=rng,
                )
            else:
                x, item_groups = _sample_users_from_items(
                    item_user_indices=batch["item_user_indices"],
                    user_embeddings=item_ds.user_embeddings,
                    rng=rng,
                )
        x = x.to(device)
        if item_groups is not None:
            item_groups = [group.to(device) for group in item_groups]

        use_mi_now = use_item_mi_loss and step > mi_warmup_steps

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Gumbel temperature annealing: start high (1.0) and decay to 0.1
        gumbel_t = max(0.1, 1.0 * (0.999 ** step))
        quantized = model.get_semantic_ids(x, gumbel_t=gumbel_t)
        x_hat = model.decode(quantized.embeddings.sum(axis=-1))
        if model.n_cat_feats > 0:
            x_hat = torch.cat(
                [l2norm(x_hat[..., :-model.n_cat_feats]), x_hat[..., -model.n_cat_feats:]],
                dim=-1,
            )
        else:
            x_hat = l2norm(x_hat)

        reconstruction_loss = model.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        base_loss = (reconstruction_loss + rqvae_loss).mean()

        mi_reg = x.new_zeros(())
        mi_metrics = {
            "mi_h_global": x.new_zeros(()),
            "mi_h_cond": x.new_zeros(()),
        }
        if use_mi_now and item_groups is not None:
            mi_reg, mi_metrics = _compute_item_mi_regularizer(
                model=model,
                residuals=quantized.residuals,
                item_groups=item_groups,
                mi_alpha=mi_alpha,
                mi_beta=mi_beta,
                mi_tau=mi_tau,
                mi_topk=mi_topk,
                mi_reg_layers=mi_reg_layers,
            )

        loss = base_loss + mi_weight * mi_reg
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 1 or step == iterations:
            # Compute codebook utilization per layer
            cb_utils = []
            for layer in model.layers:
                active = (layer.code_usage_ema >= layer.dead_code_threshold).sum().item()
                cb_utils.append(f"{int(active)}/{layer.n_embed}")
            cb_str = ",".join(cb_utils)

            msg = (
                f"[train] step={step}/{iterations} "
                f"loss={loss.item():.4f} "
                f"base={base_loss.item():.4f} "
                f"rec={reconstruction_loss.mean().item():.4f} "
                f"vq={rqvae_loss.mean().item():.4f} "
                f"t={gumbel_t:.3f} cb=[{cb_str}]"
            )
            if use_item_mi_loss:
                warmup_state = "on" if use_mi_now else f"warmup(step {step}/{mi_warmup_steps})"
                msg += (
                    f" mi_state={warmup_state}"
                    f" mi={mi_reg.item():.4f}"
                    f" h_global={mi_metrics['mi_h_global'].item():.4f}"
                    f" h_cond={mi_metrics['mi_h_cond'].item():.4f}"
                )
            print(msg)
        if wandb_logging:
            logs = {
                "step": step,
                "loss": float(loss.item()),
                "base_loss": float(base_loss.item()),
                "reconstruction_loss": float(reconstruction_loss.mean().item()),
                "vq_loss": float(rqvae_loss.mean().item()),
                "gumbel_t": float(gumbel_t),
            }
            for layer_idx, layer in enumerate(model.layers):
                active = (layer.code_usage_ema >= layer.dead_code_threshold).sum().item()
                logs[f"cb_active_layer{layer_idx}"] = active
            if use_item_mi_loss:
                logs.update(
                    {
                        "mi_reg": float(mi_reg.item()),
                        "mi_h_global": float(mi_metrics["mi_h_global"].item()),
                        "mi_h_cond": float(mi_metrics["mi_h_cond"].item()),
                    }
                )
            wandb.log(logs)

    if wandb_logging:
        wandb.finish()
    return model


def _extract_semantic_ids(model, user_embeddings: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    model.eval()
    all_ids = []
    with torch.no_grad():
        for i in range(0, user_embeddings.shape[0], batch_size):
            xb = user_embeddings[i : i + batch_size].to(device)
            all_ids.append(model.get_semantic_ids(xb, gumbel_t=0.001).sem_ids.detach().cpu())
    return torch.cat(all_ids, dim=0)


def _save_outputs(output_dir: str, user_ids: List[str], sem_ids: torch.Tensor, model) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "user_semantic_ids.csv")
    jsonl_path = os.path.join(output_dir, "user_semantic_ids.jsonl")
    ckpt_path = os.path.join(output_dir, "user_rqvae.pt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id_raw", "user_id", "rqid_0", "rqid_1", "rqid_2"])
        for idx, uid in enumerate(user_ids):
            writer.writerow([uid, idx] + sem_ids[idx].tolist())

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, uid in enumerate(user_ids):
            f.write(json.dumps({"user_id_raw": uid, "user_id": idx, "rqvae_id": sem_ids[idx].tolist()}) + "\n")

    torch.save({"model": model.state_dict(), "model_config": model.config}, ckpt_path)
    print(f"Saved:\n- {csv_path}\n- {jsonl_path}\n- {ckpt_path}")


def _resolve_device(device_arg: str) -> str:
    requested = (device_arg or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested device=cuda but CUDA is not available.")
        return "cuda"

    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"Requested device={requested} but CUDA is not available.")
        try:
            idx = int(requested.split(":", 1)[1])
        except ValueError as e:
            raise ValueError(f"Invalid CUDA device format: {device_arg}") from e
        if idx < 0 or idx >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index out of range: {idx}. "
                f"Available count={torch.cuda.device_count()}."
            )
        return f"cuda:{idx}"

    raise ValueError(
        f"Unsupported --device={device_arg}. Use one of: auto, cpu, cuda, cuda:<index>."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train user RQ-VAE from prebuilt Amazon dataset cache.")
    parser.add_argument(
        "--built-data-path",
        type=str,
        default="/home/yuanhanyang.yhy/model_hub/amazon_user/amazon_user_item_dataset.user.npz",
        help="Path to dataset base .npz or split .user.npz (for example amazon_user_item_dataset.npz or amazon_user_item_dataset.user.npz).",
    )
    parser.add_argument(
        "--item-user-map-path",
        type=str,
        default="/home/yuanhanyang.yhy/model_hub/amazon_user/raw/item_to_user_ids.json",
        help="JSON file mapping item_id -> [user_id], used to build item-side user distribution for sampling.",
    )
    parser.add_argument("--output-dir", type=str, default="/home/yuanhanyang.yhy/project_6_outputs/sid")
    parser.add_argument("--sample-by", type=str, choices=["user", "item"], default="user")
    parser.add_argument("--train-iterations", type=int, default=10000)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--codebook-size", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--commitment-weight", type=float, default=0.1)
    parser.add_argument("--enable-item-mi-loss", action="store_true")
    parser.add_argument("--mi-alpha", type=float, default=1.0) # 互信息损失的权重
    parser.add_argument("--mi-beta", type=float, default=1.0)
    parser.add_argument("--mi-weight", type=float, default=1.0) # MI loss 的权重
    parser.add_argument("--mi-tau", type=float, default=0.2)
    parser.add_argument("--mi-topk", type=int, default=32)
    parser.add_argument("--mi-reg-layers", type=int, default=3)
    parser.add_argument("--mi-warmup-steps", type=int, default=0)
    parser.add_argument("--no-dedup-users-in-item-batch", action="store_false", dest="dedup_users_in_item_batch")
    parser.set_defaults(dedup_users_in_item_batch=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-logging", action="store_true")
    args = parser.parse_args()

    if args.n_layers != 3:
        raise ValueError("This pipeline enforces 3-layer user semantic IDs.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _disable_torch_compile()
    _add_local_rqvae_to_path()
    device = _resolve_device(args.device)
    print(f"[env] Using device: {device}")

    print("[1/3] Loading built dataset...")
    user_ds, item_ds = _load_built_dataset(args.built_data_path, item_user_map_path=args.item_user_map_path)
    print(f"Users={len(user_ds)} Items={len(item_ds)} EmbDim={user_ds.user_embeddings.shape[1]}")

    print(f"[2/3] Training RQ-VAE (sample-by={args.sample_by})...")
    model = _train_rqvae(
        user_ds=user_ds,
        item_ds=item_ds,
        sample_by=args.sample_by,
        iterations=args.train_iterations,
        batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        hidden_dims=args.hidden_dims,
        embed_dim=args.embed_dim,
        codebook_size=args.codebook_size,
        n_layers=args.n_layers,
        commitment_weight=args.commitment_weight,
        device=device,
        seed=args.seed,
        wandb_logging=args.wandb_logging,
        enable_item_mi_loss=args.enable_item_mi_loss,
        dedup_users_in_item_batch=args.dedup_users_in_item_batch,
        mi_alpha=args.mi_alpha,
        mi_beta=args.mi_beta,
        mi_weight=args.mi_weight,
        mi_tau=args.mi_tau,
        mi_topk=args.mi_topk,
        mi_reg_layers=args.mi_reg_layers,
        mi_warmup_steps=args.mi_warmup_steps,
    )

    print("[3/3] Exporting semantic IDs...")
    sem_ids = _extract_semantic_ids(model, user_ds.user_embeddings, args.train_batch_size, device)
    _save_outputs(args.output_dir, user_ds.user_ids, sem_ids, model)


if __name__ == "__main__":
    main()
