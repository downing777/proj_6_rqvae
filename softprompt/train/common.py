import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATE = "你是电商标题生成助手。\n商品信息：{context}\n请为该 SID 用户群生成一个吸引点击的商品标题："


def load_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def render_prompt(context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context)


@dataclass
class TokenizedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sid: torch.Tensor


class SFTJsonlDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        return {
            "sid": row["sid"],
            "prompt": render_prompt(row["context"]),
            "target": row["target_title"],
        }


class DPOJsonlDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        return {
            "sid": row["sid"],
            "prompt": render_prompt(row["context"]),
            "chosen": row["title_chosen"],
            "rejected": row["title_rejected"],
        }


def build_prompt_target_tensors(
    tokenizer,
    prompts: List[str],
    targets: List[str],
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eos = tokenizer.eos_token or ""
    texts = [f"{p}\n标题：{t}{eos}" for p, t in zip(prompts, targets)]
    prompt_only = [f"{p}\n标题：" for p in prompts]

    packed = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    prompt_packed = tokenizer(
        prompt_only,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    labels = packed["input_ids"].clone()
    labels[packed["attention_mask"] == 0] = -100
    for i in range(prompt_packed["attention_mask"].size(0)):
        prompt_len = int(prompt_packed["attention_mask"][i].sum().item())
        labels[i, :prompt_len] = -100
    return packed["input_ids"], packed["attention_mask"], labels


def sequence_logp(model, tokenizer, sid: torch.Tensor, prompts: List[str], targets: List[str], max_length: int):
    input_ids, attention_mask, labels = build_prompt_target_tensors(
        tokenizer=tokenizer, prompts=prompts, targets=targets, max_length=max_length
    )
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    sid = sid.to(device)

    with torch.set_grad_enabled(model.training):
        out = model(input_ids=input_ids, attention_mask=attention_mask, sid=sid, labels=labels)
        logits = out.logits[:, :-1, :]
        target = labels[:, 1:]
        valid = target.ne(-100)
        token_logp = torch.gather(logits.log_softmax(-1), dim=-1, index=target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid
        seq_logp = token_logp.sum(dim=-1)
    return seq_logp, out, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collated_sid_to_tensor(sid_batch, device: str) -> torch.Tensor:
    """
    Convert DataLoader-collated SID batch into shape [B, L].

    PyTorch default collate turns list-valued `sid` into:
      [tensor(level0_ids), tensor(level1_ids), tensor(level2_ids)]
    so we stack by levels and transpose back to batch-major.
    """
    if isinstance(sid_batch, torch.Tensor):
        sid = sid_batch.long()
    elif isinstance(sid_batch, list):
        if len(sid_batch) == 0:
            raise ValueError("Empty sid batch.")
        if isinstance(sid_batch[0], torch.Tensor):
            sid = torch.stack([x.long() for x in sid_batch], dim=1)
        else:
            sid = torch.tensor(sid_batch, dtype=torch.long)
    else:
        sid = torch.tensor(sid_batch, dtype=torch.long)

    if sid.dim() == 1:
        sid = sid.unsqueeze(0)
    return sid.to(device)
