import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


import re as _re

PROMPT_TEMPLATE = (
    "You are a product title generator.\n"
    "Product info: {context}\n"
    "Generate a short, compelling English product title for the target SID user group:"
)


def strip_thinking_tags(text: str) -> str:
    """Remove  blocks and other assistant/role markers from generated text."""
    text = _re.sub(r"<think>[\s\S]*?</think>", "", text)
    text = _re.sub(r"<think>[\s\S]*$", "", text)  # unclosed <think>
    text = _re.sub(r"^assistant\s*", "", text, flags=_re.IGNORECASE)
    return text.strip()


def load_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def truncate_context_in_rows(
    rows: List[Dict[str, object]], max_context_chars: int
) -> List[Dict[str, object]]:
    """
    过长 context 在 token 化时易占满 max_length, 被截断后没有「目标标题」可监督, loss=nan.
    仅截断 `context` 字符串: 保尾部(评论/要点往往在末段), 见训练脚本 --max-context-chars。
    """
    if max_context_chars <= 0:
        return rows
    out: List[Dict[str, object]] = []
    for r in rows:
        c = r.get("context", "")
        if not isinstance(c, str) or len(c) <= max_context_chars:
            out.append(r)
            continue
        rc = {**r, "context": c[-max_context_chars:]}
        out.append(rc)
    return out


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
    texts = [f"{p}\nTitle: {t}{eos}" for p, t in zip(prompts, targets)]
    prompt_only = [f"{p}\nTitle: " for p in prompts]

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

        # 关键: SidConditionedCausalLM.forward 在 inputs_embeds 前面拼了 prefix_len 个软前缀,
        # 因此 out.logits 形状是 [B, prefix_len + K, V] 而 labels 仍是 [B, K]。
        # 必须把 labels 也补 prefix_len 个 -100 再 shift, 否则 gather 会悄悄只用 logits 的前 K-1
        # 个位置 (大部分落在 prefix 区间), 等价于把每个 target token 错位到 prefix_len 个位置之外
        # 的 logit 上去查概率, 给 DPO 喂的是噪声梯度。
        prefix_len = out.logits.size(1) - labels.size(1)
        if prefix_len > 0:
            pad = torch.full(
                (labels.size(0), prefix_len), -100,
                device=labels.device, dtype=labels.dtype,
            )
            full_labels = torch.cat([pad, labels], dim=1)        # [B, prefix_len + K]
        else:
            full_labels = labels

        shift_logits = out.logits[:, :-1, :]                     # [B, prefix_len + K - 1, V]
        shift_labels = full_labels[:, 1:]                        # [B, prefix_len + K - 1]
        valid = shift_labels.ne(-100)
        token_logp = torch.gather(
            shift_logits.log_softmax(-1),
            dim=-1,
            index=shift_labels.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)
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
