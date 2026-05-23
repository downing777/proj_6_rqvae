from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .sid_prefix import SidPrefixConfig, SidPrefixEncoder


@dataclass
class SidModelLoadConfig:
    base_model_name_or_path: str
    sid_dims: tuple
    sid_embed_dim: int = 64
    num_virtual_tokens: int = 16
    num_basis_tokens: int = 64
    dropout: float = 0.1


class SidConditionedCausalLM(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, sid_prefix: SidPrefixEncoder):
        super().__init__()
        self.base_model = base_model
        self.sid_prefix = sid_prefix
        self.hidden_size = base_model.config.hidden_size

    def freeze_backbone(self) -> None:
        for p in self.base_model.parameters():
            p.requires_grad = False

    def trainable_parameters_report(self) -> Dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    def build_inputs_with_sid(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix = self.sid_prefix(sid).to(dtype=input_embeds.dtype)
        bsz, prefix_len, _ = prefix.shape
        prefix_mask = torch.ones((bsz, prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
        full_embeds = torch.cat([prefix, input_embeds], dim=1)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return {"inputs_embeds": full_embeds, "attention_mask": full_mask}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sid: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        packed = self.build_inputs_with_sid(input_ids=input_ids, attention_mask=attention_mask, sid=sid)
        if labels is not None:
            prefix_len = packed["inputs_embeds"].size(1) - labels.size(1)
            ignore_labels = torch.full(
                (labels.size(0), prefix_len), -100, device=labels.device, dtype=labels.dtype
            )
            packed["labels"] = torch.cat([ignore_labels, labels], dim=1)
        packed.update(kwargs)
        return self.base_model(**packed)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sid: torch.Tensor,
        max_new_tokens: int = 24,
        num_beams: int = 1,
        temperature: float = 0.8,
        suppress_tokens: list = None,
        eos_token_id=None,
    ) -> torch.Tensor:
        if num_beams != 1:
            raise NotImplementedError("Current SID wrapper supports num_beams=1 only.")

        generated = input_ids
        mask = attention_mask
        # EOS 解析优先级:
        #   1) 调用方显式传入 (推理脚本必须传 tokenizer.eos_token_id, 跟训练拼的 eos 对齐)
        #   2) 顶层 config.eos_token_id
        #   3) config.text_config.eos_token_id (兼容多模态/组合 config, e.g. Qwen3.5-VL)
        # 注意: model.config.eos_token_id 和 tokenizer.eos_token_id 在 Qwen 系常常不一致
        #       (base EOS vs chat EOS), 推理必须用 tokenizer 那一个, 否则 break 永远不触发。
        if eos_token_id is None:
            cfg = self.base_model.config
            eos_token_id = getattr(cfg, "eos_token_id", None)
            if eos_token_id is None:
                text_cfg = getattr(cfg, "text_config", None)
                if text_cfg is not None:
                    eos_token_id = getattr(text_cfg, "eos_token_id", None)
        # 支持 int 或 list[int] (生成多个候选 EOS 都触发停止)
        if isinstance(eos_token_id, int):
            eos_set = {eos_token_id}
        elif eos_token_id is None:
            eos_set = set()
        else:
            eos_set = {int(x) for x in eos_token_id}

        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=generated,
                attention_mask=mask,
                sid=sid,
            )
            next_logits = out.logits[:, -1, :]
            if suppress_tokens:
                next_logits[:, suppress_tokens] = float("-inf")
            if temperature > 0:
                probs = torch.softmax(next_logits / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
            append_mask = torch.ones(
                (mask.size(0), 1), dtype=mask.dtype, device=mask.device
            )
            mask = torch.cat([mask, append_mask], dim=1)

            if eos_set:
                nt = next_token.squeeze(-1)
                # 整个 batch 都命中任一 EOS 才停
                hit = torch.zeros_like(nt, dtype=torch.bool)
                for eid in eos_set:
                    hit |= nt == eid
                if torch.all(hit):
                    break
        return generated


def _patch_kernels_trust():
    """Allow FP8 kernels to load without Hub connectivity check."""
    try:
        import kernels.utils as _ku
        _ku._check_trust_remote_code = lambda repo_id, trust: None
    except ImportError:
        pass

def build_sid_model(load_config: SidModelLoadConfig, device: str) -> SidConditionedCausalLM:
    _patch_kernels_trust()
    base_model = AutoModelForCausalLM.from_pretrained(
        load_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    sid_prefix = SidPrefixEncoder(
        SidPrefixConfig(
            sid_dims=load_config.sid_dims,
            sid_embed_dim=load_config.sid_embed_dim,
            num_virtual_tokens=load_config.num_virtual_tokens,
            hidden_size=base_model.config.hidden_size,
            num_basis_tokens=load_config.num_basis_tokens,
            dropout=load_config.dropout,
        )
    )
    model = SidConditionedCausalLM(base_model=base_model, sid_prefix=sid_prefix).to(device)
    return model
