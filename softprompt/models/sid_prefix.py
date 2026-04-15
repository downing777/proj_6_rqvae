from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class SidPrefixConfig:
    sid_dims: Sequence[int]
    sid_embed_dim: int
    num_virtual_tokens: int
    hidden_size: int
    num_basis_tokens: int = 64
    dropout: float = 0.1


class SidPrefixEncoder(nn.Module):
    """
    Encode discrete SID tuples into multiple virtual prefix tokens.

    This implements a basis-mixing variant of:
      SID -> P -> token_basis -> prefix_tokens
    where each virtual token is a weighted sum of shared basis vectors.
    """

    def __init__(self, config: SidPrefixConfig) -> None:
        super().__init__()
        if len(config.sid_dims) == 0:
            raise ValueError("sid_dims must not be empty.")
        self.config = config

        self.sid_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, config.sid_embed_dim) for cardinality in config.sid_dims]
        )
        joined_dim = len(config.sid_dims) * config.sid_embed_dim
        self.sid_projector = nn.Sequential(
            nn.Linear(joined_dim, joined_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.weight_head = nn.Linear(
            joined_dim, config.num_virtual_tokens * config.num_basis_tokens
        )
        self.token_basis = nn.Parameter(
            torch.randn(config.num_basis_tokens, config.hidden_size) * 0.02
        )
        self.prefix_pos = nn.Parameter(
            torch.randn(config.num_virtual_tokens, config.hidden_size) * 0.01
        )
        self.out_norm = nn.LayerNorm(config.hidden_size)
        self.out_drop = nn.Dropout(config.dropout)

    def forward(self, sid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sid: [batch_size, sid_levels] integer SID tensor.

        Returns:
            prefix: [batch_size, num_virtual_tokens, hidden_size]
        """
        if sid.dim() != 2:
            raise ValueError(f"sid should be rank-2 [B, L], got shape={tuple(sid.shape)}")
        if sid.size(1) != len(self.sid_embeddings):
            raise ValueError(
                f"sid level mismatch: expected {len(self.sid_embeddings)}, got {sid.size(1)}"
            )

        embeds = []
        for level, emb in enumerate(self.sid_embeddings):
            embeds.append(emb(sid[:, level]))
        sid_hidden = torch.cat(embeds, dim=-1)
        sid_hidden = self.sid_projector(sid_hidden)

        raw_weights = self.weight_head(sid_hidden)
        raw_weights = raw_weights.view(
            sid.size(0), self.config.num_virtual_tokens, self.config.num_basis_tokens
        )
        weights = torch.softmax(raw_weights, dim=-1)
        prefix = torch.einsum("bmk,kh->bmh", weights, self.token_basis)
        prefix = prefix + self.prefix_pos.unsqueeze(0)
        return self.out_drop(self.out_norm(prefix))
