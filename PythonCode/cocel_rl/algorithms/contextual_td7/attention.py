import torch
from torch import nn


class DirectionalCrossAttention(nn.Module):
    """One-query cross-attention with explicit pre/post normalization."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.key_value_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        center_embedding: torch.Tensor,
        neighbor_embedding: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.query_norm(center_embedding).unsqueeze(1)
        key_value = self.key_value_norm(neighbor_embedding)
        output, weights = self.attention(
            query,
            key_value,
            key_value,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        context = self.output_norm(center_embedding + output[:, 0, :])
        return context, weights
