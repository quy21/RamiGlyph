"""Direction-aware multi-head attention for tree-structured graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class DirectionAwareAttention(nn.Module):
    """Apply global attention with branch direction and distance biases."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_level: int = 20,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.max_level = max_level

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.centripetal_embedding = nn.Embedding(max_level + 1, num_heads)
        self.centripetal_weight = nn.Parameter(torch.zeros(num_heads))

        self.centrifugal_embedding = nn.Embedding(max_level + 1, num_heads)
        self.centrifugal_weight = nn.Parameter(torch.zeros(num_heads))

        self.intra_branch_embedding = nn.Embedding(max_level + 1, num_heads)
        self.intra_branch_weight = nn.Parameter(torch.zeros(num_heads))

        self.inter_branch_embedding = nn.Embedding(max_level + 1, num_heads)
        self.inter_branch_weight = nn.Parameter(torch.zeros(num_heads))

        self.distance_proj = nn.Linear(1, num_heads)
        self.distance_weight = nn.Parameter(torch.zeros(num_heads))

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize projections, embeddings, and directional weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

        nn.init.normal_(self.centripetal_embedding.weight, std=0.02)
        nn.init.normal_(self.centrifugal_embedding.weight, std=0.02)
        nn.init.normal_(self.intra_branch_embedding.weight, std=0.02)
        nn.init.normal_(self.inter_branch_embedding.weight, std=0.02)

        nn.init.constant_(self.centripetal_weight, 0.1)
        nn.init.constant_(self.centrifugal_weight, 0.1)
        nn.init.constant_(self.intra_branch_weight, 0.1)
        nn.init.constant_(self.inter_branch_weight, -0.1)

        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.zeros_(self.distance_proj.bias)
        nn.init.constant_(self.distance_weight, 0.1)

    def _compute_direction_bias(self, dense_lv, dense_bid, mask):
        """Return branch-aware attention biases with shape [B, H, N, N]."""
        lv_i = dense_lv.unsqueeze(2)  # [B, N, 1]
        lv_j = dense_lv.unsqueeze(1)  # [B, 1, N]
        level_diff = (lv_i - lv_j).long()  # [B, N, N]

        cp_mask = level_diff > 0  # centripetal
        cf_mask = level_diff < 0  # centrifugal
        lt_mask = level_diff == 0  # lateral

        bias = (
            self.centripetal_embedding(level_diff.clamp(0, self.max_level).long())
            * self.centripetal_weight
            * cp_mask.unsqueeze(-1).float()
        )
        del cp_mask

        bias = bias + (
            self.centrifugal_embedding((-level_diff).clamp(0, self.max_level).long())
            * self.centrifugal_weight
            * cf_mask.unsqueeze(-1).float()
        )
        del cf_mask, level_diff

        lv_i_idx = lv_i.clamp(0, self.max_level).long()  # [B, N, 1]
        intra_emb = (
            self.intra_branch_embedding(lv_i_idx) * self.intra_branch_weight
        )  # [B, N, 1, H]
        inter_emb = (
            self.inter_branch_embedding(lv_i_idx) * self.inter_branch_weight
        )  # [B, N, 1, H]
        del lv_i_idx

        if dense_bid is not None:
            same_branch = dense_bid.unsqueeze(2) == dense_bid.unsqueeze(1)  # [B, N, N]
            intra_mf = (lt_mask & same_branch).unsqueeze(-1).float()  # [B, N, N, 1]
            inter_mf = (lt_mask & ~same_branch).unsqueeze(-1).float()  # [B, N, N, 1]
            del same_branch
        else:
            intra_mf = lt_mask.unsqueeze(-1).float()  # [B, N, N, 1]
            inter_mf = torch.zeros_like(intra_mf)
        del lt_mask

        # Broadcast [B, N, 1, H] against [B, N, N, 1].
        bias = bias + intra_emb * intra_mf
        bias = bias + inter_emb * inter_mf
        del intra_emb, inter_emb, intra_mf, inter_mf

        valid = mask.unsqueeze(2) & mask.unsqueeze(1)
        bias = bias * valid.unsqueeze(-1).float()
        return bias.permute(0, 3, 1, 2).contiguous()

    def _compute_distance_bias(self, dense_dist):
        """Return pairwise distance biases with shape [B, H, N, N]."""
        dist_diff = torch.abs(
            dense_dist.unsqueeze(2) - dense_dist.unsqueeze(1)
        )  # [B, N, N]
        dist_feat = torch.log1p(dist_diff).unsqueeze(-1)  # [B, N, N, 1]
        dist_bias = self.distance_proj(dist_feat) * self.distance_weight  # [B, N, N, H]
        return dist_bias.permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]

    def forward(self, x, batch, branch_level, distance=None, branch_id=None):
        """Encode sparse node features and return them in sparse order."""
        device = x.device

        dense_x, mask = to_dense_batch(x, batch)  # [B, N, embed_dim], [B, N]
        batch_size, max_nodes, _ = dense_x.shape

        def _to_dense(feat):
            dense = torch.zeros(
                batch_size,
                max_nodes,
                device=device,
                dtype=feat.dtype,
            )
            dense[mask] = feat
            return dense

        dense_lv = _to_dense(branch_level)
        dense_bid = _to_dense(branch_id) if branch_id is not None else None
        dense_dist = _to_dense(distance) if distance is not None else None

        query = (
            self.q_proj(dense_x)
            .view(batch_size, max_nodes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.k_proj(dense_x)
            .view(batch_size, max_nodes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.v_proj(dense_x)
            .view(batch_size, max_nodes, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if branch_level is not None:
            attn_scores = attn_scores + self._compute_direction_bias(
                dense_lv, dense_bid, mask
            )

        if distance is not None:
            attn_scores = attn_scores + self._compute_distance_bias(dense_dist)

        attn_mask = (~mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(nan=0.0)
        attn_weights = self.dropout(attn_weights)
        attn_weights = attn_weights.masked_fill(attn_mask, 0.0)

        attn_out = torch.matmul(attn_weights, value)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, max_nodes, self.embed_dim)
        )
        attn_out = self.out_proj(attn_out)

        return attn_out[mask]  # [N_total, embed_dim]

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self._reset_parameters()
