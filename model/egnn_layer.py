"""E(n)-equivariant message passing for morphology graphs."""

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean


class EGNNLayer(nn.Module):
    """Update node features and coordinates with equivariant messages."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        normalize: bool = False,
        coords_agg: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.residual = residual
        self.normalize = normalize
        self.coords_agg = coords_agg

        edge_input_dim = hidden_dim * 2 + 1  # h_i, h_j, dist²
        if edge_dim > 0:
            edge_input_dim += edge_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.attention_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, h, pos, edge_index, edge_attr=None):
        """Return updated node features and coordinates."""
        src, dst = edge_index  # src -> dst

        rel_pos = pos[src] - pos[dst]
        dist_sq = (rel_pos**2).sum(dim=-1, keepdim=True)  # [E, 1]

        edge_input = [h[src], h[dst], dist_sq]
        if edge_attr is not None and self.edge_dim > 0:
            edge_input.append(edge_attr)

        edge_input = torch.cat(edge_input, dim=-1)
        m_ij = self.edge_mlp(edge_input)  # [E, hidden_dim]

        attn = self.attention_mlp(m_ij)  # [E, 1]
        m_ij = m_ij * attn

        coord_weight = self.coord_mlp(m_ij)

        if self.normalize:
            rel_pos_norm = rel_pos / (dist_sq.sqrt() + 1e-8)
            coord_diff = rel_pos_norm * coord_weight
        else:
            coord_diff = rel_pos * coord_weight

        if self.coords_agg == "mean":
            coord_update = scatter_mean(coord_diff, dst, dim=0, dim_size=h.size(0))
        else:
            coord_update = scatter_add(coord_diff, dst, dim=0, dim_size=h.size(0))

        pos_out = pos + coord_update  # [N, 3]

        m_agg = scatter_add(m_ij, dst, dim=0, dim_size=h.size(0))  # [N, hidden_dim]

        node_input = torch.cat([h, m_agg], dim=-1)
        h_out = self.node_mlp(node_input)  # [N, hidden_dim]

        if self.residual:
            h_out = h + h_out

        return h_out, pos_out

    def reset_parameters(self):
        """Reset all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


EGNN_Layer = EGNNLayer
