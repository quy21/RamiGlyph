"""RamiGlyph dual-branch architecture and self-supervised objectives."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import ResGatedGraphConv, global_mean_pool
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import Adj

from .direction_attention import DirectionAwareAttention
from .egnn_layer import EGNNLayer
from .topo_encoder import TopologyEncoder


class GeometricGraphLayer(nn.Module):
    """Fuse graph convolution, EGNN, and direction-aware attention."""

    def __init__(
        self,
        channels: int,
        conv: Optional[nn.Module],
        heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.5,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        max_branch_level: int = 20,
    ):
        super().__init__()
        self.channels = channels
        self.conv = conv
        self.dropout = dropout

        self.egnn = EGNNLayer(
            hidden_dim=channels,
            edge_dim=channels,
            residual=False,
            normalize=True,
            coords_agg="mean",
        )

        self.attn = DirectionAwareAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=attn_dropout,
            max_level=max_branch_level,
        )

        self.gate = nn.Sequential(
            nn.Linear(channels * 3, channels), nn.ReLU(), nn.Linear(channels, 3)
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm4 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = self.norm1.forward.__code__.co_varnames
            self.norm_with_batch = "batch" in signature

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        branch_level: Optional[Tensor] = None,
        distance: Optional[Tensor] = None,
        branch_id: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        **kwargs
    ):
        """Return fused node features and updated coordinates."""

        if self.conv is not None:
            h_gcn = self.conv(x, edge_index, edge_attr=edge_attr)
            h_gcn = F.dropout(h_gcn, p=self.dropout, training=self.training)
            h_gcn = h_gcn + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h_gcn = self.norm1(h_gcn, batch=batch)
                else:
                    h_gcn = self.norm1(h_gcn)
        else:
            h_gcn = x

        h_egnn, pos_new = self.egnn(x, pos, edge_index, edge_attr)
        h_egnn = F.dropout(h_egnn, p=self.dropout, training=self.training)
        h_egnn = h_egnn + x
        if self.norm2 is not None:
            if self.norm_with_batch:
                h_egnn = self.norm2(h_egnn, batch=batch)
            else:
                h_egnn = self.norm2(h_egnn)

        h_attn = self.attn(
            x, batch, branch_level, distance=distance, branch_id=branch_id
        )
        h_attn = F.dropout(h_attn, p=self.dropout, training=self.training)
        h_attn = h_attn + x
        if self.norm3 is not None:
            if self.norm_with_batch:
                h_attn = self.norm3(h_attn, batch=batch)
            else:
                h_attn = self.norm3(h_attn)

        concat = torch.cat([h_gcn, h_egnn, h_attn], dim=-1)
        weights = F.softmax(self.gate(concat), dim=-1)
        out = (
            weights[:, 0:1] * h_gcn
            + weights[:, 1:2] * h_egnn
            + weights[:, 2:3] * h_attn
        )

        out = out + x
        out = out + self.mlp(out)

        if self.norm4 is not None:
            if self.norm_with_batch:
                out = self.norm4(out, batch=batch)
            else:
                out = self.norm4(out)

        return out, pos_new

    def reset_parameters(self):
        """Reset all submodules."""
        for module in self.gate:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if self.conv is not None:
            self.conv.reset_parameters()
        self.egnn.reset_parameters()
        self.attn.reset_parameters()
        for module in self.mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for norm in (self.norm1, self.norm2, self.norm3, self.norm4):
            if norm is not None:
                norm.reset_parameters()


class StructuralEncoder(nn.Module):
    """Encode structural morphology features with stacked graph layers."""

    def __init__(
        self,
        in_features: int,
        channels: int,
        pe_dim: int,
        num_layer: int,
        heads: int,
        dropout: float,
        attn_dropout: float,
        max_branch_level: int = 20,
    ):
        super().__init__()

        node_emb_dim = channels - pe_dim
        self.node_emb = nn.Linear(in_features, node_emb_dim)
        self.pe_lin = nn.Linear(20, pe_dim)
        self.pe_norm = nn.BatchNorm1d(20)
        self.edge_emb = nn.Linear(32, channels)

        self.max_branch_level = max_branch_level

        self.convs = nn.ModuleList()
        for _ in range(num_layer):
            local_gnn = ResGatedGraphConv(
                in_channels=channels,
                out_channels=channels,
                act=nn.ReLU(),
                edge_dim=channels,
            )
            layer = GeometricGraphLayer(
                channels,
                local_gnn,
                heads=heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                max_branch_level=max_branch_level,
            )
            self.convs.append(layer)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
        )

    def forward(
        self,
        x,
        pe,
        edge_index,
        edge_attr,
        batch,
        pos,
        branch_level=None,
        distance=None,
        branch_id=None,
    ):
        """Return graph-level structural embeddings."""
        pe = self.pe_norm(pe)
        pe = self.pe_lin(pe)
        node_emb = self.node_emb(x.squeeze(-1))

        x = torch.cat((node_emb, pe), dim=1)
        edge_types = edge_attr.long()
        one_hot = torch.zeros(
            edge_types.shape[0],
            32,
            device=edge_types.device,
            dtype=torch.float,
        )
        one_hot.scatter_(1, edge_types.unsqueeze(1), 1.0)
        edge_attr = self.edge_emb(one_hot)

        for conv in self.convs:
            x, pos = conv(
                x,
                edge_index,
                batch,
                pos=pos,
                branch_level=branch_level,
                distance=distance,
                branch_id=branch_id,
                edge_attr=edge_attr,
            )

        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x

    def reset_parameters(self):
        """Reset all encoder layers."""
        self.node_emb.reset_parameters()
        self.pe_lin.reset_parameters()
        self.pe_norm.reset_parameters()
        self.edge_emb.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for module in self.mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class DualBranchEncoder(nn.Module):
    """Combine structural and topological encoders."""

    def __init__(
        self,
        in_features: int,
        channels: int,
        pe_dim: int,
        num_layer: int,
        heads: int,
        dropout: float,
        attn_dropout: float,
        topo_channels: int,
        max_branch_level: int = 20,
    ):
        super().__init__()

        self.struct_encoder = StructuralEncoder(
            in_features=in_features,
            channels=channels,
            pe_dim=pe_dim,
            num_layer=num_layer,
            heads=heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            max_branch_level=max_branch_level,
        )

        self.topo_encoder = TopologyEncoder(dim_out=topo_channels)

        self.channels = channels
        self.topo_channels = topo_channels

    def forward(self, struct_input, topo_input):
        """Return structural and topological graph embeddings."""
        branch_level = getattr(struct_input, "branch_level", None)
        distance = getattr(struct_input, "distance", None)
        branch_id = getattr(struct_input, "branch_id", None)
        pos = getattr(struct_input, "pos", None)

        h_struct = self.struct_encoder(
            struct_input.x,
            struct_input.pe,
            struct_input.edge_index,
            struct_input.edge_attr,
            struct_input.batch,
            pos=pos,
            branch_level=branch_level,
            distance=distance,
            branch_id=branch_id,
        )

        h_topo = self.topo_encoder(topo_input)

        return h_struct, h_topo

    def reset_parameters(self):
        """Reset both encoder branches."""
        self.struct_encoder.reset_parameters()
        self.topo_encoder.reset_parameters()


class RamiGlyph(nn.Module):
    """Dual-branch SwAV model for morphology representation learning."""

    def __init__(self, encoder, nmb_prototypes, feat_dim_struct, feat_dim_topo):
        super().__init__()
        self.encoder = encoder
        self.feat_dim_struct = feat_dim_struct
        self.feat_dim_topo = feat_dim_topo

        self.prototypes_struct = nn.Linear(feat_dim_struct, nmb_prototypes, bias=False)

        self.prototypes_topo = nn.Linear(feat_dim_topo, nmb_prototypes, bias=False)

        self.projection = nn.Linear(feat_dim_struct + feat_dim_topo, feat_dim_struct)

        self._init_prototypes()

    def _init_prototypes(self):
        """Normalize prototype weights at initialization."""
        for proto in [self.prototypes_struct, self.prototypes_topo]:
            proto.weight.data = F.normalize(proto.weight.data, dim=1, p=2)

    def trainable_parameters(self):
        """Return all parameters optimized during training."""
        return list(self.parameters())

    @torch.no_grad()
    def normalize_prototypes(self):
        """Normalize prototype weights in place."""
        for proto in [self.prototypes_struct, self.prototypes_topo]:
            w = proto.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            proto.weight.copy_(w)

    def forward(self, struct_input, topo_input):
        """Return branch embeddings, prototype logits, and fused features."""

        h_struct, h_topo = self.encoder(struct_input, topo_input)

        embedding_struct = F.normalize(h_struct, dim=1, p=2, eps=1e-6)
        embedding_topo = F.normalize(h_topo, dim=1, p=2, eps=1e-6)

        output_struct = self.prototypes_struct(embedding_struct)
        output_topo = self.prototypes_topo(embedding_topo)

        fused = torch.cat([h_struct, h_topo], dim=1)
        fused_feature = self.projection(fused)

        return (
            embedding_struct,
            embedding_topo,
            output_struct,
            output_topo,
            fused_feature,
        )

    def get_fused_feature(self, struct_input, topo_input):
        """Return fused features for downstream evaluation."""
        h_struct, h_topo = self.encoder(struct_input, topo_input)
        fused = torch.cat([h_struct, h_topo], dim=1)
        fused_feature = self.projection(fused)
        return fused_feature


@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    """Compute balanced soft assignments with Sinkhorn normalization."""
    logits = out.float()
    assignments = torch.exp(logits / epsilon).t()
    batch_size = assignments.shape[1]
    num_prototypes = assignments.shape[0]

    total = torch.sum(assignments)
    assignments /= total + 1e-8

    for _ in range(sinkhorn_iterations):
        row_sums = torch.sum(assignments, dim=1, keepdim=True)
        assignments /= row_sums + 1e-8
        assignments /= num_prototypes

        column_sums = torch.sum(assignments, dim=0, keepdim=True)
        assignments /= column_sums + 1e-8
        assignments /= batch_size

    assignments *= batch_size
    return assignments.t()


def swav_loss(output1, output2, temperature=0.1, epsilon=0.05, sinkhorn_iterations=3):
    """Compute the swapped prediction loss for two augmented views."""

    with torch.no_grad():
        q1 = sinkhorn(output1.detach(), epsilon, sinkhorn_iterations)
        q2 = sinkhorn(output2.detach(), epsilon, sinkhorn_iterations)

    p1 = F.log_softmax(output1.float() / temperature, dim=1)
    p2 = F.log_softmax(output2.float() / temperature, dim=1)

    loss = -0.5 * (
        torch.mean(torch.sum(q1 * p2, dim=1)) + torch.mean(torch.sum(q2 * p1, dim=1))
    )

    return loss


def dual_branch_swav_loss(
    out_struct_1,
    out_struct_2,
    out_topo_1,
    out_topo_2,
    temperature=0.1,
    epsilon=0.05,
    sinkhorn_iterations=3,
    w_struct=0.5,
    w_topo=0.5,
):
    """Combine structural and topological SwAV losses."""
    loss_struct = swav_loss(
        out_struct_1, out_struct_2, temperature, epsilon, sinkhorn_iterations
    )
    loss_topo = swav_loss(
        out_topo_1, out_topo_2, temperature, epsilon, sinkhorn_iterations
    )

    total_loss = w_struct * loss_struct + w_topo * loss_topo

    return total_loss, loss_struct, loss_topo


Geometric_GG_layer = GeometricGraphLayer
Transformer_encoder = StructuralEncoder
DualBranch_Encoder = DualBranchEncoder
DualBranch_SwAV = RamiGlyph


def build_ramiglyph_model(config, device="cuda"):
    """Build RamiGlyph from a configuration dictionary."""
    model_cfg = config["model"]
    branch_level_cfg = config.get("branch_level", {})
    swav_cfg = config.get("swav", {})

    encoder = DualBranchEncoder(
        in_features=model_cfg.get("in_features", 9),
        channels=model_cfg["channels"],
        pe_dim=model_cfg["pe_dim"],
        num_layer=model_cfg["layers"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
        attn_dropout=model_cfg["attn_dropout"],
        topo_channels=model_cfg.get("topo_channels", model_cfg["channels"]),
        max_branch_level=branch_level_cfg.get("max_level", 20),
    )

    model = RamiGlyph(
        encoder=encoder,
        nmb_prototypes=swav_cfg.get("nmb_prototypes", 100),
        feat_dim_struct=model_cfg["channels"],
        feat_dim_topo=model_cfg.get("topo_channels", model_cfg["channels"]),
    ).to(device)

    return model


build_dual_branch_swav_model = build_ramiglyph_model
