from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNN(nn.Module):
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Build a heterogeneous GraphSAGE encoder.

        Each node type gets its own input/output linear projection so that all
        types share a common hidden dimension. HeteroConv applies one SAGEConv
        per edge type per layer and aggregates the per-type messages with mean.
        """
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels_dict[nt], hidden_channels)
            for nt in node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {et: SAGEConv(hidden_channels, hidden_channels) for et in edge_types},
                aggr="mean",
            )
            self.convs.append(conv)

        self.output_proj = nn.ModuleDict({
            nt: nn.Linear(hidden_channels, out_channels)
            for nt in node_types
        })

    def forward(self, x_dict, edge_index_dict):
        """Run a forward pass over the heterogeneous graph.

        Projects all node features into the shared hidden space, runs the
        stack of HeteroConv layers (ReLU + dropout between layers) and applies
        the per-type output projection. Returns refined embeddings shaped
        (num_nodes, out_channels) per node type.
        """
        x_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}
            x_dict = {
                nt: F.dropout(x, p=self.dropout, training=self.training)
                for nt, x in x_dict.items()
            }

        return {nt: self.output_proj[nt](x) for nt, x in x_dict.items()}


class LinkPredictor(nn.Module):
    def forward(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        """Score (src, tgt) node pairs by dot product. Returns raw logits."""
        return (z_src * z_tgt).sum(dim=-1)
