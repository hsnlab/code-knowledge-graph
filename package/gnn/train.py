import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .dataset import build_hetero_data, load_knowledge_graph, split_edges
from .model import HeteroGNN, LinkPredictor


def _supervision_pair(data, edge_type):
    """Return (src_type, tgt_type, edge_label_index, edge_label) for the supervised edge."""
    src_t, _, tgt_t = edge_type
    eli = data[edge_type].edge_label_index
    el = data[edge_type].edge_label
    return src_t, tgt_t, eli, el


def train_step(model, predictor, optimizer, train_data, edge_type):
    """Run one optimization step.

    Forward pass through the GNN, score the supervised positive and negative
    edges with the link predictor, optimize binary cross-entropy.
    """
    model.train()
    predictor.train()
    optimizer.zero_grad()

    z_dict = model(train_data.x_dict, train_data.edge_index_dict)
    src_t, tgt_t, eli, el = _supervision_pair(train_data, edge_type)
    logits = predictor(z_dict[src_t][eli[0]], z_dict[tgt_t][eli[1]])
    loss = F.binary_cross_entropy_with_logits(logits, el.float())

    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model, predictor, data, edge_type):
    """Compute ROC-AUC on the supervised positive/negative edges of `data`."""
    model.eval()
    predictor.eval()

    z_dict = model(data.x_dict, data.edge_index_dict)
    src_t, tgt_t, eli, el = _supervision_pair(data, edge_type)
    logits = predictor(z_dict[src_t][eli[0]], z_dict[tgt_t][eli[1]])
    probs = torch.sigmoid(logits).cpu().numpy()
    return float(roc_auc_score(el.cpu().numpy(), probs))


def main(
    kg_pkl: str,
    out_dir: str = "./gnn_out",
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden: int = 256,
    out_dim: int = 256,
    layers: int = 2,
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
):
    """End-to-end training entry point.

    Loads the KG pickle, builds HeteroData with text-derived initial features,
    splits the function-calls edge type into train/val/test, trains the GNN
    with self-supervised link prediction, and saves the best-AUC checkpoint.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading knowledge graph from {kg_pkl}")
    kg = load_knowledge_graph(kg_pkl)

    print(f"[2/4] Building HeteroData (encoder={encoder_model}, device={device})")
    data, id_maps = build_hetero_data(kg, encoder_model=encoder_model, device=device)
    print(data)

    supervision_edge = ("function", "calls", "function")
    print(f"[3/4] Splitting edges for supervision: {supervision_edge}")
    train_data, val_data, test_data = split_edges(data, supervision_edge)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    in_channels_dict = {nt: data[nt].x.size(-1) for nt in data.node_types}
    model = HeteroGNN(
        node_types=list(data.node_types),
        edge_types=list(data.edge_types),
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden,
        out_channels=out_dim,
        num_layers=layers,
    ).to(device)
    predictor = LinkPredictor().to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    print(f"[4/4] Training for {epochs} epochs")
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        loss = train_step(model, predictor, optimizer, train_data, supervision_edge)
        val_auc = evaluate(model, predictor, val_data, supervision_edge)
        marker = ""
        if val_auc > best_val:
            best_val = val_auc
            marker = " *"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "predictor_state": predictor.state_dict(),
                    "id_maps": id_maps,
                    "config": {
                        "node_types": list(data.node_types),
                        "edge_types": list(data.edge_types),
                        "in_channels_dict": in_channels_dict,
                        "hidden": hidden,
                        "out_dim": out_dim,
                        "layers": layers,
                        "encoder_model": encoder_model,
                    },
                },
                out_path / "best_model.pt",
            )
        print(f"Epoch {epoch:03d} | loss={loss:.4f} | val_auc={val_auc:.4f}{marker}")

    test_auc = evaluate(model, predictor, test_data, supervision_edge)
    print(f"Best val AUC: {best_val:.4f} | Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_pkl", required=True)
    parser.add_argument("--out_dir", default="./gnn_out")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--encoder_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(**vars(args))
