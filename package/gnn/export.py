import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import torch
from neo4j import GraphDatabase

from .dataset import build_hetero_data, load_knowledge_graph
from .model import HeteroGNN


@torch.no_grad()
def compute_embeddings(
    checkpoint_path: str,
    kg_pkl: str,
    device: str = "cpu",
) -> Dict[int, List[float]]:
    """Run a single full-graph forward pass with the trained model.

    Rebuilds HeteroData from the KG pickle, restores the checkpointed model,
    and returns a {function_id: embedding_list} dict mapping each original
    FUNCTION node id to its refined GNN embedding.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    kg = load_knowledge_graph(kg_pkl)
    data, id_maps = build_hetero_data(
        kg, encoder_model=cfg.get("encoder_model"), device=device
    )
    data = data.to(device)

    model = HeteroGNN(
        node_types=cfg["node_types"],
        edge_types=cfg["edge_types"],
        in_channels_dict=cfg["in_channels_dict"],
        hidden_channels=cfg["hidden"],
        out_channels=cfg["out_dim"],
        num_layers=cfg["layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    z_dict = model(data.x_dict, data.edge_index_dict)
    fn_emb = z_dict["function"].cpu().numpy()

    inv_map = {idx: fid for fid, idx in id_maps["function"].items()}
    return {int(inv_map[i]): fn_emb[i].tolist() for i in range(fn_emb.shape[0])}


def export_to_pickle(embeddings: Dict[int, List[float]], out_path: str) -> None:
    """Serialize the {function_id: embedding} dict to a pickle file."""
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)


def export_to_neo4j(
    embeddings: Dict[int, List[float]],
    neo4j_uri: str,
    user: str,
    password: str,
    batch_size: int = 500,
) -> None:
    """Write each embedding as a `gnn_embedding` property on the matching FUNCTION node.

    Uses batched UNWIND queries that look up nodes by their `global_id`
    (FUNCTION:<id>) which is the convention set by KnowledgeGraphBuilder.
    """
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    items = list(embeddings.items())
    with driver.session() as session:
        for i in range(0, len(items), batch_size):
            batch = [
                {"fid": fid, "emb": emb}
                for fid, emb in items[i : i + batch_size]
            ]
            session.run(
                """
                UNWIND $batch AS row
                MATCH (n:FUNCTION {global_id: 'FUNCTION:' + toString(row.fid)})
                SET n.gnn_embedding = row.emb
                """,
                batch=batch,
            )
    driver.close()


def main(
    checkpoint: str,
    kg_pkl: str,
    output: str = "gnn_embeddings.pkl",
    neo4j_uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    device: str = "cpu",
):
    """End-to-end export entry point: compute embeddings, save pickle, optionally update Neo4j."""
    print(f"[1/3] Computing embeddings from {checkpoint}")
    embs = compute_embeddings(checkpoint, kg_pkl, device=device)
    print(f"      {len(embs)} function embeddings computed.")

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[2/3] Saving pickle to {out}")
        export_to_pickle(embs, str(out))

    if neo4j_uri:
        print(f"[3/3] Writing to Neo4j at {neo4j_uri}")
        export_to_neo4j(embs, neo4j_uri, user, password)
    else:
        print("[3/3] Neo4j URI not provided, skipping DB write.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kg_pkl", required=True)
    parser.add_argument("--output", default="gnn_embeddings.pkl")
    parser.add_argument("--neo4j_uri", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    main(**vars(args))
