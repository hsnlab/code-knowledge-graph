import pickle
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit


EDGE_SPECS = [
    ("function_edges",                ("function", "calls",    "function"), "function", "function"),
    ("import_function_usage_edges",   ("import",   "used_by",  "function"), "import",   "function"),
    ("cluster_function_edges",        ("cluster",  "groups",   "function"), "cluster",  "function"),
    ("class_function_edges",          ("class",    "owns",     "function"), "class",    "function"),
    ("file_function_edges",           ("file",     "contains", "function"), "file",     "function"),
    ("dfg_function_edges",            ("dfg",      "flows_in", "function"), "dfg",      "function"),
    ("class_class_edges",             ("class",    "extends",  "class"),    "class",    "class"),
]


def load_knowledge_graph(pkl_path: str) -> dict:
    """Load the KnowledgeGraphBuilder output dict from a pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _embed_texts(
    texts: List[str],
    encoder: SentenceTransformer,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode a list of strings into an (N, dim) float tensor with the encoder."""
    embs = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return torch.tensor(embs, dtype=torch.float)


def _node_text(df: pd.DataFrame, columns: List[str]) -> List[str]:
    """Concatenate the listed string columns into a single space-separated text per row."""
    parts = []
    for col in columns:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        return [""] * len(df)
    return parts[0].str.cat(parts[1:], sep=" ").tolist()


def build_hetero_data(
    kg: dict,
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
    add_reverse_edges: bool = True,
) -> Tuple[HeteroData, Dict[str, Dict[int, int]]]:
    """Convert a knowledge-graph dict into a PyG HeteroData object.

    For each supported node type (function, import, cluster, class) it:
      1. Reindexes the original IDs to a contiguous 0..N-1 range and stores
         the {original_id: new_index} mapping in id_maps.
      2. Builds an informative text per node and embeds it with the chosen
         SentenceTransformer encoder.
      3. Loads each edge type listed in EDGE_SPECS, remapping endpoints with
         id_maps and dropping edges with unknown endpoints.
      4. Optionally adds a reversed copy of every edge type (rev_<rel>) so
         that message passing can flow both ways during training.

    Returns the populated HeteroData and the id_maps needed later to map
    embeddings back to original node IDs.
    """
    encoder = SentenceTransformer(encoder_model, device=device)
    data = HeteroData()
    id_maps: Dict[str, Dict[int, int]] = {}

    fn_df = kg["function_nodes"].reset_index(drop=True)
    id_maps["function"] = dict(zip(fn_df["ID"].astype(int), fn_df.index))
    data["function"].x = _embed_texts(
        _node_text(fn_df, ["combinedName", "docstring"]), encoder
    )

    if "import_nodes" in kg and not kg["import_nodes"].empty:
        imp_df = kg["import_nodes"].reset_index(drop=True)
        id_maps["import"] = dict(zip(imp_df["ID"].astype(int), imp_df.index))
        data["import"].x = _embed_texts(
            _node_text(imp_df, ["import_from", "import_name", "import_as_name"]),
            encoder,
        )

    if "cluster_nodes" in kg and not kg["cluster_nodes"].empty:
        cl_df = kg["cluster_nodes"].reset_index(drop=True)
        text_cols = [c for c in cl_df.columns if c != "ID" and cl_df[c].dtype == object]
        id_maps["cluster"] = dict(zip(cl_df["ID"].astype(int), cl_df.index))
        data["cluster"].x = _embed_texts(_node_text(cl_df, text_cols), encoder)

    if "class_nodes" in kg and not kg["class_nodes"].empty:
        cls_df = kg["class_nodes"].reset_index(drop=True)
        id_maps["class"] = dict(zip(cls_df["ID"].astype(int), cls_df.index))
        data["class"].x = _embed_texts(
            _node_text(cls_df, ["class_name"]), encoder
        )

    if "file_nodes" in kg and not kg["file_nodes"].empty:
        f_df = kg["file_nodes"].reset_index(drop=True)
        id_maps["file"] = dict(zip(f_df["ID"].astype(int), f_df.index))
        data["file"].x = _embed_texts(
            _node_text(f_df, ["path", "name"]), encoder
        )

    if "dfg_nodes" in kg and not kg["dfg_nodes"].empty:
        d_df = kg["dfg_nodes"].reset_index(drop=True)
        id_maps["dfg"] = dict(zip(d_df["ID"].astype(int), d_df.index))
        data["dfg"].x = _embed_texts(
            _node_text(d_df, ["name", "node_type", "code"]), encoder
        )

    for kg_key, edge_type, src_t, tgt_t in EDGE_SPECS:
        if kg_key not in kg or kg[kg_key].empty:
            continue
        if src_t not in id_maps or tgt_t not in id_maps:
            continue
        df = kg[kg_key]
        src_idx = df["source"].astype(int).map(id_maps[src_t])
        tgt_idx = df["target"].astype(int).map(id_maps[tgt_t])
        mask = src_idx.notna() & tgt_idx.notna()
        if mask.sum() == 0:
            continue
        edge_index = torch.tensor(
            [src_idx[mask].astype(int).tolist(), tgt_idx[mask].astype(int).tolist()],
            dtype=torch.long,
        )
        data[edge_type].edge_index = edge_index

    if add_reverse_edges:
        for et in list(data.edge_types):
            src, rel, tgt = et
            rev_et = (tgt, f"rev_{rel}", src)
            if rev_et in data.edge_types:
                continue
            data[rev_et].edge_index = data[et].edge_index.flip(0)

    return data, id_maps


def split_edges(
    data: HeteroData,
    supervision_edge: Tuple[str, str, str],
    num_val: float = 0.05,
    num_test: float = 0.10,
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """Split one edge type into train/val/test sets for link prediction.

    Other edge types remain unchanged and act as message-passing context.
    Negative samples are drawn at a 1:1 ratio with positive edges. If a
    reverse edge type exists for the supervised relation it is split jointly
    so that no validation/test edge leaks via its reverse copy.
    """
    src, rel, tgt = supervision_edge
    rev_edge = (tgt, f"rev_{rel}", src)
    rev_edge_types = [rev_edge] if rev_edge in data.edge_types else None

    transform = RandomLinkSplit(
        num_val=num_val,
        num_test=num_test,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=[supervision_edge],
        rev_edge_types=[rev_edge] if rev_edge_types else None,
    )
    return transform(data)
