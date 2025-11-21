# =========================
# 0) IMPORTOK + ALAP BEÁLLÍTÁS
# =========================
import os, json, random, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path

# <<< CUDA allocator: ezt a torch import ELŐTT kell beállítani!
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
...

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")      # NVIDIA, vagy PyTorch-ROCm build AMD-re (Linux)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")       # Intel GPU (IPEX)
    try:
        import torch_directml            # Windows DirectML
        return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")

device = pick_device()
print("device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# =========================
# 1) PARAMÉTEREK
# =========================
SELECT_DATASET = 'draper_hf'     # 'code_x_glue' | 'draper_hf'
LANG = 'cpp'                     # 'c' | 'cpp'
MAX_SAMPLES = 20000              # 0 → mind (óvatosan RAM/VRAM miatt)
BATCH_TRAIN, BATCH_EVAL = 64, 128
EPOCHS_GGNN, EPOCHS_GINE = 30, 20

# one-hot hash bucket a levelek szövegéhez (normalizálás miatt lehet kicsi)
TOK_DIM = 128                    # korábban 1024; normalizálással bőven elég 64–256
TOK_SENTINEL = TOK_DIM           # üres/nem-levél → sentinel id

EDGE_TYPES = {'parent':0, 'next_sibling':1, 'next_token':2}

# =========================
# 2) TREE-SITTER INIT (C/C++)
# =========================
from tree_sitter import Language, Parser
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

TS_LANG = Language(tsc.language()) if LANG.lower()=='c' else Language(tscpp.language())
parser = Parser(TS_LANG)
print('Tree-Sitter OK, LANG =', LANG)

# =========================
# 3) ADATBETÖLTÉS + CÉL OSZLOPOK
# =========================
def load_any_dataset(select: str):
    if select == 'code_x_glue':
        ds = load_dataset("google/code_x_glue_cc_defect_detection")
        rows = []
        for split in ["train", "validation", "test"]:
            for ex in ds[split]:
                rows.append({
                    "code":  ex["func"],
                    "label": int(ex["target"]),
                })
        return pd.DataFrame(rows)

    elif select == 'draper_hf':
        ds = load_dataset("claudios/Draper")
        rows = []
        for split in ["train", "validation", "test"]:
            for ex in ds[split]:
                rows.append({
                    "code":  ex["functionSource"],
                    "label": int(ex["combine"]),
                })
        return pd.DataFrame(rows)

    else:
        raise ValueError(select)


raw_df = load_any_dataset(SELECT_DATASET)

# --- Kiegyensúlyozás: kb. 10% pozitív (összes pozitív megtartása, negatívakból mintavétel) ---
def make_about_10pct_pos(df: pd.DataFrame, seed: int = SEED, neg_per_pos: int = 9) -> pd.DataFrame:
    df = df.copy()
    df['label'] = df['label'].astype(int)
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    if len(df_pos) == 0:
        raise ValueError("Nincs pozitív minta a datasetben, nem lehet kiegyensúlyozni.")
    target_neg = min(len(df_neg), neg_per_pos * len(df_pos))
    df_neg_sampled = df_neg.sample(target_neg, random_state=seed)
    df_bal = pd.concat([df_pos, df_neg_sampled], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_bal

raw_df = make_about_10pct_pos(raw_df, seed=SEED, neg_per_pos=3)

if MAX_SAMPLES > 0 and len(raw_df) > MAX_SAMPLES:
    raw_df = raw_df.sample(MAX_SAMPLES, random_state=SEED).reset_index(drop=True)

print("Dataset shape (kiegyensúlyozott, max mintaszám után):", raw_df.shape)
print(raw_df['label'].value_counts())

# --- Train/Val/Test split ---
from sklearn.model_selection import train_test_split
# =========================
# FIX VALIDATION (2000) + TEST (2000) SPLIT — STRATIFIED
# =========================

VAL_SIZE = 2000
TEST_SIZE = 2000

# véletlen keverés (a balanszolt raw_df-en)
raw_df = raw_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

# --- stratifikált validációs mintavétel ---
val_df = raw_df.groupby("label", group_keys=False).apply(
    lambda x: x.sample(int(VAL_SIZE * len(x) / len(raw_df)), random_state=SEED)
)

remaining = raw_df.drop(val_df.index)

# --- stratifikált teszt mintavétel ---
test_df = remaining.groupby("label", group_keys=False).apply(
    lambda x: x.sample(int(TEST_SIZE * len(x) / len(remaining)), random_state=SEED)
)

train_df = remaining.drop(test_df.index)

print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))


# =========================
# 4) KÓD NORMALIZÁLÁS (FORMATTING + KOMMENT ELTÁVOLÍTÁS)
# =========================
import re, textwrap

def remove_comments_and_strings(code: str) -> str:
    if code is None:
        return ""
    code = str(code)
    # C/C++ kommentek + string literálok egyszerű eltávolítása
    pattern = r"""
        //.*?$           |   # C++ egy soros komment
        /\*.*?\*/        |   # C több soros komment
        "([^"\\]|\\.)*"  |   # string literál
        '([^'\\]|\\.)*'      # char literál
    """
    return re.sub(pattern, " ", code, flags=re.MULTILINE | re.DOTALL | re.VERBOSE)

def normalize_code(code: str) -> str:
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = code.split("\n")
    lines = [l.rstrip() for l in lines if l.strip() != ""]
    code = "\n".join(lines)
    code = remove_comments_and_strings(code)
    code = re.sub(r"\s+", " ", code)
    return code.strip()

train_df["code_norm"] = train_df["code"].astype(str).apply(normalize_code)
val_df["code_norm"]   = val_df["code"].astype(str).apply(normalize_code)
test_df["code_norm"]  = test_df["code"].astype(str).apply(normalize_code)

print("Norm kód minta:\n", train_df["code_norm"].iloc[0][:200])

# =========================
# 5) STRING → AUGMENTED AST (Tree-Sitter segítségével)
# =========================
@dataclass
class ASTNode:
    id: int
    type: str
    text: str
    depth: int
    is_leaf: int

@dataclass
class ASTGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, str]]
    label: int
    raw: str

def build_augmented_ast(code: str) -> ASTGraph:
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node
    nodes: List[Dict[str, Any]] = []
    edges: List[Tuple[int, int, str]] = []
    nid = 0

    def walk(node, parent_id=None, last_sib=None, depth=0):
        nonlocal nid
        cur_id = nid
        nid += 1
        text = code[node.start_byte:node.end_byte]
        is_leaf = 1 if len(node.children) == 0 else 0
        nodes.append({
            "id": cur_id,
            "type": node.type,
            "text": text,
            "depth": depth,
            "is_leaf": is_leaf,
        })
        if parent_id is not None:
            edges.append((parent_id, cur_id, "parent"))
        prev_child_id = None
        for ch in node.children:
            child_id_before = nid
            walk(ch, cur_id, prev_child_id, depth+1)
            if prev_child_id is not None:
                edges.append((prev_child_id, child_id_before, "next_sibling"))
            prev_child_id = child_id_before

    walk(root, parent_id=None, last_sib=None, depth=0)

    # next_token élek: a "levél" node-ok sorrendje a forrásban
    leaf_ids = [n["id"] for n in nodes if n["is_leaf"] == 1]
    for i in range(len(leaf_ids)-1):
        edges.append((leaf_ids[i], leaf_ids[i+1], "next_token"))

    return ASTGraph(nodes=nodes, edges=edges, label=0, raw=code)

def df_to_ast_graphs(df: pd.DataFrame) -> List[ASTGraph]:
    graphs = []
    for _, row in df.iterrows():
        g = build_augmented_ast(row["code_norm"])
        g.label = int(row["label"])
        graphs.append(g)
    return graphs

print("AST build a train első 3 mintára...")
sample_graphs = df_to_ast_graphs(train_df.iloc[:3])
print("Első gráf: #nodes =", len(sample_graphs[0].nodes), "| #edges =", len(sample_graphs[0].edges))

# =========================
# 7) PyG KONVERZIÓ + VOCAB
# =========================
from torch_geometric.data import Data, DataLoader

class Vocab:
    def __init__(self): self.map = {}
    def id(self, k):
        if k not in self.map: self.map[k] = len(self.map)
        return self.map[k]
    def size(self): return len(self.map)

type_vocab = Vocab()

def _hash_bucket(s: str, D: int = TOK_DIM) -> int:
    if not s or not s.strip(): return TOK_SENTINEL
    h = hashlib.md5(s.strip().encode('utf8')).hexdigest()
    return int(h, 16) % D

def astgraph_to_pyg(graphs: List[ASTGraph]) -> List[Data]:
    pyg = []
    for g in graphs:
        node_types = [type_vocab.id(n["type"]) for n in g.nodes]
        node_texts = [n["text"] for n in g.nodes]
        depths = [n["depth"] for n in g.nodes]
        is_leaf = [n["is_leaf"] for n in g.nodes]

        x_type = torch.tensor(node_types, dtype=torch.long)
        x_tok = torch.tensor([_hash_bucket(t, TOK_DIM) for t in node_texts], dtype=torch.long)
        x_small = torch.tensor(
            np.stack([
                np.array(depths, dtype=np.float32),
                np.array(is_leaf, dtype=np.float32),
            ], axis=-1),
            dtype=torch.float,
        )

        if len(g.edges) == 0:
            edge_index = torch.empty((2,0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
        else:
            src = [s for s,_,_ in g.edges]; dst = [d for _,d,_ in g.edges]
            et  = [EDGE_TYPES[t] for *_,t in g.edges]
            edge_index = torch.tensor([src,dst], dtype=torch.long)
            edge_type  = torch.tensor(et, dtype=torch.long)

        data = Data(edge_index=edge_index, y=torch.tensor([g.label], dtype=torch.long))
        data.edge_type = edge_type
        data.x_type = x_type
        data.x_tok = x_tok
        data.x_small = x_small
        data.x = x_type.clone()  # kompatibilitás miatt
        pyg.append(data)
    return pyg

train_graphs = df_to_ast_graphs(train_df)
val_graphs   = df_to_ast_graphs(val_df)
test_graphs  = df_to_ast_graphs(test_df)

pyg_train = astgraph_to_pyg(train_graphs)
pyg_val   = astgraph_to_pyg(val_graphs)
pyg_test  = astgraph_to_pyg(test_graphs)

vocab_size = type_vocab.size()
print('PyG gráfok:', len(pyg_train), len(pyg_val), len(pyg_test), '| vocab_size =', vocab_size)

# Loader-ek
train_loader = DataLoader(pyg_train, batch_size=BATCH_TRAIN, shuffle=True)
val_loader   = DataLoader(pyg_val,   batch_size=BATCH_EVAL)
test_loader  = DataLoader(pyg_test,  batch_size=BATCH_EVAL)

# Osztálysúly BCE-hez: pos_weight = neg/pos
y_train = np.array([int(g.y.item()) for g in pyg_train])
pos = (y_train==1).sum(); neg = (y_train==0).sum()
pos_weight = torch.tensor([max(1.0, neg/max(1,pos))], dtype=torch.float, device=device)
print('pos_weight (BCE):', float(pos_weight.item()))

# =========================
# 8) GGNN MODELL (változatlan mélység: blocks=5, steps=10) + LeakyReLU + Attention Pooling
# =========================
import torch.nn.functional as F
from torch import nn


from torch_geometric.nn import (
    GatedGraphConv,
    global_mean_pool,
    GlobalAttention,
    GCNConv,
    GINConv,
)

class GGNNBlockFeats(nn.Module):
    def __init__(self, channels: int, steps: int, num_edge_types: int = 3):
        super().__init__()
        self.num_edge_types = max(1, num_edge_types)
        self.convs = nn.ModuleList([GatedGraphConv(channels, num_layers=steps) for _ in range(self.num_edge_types)])
        self.norm = nn.LayerNorm(channels)
    def forward(self, h, edge_index, edge_type=None):
        if (edge_type is None) or (self.num_edge_types==1):
            h_msg = self.convs[0](h, edge_index)
        else:
            # típusonként külön GatedGraphConv, majd összeadás
            parts = []
            for etype, conv in enumerate(self.convs):
                mask = (edge_type == etype)
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=False).view(-1)
                ei = edge_index[:, idx]
                parts.append(conv(h, ei))
            h_msg = torch.stack(parts, dim=0).sum(dim=0) if parts else torch.zeros_like(h)
        h_out = self.norm(h + h_msg)
        return h_out

class GGNNClassifierFeatsNoEmb(nn.Module):
    def __init__(self, num_types:int, tok_dim:int, small_dim:int=2,
                 steps:int=10, blocks:int=5, num_edge_types:int=3, dropout:float=0.3):
        super().__init__()
        self.dim_type=num_types
        self.dim_tok=tok_dim+1
        self.dim_small=small_dim
        self.channels = self.dim_type + self.dim_tok + self.dim_small
        self.blocks = nn.ModuleList([GGNNBlockFeats(self.channels, steps, num_edge_types) for _ in range(blocks)])
        self.drop = nn.Dropout(dropout)

        # Attention pooling kapu-hálóval
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.channels, self.channels // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(self.channels // 2, 1)
            )
        )

        # 1 logit a BCE-hez
        self.head = nn.Sequential(
            nn.Linear(self.channels,self.channels),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(self.channels,1)
        )
    def build_features(self, data):
        xt = getattr(data,'x_type', getattr(data,'x'))
        xt = xt.squeeze(-1) if xt.dim()==2 else xt
        h_type = F.one_hot(xt.long(), num_classes=self.dim_type).float()

        if hasattr(data,'x_tok'):
            xk = data.x_tok; xk = xk.squeeze(-1) if xk.dim()==2 else xk
            xk = xk.clamp(0, self.dim_tok-1).long()
            h_tok = F.one_hot(xk, num_classes=self.dim_tok).float()
        else:
            N = h_type.size(0)
            h_tok = torch.zeros((N,self.dim_tok), dtype=torch.float, device=h_type.device)
            h_tok[:, self.dim_tok-1] = 1.0

        h_small = getattr(
            data,
            "x_small",
            torch.zeros((h_type.size(0),  self.dim_small), dtype=torch.float, device=h_type.device),
        )
        return torch.cat([h_type, h_tok, h_small], dim=1)

    def forward(self, data):
        h = self.build_features(data)
        et = getattr(data,'edge_type', None)
        for blk in self.blocks:
            h = blk(h, data.edge_index, et)
        h = self.drop(h)
        hg = self.pool(h, data.batch)
        return self.head(hg).view(-1)  # [B]

from torch_geometric.nn import global_max_pool
class MLPBaseline(nn.Module):
    """Egyszerű baseline: nem használ gráfot.
    A node feature-ökből (AST csúcsok) globális aggregált statisztikákat számolunk
    (mean, max, std), és ez megy át egy MLP-n.
    """
    def __init__(self, num_types:int, tok_dim:int, small_dim:int=2,
                 hidden:int=256, dropout:float=0.3):
        super().__init__()
        self.dim_type = num_types
        self.dim_tok = tok_dim + 1
        self.dim_small = small_dim
        self.in_dim = self.dim_type + self.dim_tok + self.dim_small

        agg_dim = self.in_dim * 3  # mean + max + std

        self.mlp = nn.Sequential(
            nn.Linear(agg_dim, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def build_features(self, data):
        xt = getattr(data, "x_type", getattr(data, "x"))
        xt = xt.squeeze(-1) if xt.dim() == 2 else xt
        h_type = F.one_hot(xt.long(), num_classes=self.dim_type).float()

        if hasattr(data, "x_tok"):
            xk = data.x_tok
            xk = xk.squeeze(-1) if xk.dim() == 2 else xk
            xk = xk.clamp(0, self.dim_tok - 1).long()
            h_tok = F.one_hot(xk, num_classes=self.dim_tok).float()
        else:
            N = h_type.size(0)
            h_tok = torch.zeros((N, self.dim_tok), dtype=torch.float, device=h_type.device)
            h_tok[:, self.dim_tok - 1] = 1.0

        h_small = getattr(
            data,
            "x_small",
            torch.zeros((h_type.size(0), self.dim_small), dtype=torch.float, device=h_type.device),
        )

        return torch.cat([h_type, h_tok, h_small], dim=1)

    def forward(self, data):
        x = self.build_features(data)
        batch = data.batch

        # mean és max PyG-ből:
        mean = global_mean_pool(x, batch)      # [B, F]
        max_ = global_max_pool(x, batch)       # [B, F]

        # std-t kiszámoljuk: std = sqrt(E[x^2] - E[x]^2)
        mean_sq = global_mean_pool(x * x, batch)   # E[x^2]
        var = (mean_sq - mean * mean).clamp(min=0)
        std = torch.sqrt(var + 1e-8)

        h = torch.cat([mean, max_, std], dim=1)    # [B, 3F]
        return self.mlp(h).view(-1)



class GCNClassifierFeatsNoEmb(nn.Module):
    """GCN baseline: a GGNN build_features-ét használjuk, csak a message passing más."""
    def __init__(self, num_types:int, tok_dim:int, small_dim:int=2,
                 hidden:int=256, layers:int=3, dropout:float=0.3):
        super().__init__()
        self.dim_type = num_types
        self.dim_tok = tok_dim + 1
        self.dim_small = small_dim
        in_dim = self.dim_type + self.dim_tok + self.dim_small

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden // 2, 1),
            )
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def build_features(self, data):
        xt = getattr(data, "x_type", getattr(data, "x"))
        xt = xt.squeeze(-1) if xt.dim() == 2 else xt
        h_type = F.one_hot(xt.long(), num_classes=self.dim_type).float()

        if hasattr(data, "x_tok"):
            xk = data.x_tok
            xk = xk.squeeze(-1) if xk.dim() == 2 else xk
            xk = xk.clamp(0, self.dim_tok - 1).long()
            h_tok = F.one_hot(xk, num_classes=self.dim_tok).float()
        else:
            N = h_type.size(0)
            h_tok = torch.zeros((N, self.dim_tok), dtype=torch.float, device=h_type.device)
            h_tok[:, self.dim_tok - 1] = 1.0

        h_small = getattr(
            data,
            "x_small",
            torch.zeros((h_type.size(0), self.dim_small), dtype=torch.float, device=h_type.device),
        )

        return torch.cat([h_type, h_tok, h_small], dim=1)

    def forward(self, data):
        x = self.build_features(data)
        edge_index = data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, 0.01)

        x = self.drop(x)
        hg = self.pool(x, data.batch)
        return self.head(hg).view(-1)


class GINClassifierFeatsNoEmb(nn.Module):
    """'GINE' helyett egyszerű GINConv-alapú modell (edge_attr nélkül)."""
    def __init__(self, num_types:int, tok_dim:int, small_dim:int=2,
                 hidden:int=256, layers:int=3, dropout:float=0.3):
        super().__init__()
        self.dim_type = num_types
        self.dim_tok = tok_dim + 1
        self.dim_small = small_dim
        in_dim = self.dim_type + self.dim_tok + self.dim_small

        self.layers = layers
        self.convs = nn.ModuleList()

        nn_first = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.convs.append(GINConv(nn_first))

        for _ in range(layers - 1):
            nn_layer = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(nn_layer))

        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden // 2, 1),
            )
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def build_features(self, data):
        xt = getattr(data, "x_type", getattr(data, "x"))
        xt = xt.squeeze(-1) if xt.dim() == 2 else xt
        h_type = F.one_hot(xt.long(), num_classes=self.dim_type).float()

        if hasattr(data, "x_tok"):
            xk = data.x_tok
            xk = xk.squeeze(-1) if xk.dim() == 2 else xk
            xk = xk.clamp(0, self.dim_tok - 1).long()
            h_tok = F.one_hot(xk, num_classes=self.dim_tok).float()
        else:
            N = h_type.size(0)
            h_tok = torch.zeros((N, self.dim_tok), dtype=torch.float, device=h_type.device)
            h_tok[:, self.dim_tok - 1] = 1.0

        h_small = getattr(
            data,
            "x_small",
            torch.zeros((h_type.size(0), self.dim_small), dtype=torch.float, device=h_type.device),
        )

        return torch.cat([h_type, h_tok, h_small], dim=1)

    def forward(self, data):
        x = self.build_features(data)
        edge_index = data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.leaky_relu(x, 0.01)

        x = self.drop(x)
        hg = self.pool(x, data.batch)
        return self.head(hg).view(-1)


def build_model(model_name: str,
                num_types: int,
                tok_dim: int,
                small_dim: int = 2,
                num_edge_types: int = 3) -> nn.Module:
    """Egységes modell factory: GGNN / GCN / GIN / MLP."""
    name = model_name.lower()
    if name == "ggnn":
        return GGNNClassifierFeatsNoEmb(
            num_types=num_types,
            tok_dim=tok_dim,
            small_dim=small_dim,
            steps=10,
            blocks=5,
            num_edge_types=num_edge_types,
            dropout=0.3,
        )
    elif name == "gcn":
        return GCNClassifierFeatsNoEmb(
            num_types=num_types,
            tok_dim=tok_dim,
            small_dim=small_dim,
            hidden=256,
            layers=3,
            dropout=0.3,
        )
    elif name in ("gine", "gin"):
        return GINClassifierFeatsNoEmb(
            num_types=num_types,
            tok_dim=tok_dim,
            small_dim=small_dim,
            hidden=256,
            layers=3,
            dropout=0.3,
        )
    elif name == "mlp":
        return MLPBaseline(
            num_types=num_types,
            tok_dim=tok_dim,
            small_dim=small_dim,
            hidden=256,
            dropout=0.3,
        )
    else:
        raise ValueError(f"Ismeretlen MODEL: {model_name}")

# =========================
# 3.5) EDA – GYORSELEMZÉS ÉS VIZUALIZÁCIÓ (train előtt)
# =========================
import os, re, math
import matplotlib.pyplot as plt

EDA_DIR = "eda_report"
os.makedirs(EDA_DIR, exist_ok=True)

print("\n[EDA] Alap statisztikák a kiegyensúlyozott mintán:")
print("Mintaszám:", len(raw_df))
print(raw_df['label'].value_counts().rename('count'))
print(raw_df['label'].value_counts(normalize=True).rename('proportion').round(3))
"""
# --- Mérőszámok a kódokról (nyers + normalizált) ---
def _safe_len(s): 
    try: return len(s)
    except: return 0

def _num_lines(s):
    try: return s.count("\n") + 1
    except: return 0

def _ratio(pattern, s):
    try:
        m = re.findall(pattern, s)
        return len(m) / max(1, len(s))
    except Exception:
        return 0.0

def _count(pattern, s):
    try:
        m = re.findall(pattern, s)
        return len(m)
    except Exception:
        return 0

sample_codes = train_df["code"].astype(str).tolist()
norm_codes   = train_df["code_norm"].astype(str).tolist()

eda_df = pd.DataFrame({
    "label": raw_df["label"].astype(int).values,
    "len_chars": [ _safe_len(c) for c in sample_codes ],
    "num_lines": [ _num_lines(c) for c in sample_codes ],
    "len_chars_norm": [ _safe_len(cn) for cn in norm_codes ],
    # egyszerű tartalmi jellemzők (nyers kódból)
    "digit_ratio": [ _ratio(r"\d", c) for c in sample_codes ],
    "upper_ratio": [ _ratio(r"[A-Z]", c) for c in sample_codes ],
    "lower_ratio": [ _ratio(r"[a-z]", c) for c in sample_codes ],
    "sym_ratio":   [ _ratio(r"[{}()\\[\\];,]", c) for c in sample_codes ],
    "num_string_literals": [ _count(r"\"([^\"\\\\]|\\\\.)*\"", c) for c in sample_codes ],
    "num_char_literals":   [ _count(r"\'([^\'\\\\]|\\\\.)*\'", c) for c in sample_codes ],
})

eda_df.describe().to_csv(os.path.join(EDA_DIR, "eda_summary.csv"), index=True)
print("EDA summary mentve:", os.path.join(EDA_DIR, "eda_summary.csv"))

plt.figure()
eda_df.boxplot(column=["len_chars", "len_chars_norm"])
plt.title("Kódhossz eloszlás (nyers vs normalizált)")
plt.savefig(os.path.join(EDA_DIR, "length_boxplot.png"))
plt.close()
"""
# =========================
# 9) OPTIMALIZÁLÓ, LOSS, AMP
# =========================
# =========================
# MODELL VÁLASZTÁS
# =========================
MODEL = 'gcn'  # 'ggnn' | 'gcn' | 'gine' | 'mlp'
num_edge_types = len(EDGE_TYPES)

model = build_model(
    MODEL,
    num_types=vocab_size,
    tok_dim=TOK_DIM,
    small_dim=2,
    num_edge_types=num_edge_types,
).to(device)

# tanulási ráta és epochs a modell típusához igazítva
if MODEL == 'ggnn':
    epochs = EPOCHS_GGNN
elif MODEL in ('gine', 'gin'):
    epochs = EPOCHS_GINE
else:
    epochs = 20  # baseline-oknak elég kevesebb is

lr = 3e-4
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# BCEWithLogitsLoss: pos_weight a pozitív osztály súlyozására
crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AMP bekapcsolása CUDA-n
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def run(loader, train=False):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        if train:
            opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch)                   # [B]
            target = batch.y.float().view(-1)       # [B]
            loss = crit(logits, target)

        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).long()
            correct += int((pred == batch.y).sum().item())
            total += batch.y.size(0)
            loss_sum += float(loss.item()) * batch.y.size(0)

    return loss_sum / max(1,total), correct / max(1,total)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def find_best_threshold(loader):
    model.eval(); y_true, probs = [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                probs += torch.sigmoid(logits).cpu().tolist()
                y_true += batch.y.cpu().tolist()
    import numpy as np
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = [1 if p >= thr else 0 for p in probs]
        f1 = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)[2]
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1

# ---- Értékelés megadott küszöbbel ----
def evaluate(loader, thr=0.5):
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)               # [B]
                prob = torch.sigmoid(logits)
                pred = (prob >= thr).long()
                y_true += batch.y.cpu().tolist()
                y_pred += pred.cpu().tolist()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

# =========================
# 10) TANÍTÁS + VALIDÁCIÓ + MENTÉS (BEST F1 ALAPJÁN)
# =========================
best_val, best_state = 0.0, None
for epoch in range(1, epochs+1):
    tr_loss, tr_acc = run(train_loader, train=True)
    va_acc, va_prec, va_rec, va_f1, _ = evaluate(val_loader, thr=0.5)
    if va_f1 > best_val:
        best_val, best_state = va_f1, model.state_dict()
    print(f"epoch {epoch:02d} | train loss {tr_loss:.4f} | train acc {tr_acc:.3f} | val acc {va_acc:.3f} | val F1 {va_f1:.3f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

if best_state is not None:
    model.load_state_dict(best_state)

# --- küszöb hangolás a validation F1 maximalizálására ---
best_thr, best_val_f1 = find_best_threshold(val_loader)
print(f"Best VAL F1 @ thr={best_thr:.3f}: {best_val_f1:.3f}")

# --- végső TEST értékelés a megtalált küszöbbel ---
te_acc, te_prec, te_rec, te_f1, te_cm = evaluate(test_loader, thr=best_thr)
print("TEST | acc:", te_acc, "| prec:", te_prec, "| rec:", te_rec, "| f1:", te_f1)
print("Confusion matrix:\n", te_cm)

# --- mentés: modell súlyai + meta-információ ---
num_params = sum(p.numel() for p in model.parameters())
out_name = f"cpp_augast_{MODEL}_bce_best.pt"
torch.save(model.state_dict(), out_name)
print("Mentve:", out_name)

meta = {
    "model": MODEL,
    "vocab_map": type_vocab.map,  # dict: node_type -> int
    "tok_dim": TOK_DIM,
    "num_edge_types": len(EDGE_TYPES),
    "epochs": int(epochs),
    "best_thr": float(best_thr),
    "val_f1": float(best_val_f1),
    "test_acc": float(te_acc),
    "test_prec": float(te_prec),
    "test_rec": float(te_rec),
    "test_f1": float(te_f1),
    "num_params": int(num_params),
}
meta_name = f"cpp_augast_{MODEL}_meta.json"
import json
with open(meta_name, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("Mentve:", meta_name)

# --- egységes összefoglaló táblázat a dolgozatodhoz ---
summary_df = pd.DataFrame([{
    "model": MODEL,
    "epochs": int(epochs),
    "best_thr": float(best_thr),
    "val_f1": float(best_val_f1),
    "test_acc": float(te_acc),
    "test_prec": float(te_prec),
    "test_rec": float(te_rec),
    "test_f1": float(te_f1),
    "num_params": int(num_params),
}])

print("\nÖsszefoglaló eredmények (egységes táblázat):")
print(summary_df.to_string(index=False))
