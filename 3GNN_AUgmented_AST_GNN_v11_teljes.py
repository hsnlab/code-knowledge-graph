# =========================
# 0) IMPORTOK + ALAP BEÁLLÍTÁS
# =========================
import os, json, random, hashlib, gc
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path

# <<< CUDA allocator: ezt a torch import ELŐTT kell beállítani!
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# =========================
# 1) ESZKÖZ VÁLASZTÁS
# =========================

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


# =========================
# 2) PARAMÉTEREK
# =========================
SELECT_DATASET = 'draper_hf'     # 'code_x_glue' | 'draper_hf'
LANG = 'cpp'                     # 'c' | 'cpp'
MAX_SAMPLES = 0                  # 0 → mind (óvatosan RAM/VRAM miatt)
BATCH_TRAIN, BATCH_EVAL = 64, 128
EPOCHS_GGNN, EPOCHS_GINE = 30, 20

# Balanszolás kapcsoló (None → nincs downsample; pl. 3 → ~25% pos, 9 → ~10% pos)
BALANCE_NEG_PER_POS = None

# Shardolás
CHUNK_SIZE = 20_000
SHARD_DIR = "shards"
os.makedirs(SHARD_DIR, exist_ok=True)

# one-hot hash bucket a levelek szövegéhez (normalizálás miatt lehet kicsi)
TOK_DIM = 128                    # korábban 1024; normalizálással bőven elég 64–256
TOK_SENTINEL = TOK_DIM           # üres/nem-levél → sentinel id

EDGE_TYPES = {'parent':0, 'next_sibling':1, 'next_token':2}

# =========================
# 3) TREE-SITTER INIT (C/C++)
# =========================
from tree_sitter import Language, Parser
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

TS_LANG = Language(tsc.language()) if LANG.lower()=='c' else Language(tscpp.language())
parser = Parser(TS_LANG)
print('Tree-Sitter OK, LANG =', LANG)

# =========================
# 4) ADATBETÖLTÉS + CÉL OSZLOPOK
# =========================

def _normalize_label(x):
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in {'1','vul','vulnerable','pos','positive','true','bug'}: return 1
        if xs in {'0','non-vul','nonvul','benign','safe','neg','negative','false','clean'}: return 0
    try:
        return 1 if int(x)==1 else 0
    except Exception:
        return None

def _auto_pick_columns(df: pd.DataFrame):
    code_col = None; label_col = None
    for c in df.columns:
        if c.lower() in {'code','func','function','source','source_code','program'}:
            code_col = c; break
    for c in df.columns:
        if c.lower() in {'label','target','y','vul','vulnerable','bug'}:
            label_col = c; break
    if code_col is None:
        for c in df.columns:
            if df[c].dtype==object:
                code_col = c; break
    if label_col is None:
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
                label_col = c; break
    return code_col, label_col

def load_any_dataset(select: str) -> pd.DataFrame:
    s = select.lower()
    if s == 'code_x_glue':
        ds = load_dataset('google/code_x_glue_cc_defect_detection')
        df = pd.DataFrame({'code': ds['train']['func'], 'label': ds['train']['target']})
        df['label'] = df['label'].apply(_normalize_label)
        return df.dropna(subset=['code','label']).reset_index(drop=True)
    elif s == 'draper_hf':
        ds = load_dataset('claudios/Draper')
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        df = pd.DataFrame({c: ds[split][c] for c in ds[split].column_names})
        ccol, lcol = _auto_pick_columns(df)
        df = df[[ccol, lcol]].rename(columns={ccol:'code', lcol:'label'})
        df['label'] = df['label'].apply(_normalize_label)
        return df.dropna(subset=['code','label']).reset_index(drop=True)
    else:
        raise ValueError(select)

raw_df = load_any_dataset(SELECT_DATASET)

# --- Kiegyensúlyozás: ~10% pozitív (összes pozitív megtartása, negatívakból mintavétel) ---

def make_about_10pct_pos(df: pd.DataFrame, seed: int = SEED, neg_per_pos: int = 9) -> pd.DataFrame:
    df = df.copy()
    df['label'] = df['label'].astype(int)
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    if len(df_pos) == 0:
        raise ValueError("Nincs pozitív minta a datasetben, nem lehet kiegyensúlyozni.")

    target_neg = min(len(df_neg), neg_per_pos * len(df_pos))   # ~10% pozitív → 1 : 9 arány
    df_neg_sampled = df_neg.sample(target_neg, random_state=seed)

    df_bal = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_bal

if BALANCE_NEG_PER_POS is not None:
    raw_df = make_about_10pct_pos(raw_df, seed=SEED, neg_per_pos=BALANCE_NEG_PER_POS)

# MAX_SAMPLES kihagyva: teljes készlet
# if MAX_SAMPLES and len(raw_df) > MAX_SAMPLES:
#     raw_df, _ = train_test_split(
#         raw_df, train_size=MAX_SAMPLES, stratify=raw_df['label'], random_state=SEED
#     )

raw_df['label'] = raw_df['label'].astype(int)
print('Kiegyensúlyozott minták száma:', len(raw_df))
print('Arányok:\n', raw_df['label'].value_counts(normalize=True).rename('proportion').round(3))
print("Oszlopok:", raw_df.columns.tolist())

# =========================
# 5) KÓDNORMALIZÁLÁS (alpha-renaming, literál-helyettesítés)
# =========================

def normalize_code(code: str) -> str:
    text = code.encode("utf8")
    try:
        tree = parser.parse(text)
        root = tree.root_node
    except Exception:
        # parser hiba esetén fallback: nagyon egyszerű normalizálás
        import re
        code = re.sub(r'\"([^"\\]|\\.)*\"', '<STR>', code)
        code = re.sub(r"\'([^'\\]|\\.)*\'", '<CHAR>', code)
        code = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', code)
        return code

    scope_stack = []  # list of dicts: {"func":str, "params":{}, "vars":{}}
    counters = {"func":0}

    def push_scope():
        scope_stack.append({"func":None, "params":{}, "vars":{}})
    def pop_scope():
        scope_stack.pop()
    def get_id(node):
        return text[node.start_byte:node.end_byte].decode("utf8","ignore")
    def assign_param(name, scope):
        if name in scope["params"]: return scope["params"][name]
        idx = len(scope["params"]) + 1
        scope["params"][name] = f"PARAM_{idx}"
        return scope["params"][name]
    def assign_var(name, scope):
        if name in scope["vars"]: return scope["vars"][name]
        idx = len(scope["vars"]) + 1
        scope["vars"][name] = f"VAR_{idx}"
        return scope["vars"][name]
    LITS = {"number_literal":"<NUM>", "string_literal":"<STR>", "char_literal":"<CHAR>"}

    # Első passz: scope-ok/deklarációk
    def first_pass(node):
        t = node.type
        if t == "function_definition":
            push_scope()
            # function_declarator: azonosító + param lista
            func_decl = None
            for ch in node.children:
                if ch.type == "function_declarator":
                    func_decl = ch; break
            if func_decl is not None:
                # függvénynév FUNC_i
                for ch in func_decl.children:
                    if ch.type == "identifier":
                        counters["func"] += 1
                        scope_stack[-1]["func"] = f"FUNC_{counters['func']}"
                        break
                # paraméter nevek
                for ch in func_decl.children:
                    if ch.type == "parameter_list":
                        for p in ch.children:
                            if p.type == "parameter_declaration":
                                for gch in p.children:
                                    if gch.type == "identifier":
                                        assign_param(get_id(gch), scope_stack[-1])
            # bejárt gyerekek
            for ch in node.children:
                first_pass(ch)
            pop_scope()
            return

        # lokális deklarációk (egyszerűsítve)
        if scope_stack:
            if t in {"init_declarator", "declarator"}:
                for ch in node.children:
                    if ch.type == "identifier":
                        assign_var(get_id(ch), scope_stack[-1])

        for ch in node.children:
            first_pass(ch)

    # Második passz: kibocsátás
    def lookup_identifier(name):
        for sc in reversed(scope_stack):
            if name in sc["params"]: return sc["params"][name]
            if name in sc["vars"]: return sc["vars"][name]
        return name

    def second_pass(node):
        t = node.type
        if t == "function_definition":
            push_scope()
            out = []
            for ch in node.children:
                out.append(second_pass(ch))
            pop_scope()
            return "".join(out)

        if t in LITS:
            return LITS[t]

        if t == "identifier":
            # ha function_declarator gyereke és ez a név: FUNC_i
            if scope_stack and scope_stack[-1]["func"] is not None:
                parent = node.parent
                if parent and parent.type == "function_declarator":
                    return scope_stack[-1]["func"]
            return lookup_identifier(get_id(node))

        # Levél: visszaadjuk az eredeti lexémát
        if len(node.children) == 0:
            return text[node.start_byte:node.end_byte].decode("utf8","ignore")
        # Összefűzzük a gyerekek kibocsátását
        parts = []
        for ch in node.children:
            parts.append(second_pass(ch))
        return "".join(parts)

    # futtatás
    try:
        first_pass(root)
        return second_pass(root)
    except Exception:
        # Ha bármi gond, biztonságos fallback literálokra
        import re
        code = text.decode("utf8","ignore")
        code = re.sub(r'\"([^"\\]|\\.)*\"', '<STR>', code)
        code = re.sub(r"\'([^'\\]|\\.)*\'", '<CHAR>', code)
        code = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', code)
        return code

# =========================
# 6) STRATIFIKÁLT SPLIT
# =========================

df_train, df_tmp = train_test_split(raw_df, test_size=0.2, stratify=raw_df['label'], random_state=SEED)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, stratify=df_tmp['label'], random_state=SEED)
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"{name}: {len(df)} | arány:\n", df['label'].value_counts(normalize=True).round(3))

# =========================
# 7) AUGMENTED AST ÉPÍTÉS (normalizált kódból!)
# =========================

@dataclass
class ASTGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, str]]
    label: int
    raw: str

def build_augmented_ast(code: str):
    tree = parser.parse(code.encode('utf8'))
    nodes, edges = [], []
    nid = 0
    def walk(node, parent_id=None, last_sib=None, depth=0):
        nonlocal nid
        my = nid; nid += 1
        snippet = code.encode('utf8')[node.start_byte:node.end_byte]
        children = node.children
        nodes.append({
            'id': my,
            'type': node.type,
            'is_leaf': int(len(children)==0),
            'depth': depth,
            'text': snippet.decode('utf8','ignore')
        })
        if parent_id is not None:
            edges.append((parent_id, my, 'parent'))
        if last_sib is not None:
            edges.append((last_sib, my, 'next_sibling'))
        prev = None
        for ch in children:
            ch_id = walk(ch, my, prev, depth+1)
            if prev is not None:
                edges.append((prev, ch_id, 'next_token'))
            prev = ch_id
        return my
    walk(tree.root_node)
    return nodes, edges

# =========================
# 8) PyG KONVERZIÓ + VOCAB
# =========================

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


def df_to_graphs(df: pd.DataFrame):
    out = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_code = str(row['code'])
        code = normalize_code(raw_code)           # <<< NORMALIZÁLT!
        y = int(row['label'])
        n, e = build_augmented_ast(code)
        out.append(ASTGraph(n,e,y,code))
    return out


def to_pyg(gs):
    pyg = []
    for g in gs:
        # type id
        type_ids = [[type_vocab.id(n['type'])] for n in g.nodes]
        x_type = torch.tensor(np.array(type_ids), dtype=torch.long)

        # token bucket csak levelek text-jéből
        tok_ids = [[_hash_bucket(n.get('text','') if n.get('is_leaf',0) else '', TOK_DIM)] for n in g.nodes]
        x_tok = torch.tensor(np.array(tok_ids), dtype=torch.long)

        # small numerikus: [is_leaf, depth_norm]
        max_depth = max([n.get('depth',0) for n in g.nodes] + [1])
        small = [[float(n.get('is_leaf',0)), float(n.get('depth',0))/float(max_depth)] for n in g.nodes]
        x_small = torch.tensor(np.array(small), dtype=torch.float)

        # élek
        if len(g.edges)==0:
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

# =========================
# 9) SHARDOLÁS: df -> (AST -> PyG) -> .pt fájlok
# =========================

def iter_chunks(df: pd.DataFrame, chunk_size: int):
    n = len(df)
    for start in range(0, n, chunk_size):
        yield df.iloc[start:start+chunk_size].reset_index(drop=True), start//chunk_size

def df_to_pyg_chunk(df: pd.DataFrame):
    gs = df_to_graphs(df)          # normalize_code + build_augmented_ast
    return to_pyg(gs)

def materialize_split_to_shards(df_split: pd.DataFrame, split_name: str):
    paths = []
    for chunk_df, idx in iter_chunks(df_split, CHUNK_SIZE):
        pyg_list = df_to_pyg_chunk(chunk_df)
        p = Path(SHARD_DIR)/f"{split_name}_shard_{idx:04d}.pt"
        torch.save(pyg_list, p)
        paths.append(str(p))
        # memória takarítás
        del pyg_list, chunk_df
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return paths

print("Shardolás indul...")
train_shards = materialize_split_to_shards(df_train, "train")
val_shards   = materialize_split_to_shards(df_val,   "val")
test_shards  = materialize_split_to_shards(df_test,  "test")
print(f"Shards | train={len(train_shards)} val={len(val_shards)} test={len(test_shards)}")

# =========================
# 10) BIZTONSÁGOS TORCH.LOAD (PyTorch 2.6+)
# =========================

def safe_torch_load(path, map_location="cpu"):
    """PyTorch 2.6-tól a torch.load default security policy szigorúbb.
    A PyG DataEdgeAttr-t explicit whitelistre tesszük, és weights_only=False-t állítunk.
    """
    from torch.serialization import safe_globals
    try:
        from torch_geometric.data.data import DataEdgeAttr
        allow = [DataEdgeAttr]
    except Exception:
        allow = []
    with safe_globals(allow):
        return torch.load(path, map_location=map_location, weights_only=False)

# =========================
# 11) MODELL DEFINÍCIÓK
# =========================

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool, GlobalAttention

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
            parts=[]
            for t, conv in enumerate(self.convs):
                mask = (edge_type==t)
                if mask.numel()>0 and int(mask.sum())>0:
                    ei = edge_index[:, mask]
                    parts.append(conv(h, ei))
            h_msg = torch.stack(parts, dim=0).sum(dim=0) if parts else torch.zeros_like(h)
        h = self.norm(h + h_msg)
        return F.leaky_relu(h, negative_slope=0.01)

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

        h_small = getattr(data,'x_small', torch.zeros((h_type.size(0),self.dim_small), dtype=torch.float, device=h_type.device))
        return torch.cat([h_type, h_tok, h_small], dim=1)

    def forward(self, data):
        h = self.build_features(data)
        et = getattr(data,'edge_type', None)
        for blk in self.blocks:
            h = blk(h, data.edge_index, et)
        h = self.drop(h)
        hg = self.pool(h, data.batch)
        return self.head(hg).view(-1)  # [B]

# =========================
# 11) POS_WEIGHT, VOCAB, MODELL INICIALIZÁLÁS
# =========================

# pos_weight a train split arányából
train_labels = df_train['label'].astype(int).to_numpy()
pos = (train_labels==1).sum(); neg = (train_labels==0).sum()
pos_weight = torch.tensor([max(1.0, neg/max(1,pos))], dtype=torch.float, device=device)
print('pos_weight (BCE):', float(pos_weight.item()))

# Vocab a shardolás közben töltődött, most fixáljuk
vocab_size = type_vocab.size()
print('vocab_size =', vocab_size)

MODEL = 'ggnn'
num_edge_types = len(EDGE_TYPES)

model = GGNNClassifierFeatsNoEmb(
    num_types=vocab_size, tok_dim=TOK_DIM, small_dim=2,
    steps=10, blocks=5, num_edge_types=num_edge_types, dropout=0.3
).to(device)

lr, epochs = 3e-4, EPOCHS_GGNN
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# BCEWithLogitsLoss: pos_weight a pozitív osztály súlyozására
crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# AMP bekapcsolása CUDA-n
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# =========================
# 12) SHARDOS TRAIN/EVAL FUNKCIÓK
# =========================

def run_loader(pyg_list, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    loader = DataLoader(pyg_list, batch_size=BATCH_TRAIN if train else BATCH_EVAL, shuffle=train)
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        if train:
            opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch)
            target = batch.y.float().view(-1)
            loss = crit(logits, target)

        if train:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(opt)
            scaler.update()

        loss_sum += loss.item() * batch.num_graphs
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        correct += int((pred == batch.y).sum())
        total += batch.num_graphs
    return (loss_sum/total if total else 0.0), (correct/total if total else 0.0)

@torch.no_grad()
def collect_probs_and_labels_from_shards(shard_paths):
    model.eval()
    y_true, probs = [], []
    for p in shard_paths:
        pyg_list = safe_torch_load(p, map_location="cpu")
        loader = DataLoader(pyg_list, batch_size=BATCH_EVAL, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch)
                probs += torch.sigmoid(logits).cpu().tolist()
                y_true += batch.y.cpu().tolist()
        del pyg_list, loader, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return y_true, probs

def f1_at_thr(y_true, probs, thr):
    y_pred = [1 if p >= thr else 0 for p in probs]
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

def find_best_thr_on_val():
    y_true, probs = collect_probs_and_labels_from_shards(val_shards)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        _, _, _, f1, _ = f1_at_thr(y_true, probs, thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1

# =========================
# 13) TANÍTÁS + VALIDÁCIÓ SHARDONKÉNT
# =========================

best_val_f1 = 0.0
best_state  = None

for epoch in range(1, EPOCHS_GGNN+1):
    # --- TRAIN: végigmegyünk a train shardeokon ---
    tr_loss_sum, tr_acc_sum, tr_n = 0.0, 0.0, 0
    for p in train_shards:
        pyg_list = safe_torch_load(p, map_location="cpu")
        loss, acc = run_loader(pyg_list, train=True)
        tr_loss_sum += loss * len(pyg_list)
        tr_acc_sum  += acc  * len(pyg_list)
        tr_n        += len(pyg_list)
        del pyg_list
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tr_loss = tr_loss_sum / max(1, tr_n)
    tr_acc  = tr_acc_sum  / max(1, tr_n)

    # --- VALIDATION: teljes val shardeon metrika @ 0.5 ---
    y_true, probs = collect_probs_and_labels_from_shards(val_shards)
    acc, prec, rec, f1, _ = f1_at_thr(y_true, probs, 0.5)

    if f1 > best_val_f1:
        best_val_f1 = f1
        best_state  = model.state_dict()

    print(f"epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {acc:.3f} | val F1 {f1:.3f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

# --- Best modell visszatöltése ---
if best_state is not None:
    model.load_state_dict(best_state)

# --- Küszöb hangolás VAL-on ---
best_thr, best_val_f12 = find_best_thr_on_val()
print(f"Best VAL F1 @ thr={best_thr:.3f}: {best_val_f12:.3f}")

# --- TEST a legjobb küszöbbel ---
y_true_te, probs_te = collect_probs_and_labels_from_shards(test_shards)
te_acc, te_prec, te_rec, te_f1, te_cm = f1_at_thr(y_true_te, probs_te, best_thr)
print("TEST | acc:", te_acc, "| prec:", te_prec, "| rec:", te_rec, "| f1:", te_f1)
print("Confusion matrix:\n", te_cm)

# =========================
# 14) MENTÉS
# =========================

out_name = f'cpp_augast_GGNN_bce_best.pt'
torch.save(model.state_dict(), out_name)
print('Mentve:', out_name)

meta = {
    "vocab_map": type_vocab.map, # dict: node_type -> int
    "tok_dim": TOK_DIM, # e.g., 128
    "num_edge_types": len(EDGE_TYPES),
    "steps": 10,
    "blocks": 5,
    "dropout": 0.3
}
with open("cpp_augast_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f)
print("Mentve: cpp_augast_meta.json")

with open("cpp_augast_meta.json", "r", encoding="utf-8") as f:
    m = json.load(f)
print(len(m["vocab_map"]), m["tok_dim"], m["num_edge_types"])
