from pathlib import Path
import re, pandas as pd

# ---- Állítsd be: ----
DATASET_DIR = Path(r"C:\Szakdolgozat\SARD\JAVA")
LANG = 'java'
SELECT_CWES = None  # pl. ['CWE-079','CWE-089'] vagy None = mind

LANG_CFG = {
    'java': {'subdir': 'JAVA', 'exts': ['.java', '.java.txt', '.txt']},
}

def _read_text_file(p: Path) -> str:
    for enc in ('utf8','latin-1','cp1252'):
        try:
            return p.read_text(encoding=enc, errors='ignore')
        except Exception:
            pass
    return ''

_CWE_RE = re.compile(r'(?:^|[\\/])(CWE[-_]?(\d{1,3}))(?:$|[\\/])', re.I)
def _canonical_cwe(path_str: str) -> str | None:
    m = _CWE_RE.search(path_str)
    return f"CWE-{m.group(2).zfill(3)}" if m else None

def _label_from_path_or_name(p: Path) -> int | None:
    s = p.as_posix().lower()
    if '/bad/' in s or '\\bad\\' in s:   return 1
    if '/good/' in s or '\\good\\' in s: return 0
    n = p.name.lower()
    if 'bad' in n:  return 1
    if 'good' in n: return 0
    return None

def _looks_like_code_java(txt: str) -> bool:
    t = txt.strip()
    if len(t) < 20: return False
    keys = ('class ', 'interface ', 'enum ', 'package ', 'import ',
            'public ', 'private ', 'protected ', 'void ', 'static ')
    return any(k in t for k in keys) and ('{' in t or ';' in t)

def load_sard_java(root: Path, select_cwes=None) -> pd.DataFrame:
    cfg = LANG_CFG['java']
    base = root if root.name.upper()==cfg['subdir'] else ((root/cfg['subdir']) if (root/cfg['subdir']).exists() else root)
    sel  = {c.upper() for c in select_cwes} if select_cwes else None

    rows = []
    for cwe_dir in base.iterdir():
        if not cwe_dir.is_dir(): 
            continue
        cwe = _canonical_cwe(cwe_dir.as_posix())
        if not cwe or (sel and cwe.upper() not in sel):
            continue
        for ext in cfg['exts']:
            for p in cwe_dir.rglob(f"*{ext}"):
                s = p.as_posix().lower()
                if 'testcasesupport' in s:
                    continue
                lbl = _label_from_path_or_name(p)
                if lbl is None:
                    continue
                code = _read_text_file(p)
                if not code.strip():
                    continue
                if p.suffix.lower()=='.txt' and not p.name.lower().endswith('.java.txt'):
                    # sima .txt → szűrés heurisztikával
                    if not _looks_like_code_java(code):
                        continue
                rows.append({'code': code, 'label': int(lbl), 'cwe': cwe, 'path': p.as_posix()})
    if not rows:
        raise RuntimeError(f"Nem találtam mintát itt: {base}")
    df = pd.DataFrame(rows, columns=['code','label','cwe','path']).dropna().reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    print("Minták:", len(df), "| arány:", df['label'].value_counts(normalize=True).round(3).to_dict())
    print("Top CWE-k:\n", df['cwe'].value_counts().head())
    return df

raw_df = load_sard_java(DATASET_DIR, SELECT_CWES)


from sklearn.model_selection import train_test_split

df_train, df_tmp = train_test_split(raw_df, test_size=0.2, stratify=raw_df['label'], random_state=42)
df_val, df_test  = train_test_split(df_tmp,  test_size=0.5, stratify=df_tmp['label'], random_state=42)
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"{name}: {len(df)} | osztályarány:", df['label'].value_counts(normalize=True).round(3).to_dict())


from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

TS_LANG = Language(tsjava.language())
parser = Parser(TS_LANG)
print("Tree-Sitter OK (Java)")

EDGE_TYPES = {'parent': 0, 'next_sibling': 1, 'next_token': 2}

@dataclass
class ASTGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, str]]
    label: int
    raw: str

def build_augmented_ast(code: str) -> ASTGraph:
    tree = parser.parse(code.encode('utf8'))
    nodes, edges = [], []
    nid, max_depth = 0, 0

    def walk(node, depth=0, parent_id=None, last_sib_id=None):
        nonlocal nid, max_depth
        my = nid; nid += 1
        max_depth = max(max_depth, depth)
        text = code.encode('utf8')[node.start_byte:node.end_byte].decode('utf8', 'ignore')
        children = node.children
        nodes.append({
            'id': my,
            'type': node.type,
            'text': text,
            'is_leaf': int(len(children)==0),
            'depth': depth
        })
        if parent_id is not None:
            edges.append((parent_id, my, 'parent'))
        if last_sib_id is not None:
            edges.append((last_sib_id, my, 'next_sibling'))
        prev = None
        for ch in children:
            ch_id = walk(ch, depth+1, my, prev)
            if prev is not None:
                edges.append((prev, ch_id, 'next_token'))
            prev = ch_id
        return my

    walk(tree.root_node)
    md = max(1, max_depth)
    for n in nodes:
        n['depth_n'] = n['depth'] / md
    return ASTGraph(nodes=nodes, edges=edges, label=-1, raw=code)

def df_to_graphs(df):
    out = []
    for _, row in df.iterrows():
        g = build_augmented_ast(str(row['code']))
        g.label = int(row['label'])
        out.append(g)
    return out

graphs_train = df_to_graphs(df_train)
graphs_val   = df_to_graphs(df_val)
graphs_test  = df_to_graphs(df_test)
len(graphs_train), len(graphs_val), len(graphs_test)


import hashlib, numpy as np, torch
from torch_geometric.data import Data

TOK_DIM = 1024
TOK_SENTINEL = TOK_DIM  # +1 dim a sentinelnek

def _hash_bucket(s: str, D: int = TOK_DIM) -> int:
    if not s or not s.strip():
        return TOK_SENTINEL
    h = hashlib.md5(s.strip().encode('utf8')).hexdigest()
    return int(h, 16) % D

class Vocab:
    def __init__(self): self.map = {}
    def id(self, k):
        if k not in self.map: self.map[k] = len(self.map)
        return self.map[k]
    def size(self): return len(self.map)

type_vocab = Vocab()
for g in graphs_train:
    for n in g.nodes:
        type_vocab.id(n['type'])
vocab_size = type_vocab.size(); print("vocab_size =", vocab_size)

def to_pyg(g: ASTGraph) -> Data:
    type_ids, tok_ids, small = [], [], []
    max_depth = max(1, max(n['depth'] for n in g.nodes) if g.nodes else 1)
    for n in g.nodes:
        type_ids.append([type_vocab.id(n['type'])])
        tok_ids.append([_hash_bucket(n['text']) if n['is_leaf'] else TOK_SENTINEL])
        small.append([float(n['is_leaf']), float(n['depth'])/max_depth])

    x_type  = torch.tensor(np.array(type_ids), dtype=torch.long)
    x_tok   = torch.tensor(np.array(tok_ids),  dtype=torch.long)
    x_small = torch.tensor(np.array(small),    dtype=torch.float)

    if not g.edges:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_type  = torch.empty((0,),  dtype=torch.long)
    else:
        src = [s for s,_,_ in g.edges]
        dst = [d for _,d,_ in g.edges]
        et  = [EDGE_TYPES[t] for *_,t in g.edges]
        edge_index = torch.tensor([src,dst], dtype=torch.long)
        edge_type  = torch.tensor(et, dtype=torch.long)

    data = Data(
        x_type=x_type, x_tok=x_tok, x_small=x_small,
        edge_index=edge_index, y=torch.tensor([g.label], dtype=torch.long)
    )
    data.edge_type = edge_type
    data.x = x_type.clone()  # kompat, ha valahol .x-re hivatkozol
    return data

pyg_train = [to_pyg(g) for g in graphs_train]
pyg_val   = [to_pyg(g) for g in graphs_val]
pyg_test  = [to_pyg(g) for g in graphs_test]
print("PyG gráfok:", len(pyg_train), len(pyg_val), len(pyg_test))


from torch_geometric.loader import DataLoader
import numpy as np

train_loader = DataLoader(pyg_train, batch_size=64, shuffle=True)
val_loader   = DataLoader(pyg_val,   batch_size=128)
test_loader  = DataLoader(pyg_test,  batch_size=128)

y_train = np.array([int(g.y.item()) for g in pyg_train])
pos = (y_train==1).sum(); neg = (y_train==0).sum()
class_weight = torch.tensor([1.0, max(1.0, neg/max(1,pos))], dtype=torch.float)
num_edge_types = 3  # parent / next_sibling / next_token
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool

class GGNNBlockFeats(nn.Module):
    def __init__(self, channels: int, steps: int, num_edge_types: int = 3):
        super().__init__()
        self.num_edge_types = max(1, num_edge_types)
        self.convs = nn.ModuleList([GatedGraphConv(channels, num_layers=steps)
                                    for _ in range(self.num_edge_types)])
        self.norm = nn.LayerNorm(channels)
    def forward(self, h, edge_index, edge_type=None):
        if (edge_type is None) or (self.num_edge_types == 1):
            h_msg = self.convs[0](h, edge_index)
        else:
            parts = []
            for t, conv in enumerate(self.convs):
                mask = (edge_type == t)
                if mask.numel() > 0 and int(mask.sum()) > 0:
                    ei = edge_index[:, mask]
                    parts.append(conv(h, ei))
            h_msg = torch.stack(parts, dim=0).sum(dim=0) if parts else torch.zeros_like(h)
        h = self.norm(h + h_msg)
        return torch.relu(h)

class GGNNClassifierFeatsNoEmb(nn.Module):
    def __init__(self, num_types: int, tok_dim: int, small_dim: int = 2,
                 steps: int = 10, blocks: int = 5, num_edge_types: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dim_type  = num_types
        self.dim_tok   = tok_dim + 1
        self.dim_small = small_dim
        self.channels  = self.dim_type + self.dim_tok + self.dim_small
        self.blocks = nn.ModuleList([GGNNBlockFeats(self.channels, steps, num_edge_types)
                                     for _ in range(blocks)])
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.channels, 2)
        )
    def build_features(self, data):
        xt = data.x_type.squeeze(-1)
        h_type = F.one_hot(xt.long(), num_classes=self.dim_type).float()
        xk = data.x_tok.squeeze(-1).clamp(0, self.dim_tok-1).long()
        h_tok = F.one_hot(xk, num_classes=self.dim_tok).float()
        h_small = data.x_small
        return torch.cat([h_type, h_tok, h_small], dim=1)
    def forward(self, data):
        h = self.build_features(data)
        et = getattr(data, "edge_type", None)
        for blk in self.blocks:
            h = blk(h, data.edge_index, et)
            h = self.drop(h)
        hg = global_mean_pool(h, data.batch)
        return self.head(h)

model_ggnn = GGNNClassifierFeatsNoEmb(num_types=vocab_size, tok_dim=TOK_DIM, small_dim=2,
                                      steps=10, blocks=5, num_edge_types=num_edge_types, dropout=0.3).to(device)
opt = torch.optim.AdamW(model_ggnn.parameters(), lr=3e-4, weight_decay=1e-4)
crit = nn.CrossEntropyLoss(weight=class_weight.to(device))

def run_epoch(model, loader, train=False):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            if train: opt.zero_grad()
            logits = model(batch)
            loss = crit(logits, batch.y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                opt.step()
            loss_sum += loss.item() * batch.num_graphs
            pred = logits.argmax(1)
            correct += int((pred == batch.y).sum())
            total   += batch.num_graphs
    return loss_sum/max(1,total), correct/max(1,total)

best_val = 0.0; best_state = None
for ep in range(1, 31):
    tr_loss, tr_acc = run_epoch(model_ggnn, train_loader, train=True)
    va_loss, va_acc = run_epoch(model_ggnn, val_loader,   train=False)
    if va_acc > best_val: best_val, best_state = va_acc, model_ggnn.state_dict()
    print(f"[GGNN] epoch {ep:02d} | train {tr_acc:.3f} | val {va_acc:.3f}")
if best_state: model_ggnn.load_state_dict(best_state)
_, te_acc = run_epoch(model_ggnn, test_loader, train=False)
print("[GGNN] TEST acc:", te_acc)


from torch_geometric.nn import GINEConv, global_mean_pool

class GINENoEmb(nn.Module):
    def __init__(self, num_types:int, tok_dim:int, small_dim:int=2, layers:int=4, dropout:float=0.3, num_edge_types:int=3):
        super().__init__()
        self.dim_type  = num_types
        self.dim_tok   = tok_dim + 1
        self.dim_small = small_dim
        self.channels  = self.dim_type + self.dim_tok + self.dim_small
        def mlp():  # GINE-hez szükséges MLP
            return nn.Sequential(nn.Linear(self.channels, self.channels), nn.ReLU(), nn.Linear(self.channels, self.channels))
        self.convs = nn.ModuleList([GINEConv(mlp()) for _ in range(layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(self.channels) for _ in range(layers)])
        self.drop  = nn.Dropout(dropout)
        self.head  = nn.Sequential(nn.Linear(self.channels, self.channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.channels, 2))
        self.num_edge_types = num_edge_types
    def build_features(self, data):
        xt = data.x_type.squeeze(-1)
        h_type = F.one_hot(xt.long(), num_classes=self.dim_type).float()
        xk = data.x_tok.squeeze(-1).clamp(0, self.dim_tok-1).long()
        h_tok = F.one_hot(xk, num_classes=self.dim_tok).float()
        h_small = data.x_small
        return torch.cat([h_type, h_tok, h_small], dim=1)
    def forward(self, data): 
        h = self.build_features(data)
        eattr = F.one_hot(data.edge_type.long(), num_classes=self.num_edge_types).float() if data.edge_type.numel()>0 else None
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, data.edge_index, eattr)
            h = bn(h); h = torch.relu(h); h = self.drop(h)
        hg = global_mean_pool(h, data.batch)
        return self.head(h)

model_gine = GINENoEmb(num_types=vocab_size, tok_dim=TOK_DIM, small_dim=2, layers=4, dropout=0.3, num_edge_types=num_edge_types).to(device)
opt = torch.optim.AdamW(model_gine.parameters(), lr=1e-3, weight_decay=1e-4)  # új optimizer, külön modellhez

best_val = 0.0; best_state = None
for ep in range(1, 21):
    tr_loss, tr_acc = run_epoch(model_gine, train_loader, train=True)
    va_loss, va_acc = run_epoch(model_gine, val_loader,   train=False)
    if va_acc > best_val: best_val, best_state = va_acc, model_gine.state_dict()
    print(f"[GINE] epoch {ep:02d} | train {tr_acc:.3f} | val {va_acc:.3f}")
if best_state: model_gine.load_state_dict(best_state)
_, te_acc = run_epoch(model_gine, test_loader, train=False)
print("[GINE] TEST acc:", te_acc)
