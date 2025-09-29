import os, json, random, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# --- Változók ---
from pathlib import Path
SELECT_DATASET = 'draper_hf'   # 'code_x_glue' | 'draper_hf'
LANG = 'cpp'                  # 'c' | 'cpp'
MAX_SAMPLES = 20000           # 0 → mind (óvatosan RAM/VRAM miatt)
BATCH_TRAIN, BATCH_EVAL = 64, 128
EPOCHS_GGNN, EPOCHS_GINE = 30, 20

# token-hash bucket dimenzió (levelek szövegéből)
TOK_DIM = 1024                 # 512/1024/2048 – VRAM/gyorsaság kompromisszum
TOK_SENTINEL = TOK_DIM         # üres/nem-levél → sentinel id

EDGE_TYPES = {'parent':0, 'next_sibling':1, 'next_token':2}

# --- Tree-Sitter beállítás (C/C++) ---
from tree_sitter import Language, Parser
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

TS_LANG = Language(tsc.language()) if LANG.lower()=='c' else Language(tscpp.language())
parser = Parser(TS_LANG)
print('Tree-Sitter OK, LANG =', LANG)

# --- Segédfüggvények: adatbetöltés ---
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
            if df[c].dtype==object: code_col = c; break
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
        ds = load_dataset('claudios/Draper')  # HF tükör; mezők változhatnak
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        df = pd.DataFrame({c: ds[split][c] for c in ds[split].column_names})
        ccol, lcol = _auto_pick_columns(df)
        df = df[[ccol, lcol]].rename(columns={ccol:'code', lcol:'label'})
        df['label'] = df['label'].apply(_normalize_label)
        return df.dropna(subset=['code','label']).reset_index(drop=True)
    else:
        raise ValueError(select)

raw_df = load_any_dataset(SELECT_DATASET)
if MAX_SAMPLES and len(raw_df) > MAX_SAMPLES:
    raw_df = raw_df.sample(MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
raw_df['label'] = raw_df['label'].astype(int)
print('Minták száma:', len(raw_df))
raw_df.head(3)

# Mi van ténylegesen a HF-tükörben?
from datasets import load_dataset
import pandas as pd

ds = load_dataset('claudios/Draper')
split = 'train' if 'train' in ds else list(ds.keys())[0]
df_raw = pd.DataFrame({c: ds[split][c] for c in ds[split].column_names})
print(df_raw.columns.tolist())
for c in df_raw.columns:
    vc = pd.Series(df_raw[c]).value_counts(dropna=False).head()
    print(f"\n{c} top values:\n{vc}")


# --- Stratifikált split (0.8/0.1/0.1) ---
df_train, df_tmp = train_test_split(raw_df, test_size=0.2, stratify=raw_df['label'], random_state=SEED)
df_val, df_test  = train_test_split(df_tmp,   test_size=0.5, stratify=df_tmp['label'], random_state=SEED)
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"{name}: {len(df)} | arány:\n", df['label'].value_counts(normalize=True).round(3))
    
# --- Augmented AST építés (C/C++) ---
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

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
        if parent_id is not None: edges.append((parent_id, my, 'parent'))
        if last_sib   is not None: edges.append((last_sib,   my, 'next_sibling'))
        prev = None
        for ch in children:
            ch_id = walk(ch, my, prev, depth+1)
            if prev is not None: edges.append((prev, ch_id, 'next_token'))
            prev = ch_id
        return my
    walk(tree.root_node)
    return nodes, edges

def df_to_graphs(df: pd.DataFrame):
    out = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        code, y = str(row['code']), int(row['label'])
        n,e = build_augmented_ast(code)
        out.append(ASTGraph(n,e,y,code))
    return out

graphs_train = df_to_graphs(df_train)
graphs_val   = df_to_graphs(df_val)
graphs_test  = df_to_graphs(df_test)
len(graphs_train), len(graphs_val), len(graphs_test)

# --- PyG konverzió: type + token (levelek szövege hash-bucket) + small ---
class Vocab:
    def __init__(self): self.map = {}
    def id(self, k):
        if k not in self.map: self.map[k] = len(self.map)
        return self.map[k]
    def size(self): return len(self.map)

type_vocab = Vocab()

def _hash_bucket(s: str, D: int = TOK_DIM) -> int:
    if not s or not s.strip():
        return TOK_SENTINEL
    h = hashlib.md5(s.strip().encode('utf8')).hexdigest()
    return int(h, 16) % D

def to_pyg(gs):
    pyg = []
    for g in gs:
        # type id\n
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
            edge_type  = torch.empty((0,), dtype=torch.long)
        else:
            src = [s for s,_,_ in g.edges]
            dst = [d for _,d,_ in g.edges]
            et  = [EDGE_TYPES[t] for *_,t in g.edges]
            edge_index = torch.tensor([src,dst], dtype=torch.long)
            edge_type  = torch.tensor(et, dtype=torch.long)
        data = Data(edge_index=edge_index, y=torch.tensor([g.label], dtype=torch.long))
        data.edge_type = edge_type
        data.x_type  = x_type
        data.x_tok   = x_tok
        data.x_small = x_small
        data.x = x_type.clone()  # kompat hibák ellen
        pyg.append(data)
    return pyg

pyg_train = to_pyg(graphs_train)
pyg_val   = to_pyg(graphs_val)
pyg_test  = to_pyg(graphs_test)
vocab_size = type_vocab.size()
print('PyG gráfok:', len(pyg_train), len(pyg_val), len(pyg_test), '| vocab_size =', vocab_size)

# --- DataLoaderek + osztálysúly ---
train_loader = DataLoader(pyg_train, batch_size=BATCH_TRAIN, shuffle=True)
val_loader   = DataLoader(pyg_val,   batch_size=BATCH_EVAL)
test_loader  = DataLoader(pyg_test,  batch_size=BATCH_EVAL)

y_train = np.array([int(g.y.item()) for g in pyg_train])
pos = (y_train==1).sum(); neg = (y_train==0).sum()
class_weight = torch.tensor([1.0, max(1.0, neg/max(1,pos))], dtype=torch.float)
print('Class weight:', class_weight.tolist())

# Sanity — indexek férjenek bele
batch = next(iter(train_loader))
xt = getattr(batch, 'x_type', getattr(batch, 'x'))
xk = getattr(batch, 'x_tok', None)
print('max type id:', int(xt.max()), '< vocab_size =', vocab_size)
if xk is not None and xk.numel()>0:
    print('max token id:', int(xk.max()), '<= TOK_DIM =', TOK_DIM)
    
# --- GGNN (no-embedding) ---
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool

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
        return torch.relu(h)

class GGNNClassifierFeatsNoEmb(nn.Module):
    def __init__(self, num_types:int, tok_dim:int, small_dim:int=2, steps:int=10, blocks:int=5, num_edge_types:int=3, dropout:float=0.3):
        super().__init__()
        self.dim_type=num_types; self.dim_tok=tok_dim+1; self.dim_small=small_dim
        self.channels = self.dim_type + self.dim_tok + self.dim_small
        self.blocks = nn.ModuleList([GGNNBlockFeats(self.channels, steps, num_edge_types) for _ in range(blocks)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(self.channels,self.channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(self.channels,2))
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
        hg = global_mean_pool(h, data.batch)
        return self.head(hg)

# --- GINE (GIN + edge_attr, no-embedding) ---
from torch_geometric.nn import GINEConv, global_mean_pool

class GINEClassifierFeatsNoEmb(torch.nn.Module):
    def __init__(self, num_types:int, tok_dim:int, small_dim:int=2, num_layers:int=4, dropout:float=0.3, num_edge_types:int=3):
        super().__init__()
        self.dim_type=num_types; self.dim_tok=tok_dim+1; self.dim_small=small_dim
        self.channels = self.dim_type + self.dim_tok + self.dim_small
        self.num_edge_types = num_edge_types
        def mlp():
            return torch.nn.Sequential(torch.nn.Linear(self.channels,self.channels), torch.nn.ReLU(), torch.nn.Linear(self.channels,self.channels))
        self.gins = torch.nn.ModuleList([GINEConv(mlp()) for _ in range(num_layers)])
        self.bns  = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.channels) for _ in range(num_layers)])
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Sequential(torch.nn.Linear(self.channels,self.channels), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(self.channels,2))
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
            h_tok = torch.zeros((N,self.dim_tok), dtype=torch.float, device=h_type.device); h_tok[:,self.dim_tok-1]=1.0
        h_small = getattr(data,'x_small', torch.zeros((h_type.size(0),self.dim_small), dtype=torch.float, device=h_type.device))
        return torch.cat([h_type, h_tok, h_small], dim=1)
    def forward(self, data):
        h = self.build_features(data)
        if hasattr(data,'edge_type') and data.edge_type.numel()>0:
            edge_attr = F.one_hot(data.edge_type.long(), num_classes=self.num_edge_types).float()
        else:
            E = data.edge_index.size(1)
            edge_attr = h.new_zeros((E, self.num_edge_types))
        for conv, bn in zip(self.gins, self.bns):
            h = conv(h, data.edge_index, edge_attr)
            h = bn(h)
            h = torch.relu(h)
            h = self.drop(h)
        hg = global_mean_pool(h, data.batch)
        return self.head(hg)
    
# --- Modell választó + tréning/eval ---
MODEL = 'ggnn'   # 'ggnn' | 'gine'
num_edge_types = len(EDGE_TYPES)

if MODEL=='ggnn':
    model = GGNNClassifierFeatsNoEmb(num_types=vocab_size, tok_dim=TOK_DIM, small_dim=2,
                                     steps=10, blocks=5, num_edge_types=num_edge_types, dropout=0.3).to(device)
    lr, epochs = 3e-4, EPOCHS_GGNN
else:
    model = GINEClassifierFeatsNoEmb(num_types=vocab_size, tok_dim=TOK_DIM, small_dim=2,
                                     num_layers=4, dropout=0.3, num_edge_types=num_edge_types).to(device)
    lr, epochs = 1e-3, EPOCHS_GINE

opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
crit = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))

def run(loader, train=False):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
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
        pred = logits.argmax(dim=1)
        correct += int((pred == batch.y).sum())
        total   += batch.num_graphs
    return (loss_sum/total if total else 0.0), (correct/total if total else 0.0)

def evaluate(loader):
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y_true += batch.y.cpu().tolist()
            y_pred += logits.argmax(1).cpu().tolist()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

best_val, best_state = 0.0, None
for epoch in range(1, epochs+1):
    tr_loss, tr_acc = run(train_loader, train=True)
    va_acc, va_prec, va_rec, va_f1, _ = evaluate(val_loader)
    if va_acc > best_val: best_val, best_state = va_acc, model.state_dict()
    print(f"epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {va_acc:.3f} | val F1 {va_f1:.3f}")

if best_state is not None:
    model.load_state_dict(best_state)
te_acc, te_prec, te_rec, te_f1, te_cm = evaluate(test_loader)
print("TEST | acc:", te_acc, "| prec:", te_prec, "| rec:", te_rec, "| f1:", te_f1)
print("Confusion matrix:\n", te_cm)
torch.save(model.state_dict(), f'cpp_augast_{MODEL}_best.pt')
print('Mentve:', f'cpp_augast_{MODEL}_best.pt')