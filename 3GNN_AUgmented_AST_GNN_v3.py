# =========================
# 0) IMPORTOK + ALAP BEÁLLÍTÁS
# =========================
import os, json, random, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path

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
# 1) PARAMÉTEREK
# =========================
SELECT_DATASET = 'draper_hf'     # 'code_x_glue' | 'draper_hf'
LANG = 'cpp'                     # 'c' | 'cpp'
MAX_SAMPLES = 20000              # 0 → mind (óvatosan RAM/VRAM miatt)
BATCH_TRAIN, BATCH_EVAL = 128, 256
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

raw_df = make_about_10pct_pos(raw_df, seed=SEED, neg_per_pos=9)

# MAX_SAMPLES alkalmazása STRATIFIKÁLTAN, hogy az arány megmaradjon
from sklearn.model_selection import train_test_split
if MAX_SAMPLES and len(raw_df) > MAX_SAMPLES:
    raw_df, _ = train_test_split(
        raw_df, train_size=MAX_SAMPLES, stratify=raw_df['label'], random_state=SEED
    )

raw_df['label'] = raw_df['label'].astype(int)
print('Kiegyensúlyozott minták száma:', len(raw_df))
print('Arányok:\n', raw_df['label'].value_counts(normalize=True).rename('proportion').round(3))
display_cols = raw_df.columns.tolist()
print("Oszlopok:", display_cols)

# =========================
# 4) KÓDNORMALIZÁLÁS (alpha-renaming, literál-helyettesítés)
#    - függvénynév: FUNC_i
#    - paraméterek: PARAM_1, PARAM_2, ...
#    - lokális változók: VAR_1, VAR_2, ...
#    - literálok: <NUM>, <STR>, <CHAR>
# Megjegyzés: ez egy gyakorlatias, egyszerűsített megoldás, a legtöbb esetet jól fedi.
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
# 5) STRATIFIKÁLT SPLIT
# =========================
df_train, df_tmp = train_test_split(raw_df, test_size=0.2, stratify=raw_df['label'], random_state=SEED)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, stratify=df_tmp['label'], random_state=SEED)
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"{name}: {len(df)} | arány:\n", df['label'].value_counts(normalize=True).round(3))

# =========================
# 6) AUGMENTED AST ÉPÍTÉS (normalizált kódból!)
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

def df_to_graphs(df: pd.DataFrame):
    out = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_code = str(row['code'])
        code = normalize_code(raw_code)           # <<< NORMALIZÁLT!
        y = int(row['label'])
        n, e = build_augmented_ast(code)
        out.append(ASTGraph(n,e,y,code))
    return out

graphs_train = df_to_graphs(df_train)
graphs_val   = df_to_graphs(df_val)
graphs_test  = df_to_graphs(df_test)
print("Gráfok:", len(graphs_train), len(graphs_val), len(graphs_test))

# =========================
# 7) PyG KONVERZIÓ + VOCAB
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

pyg_train = to_pyg(graphs_train)
pyg_val   = to_pyg(graphs_val)
pyg_test  = to_pyg(graphs_test)

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
# 8) GGNN MODELL (változatlan mélység: blocks=5, steps=10)
#    - 1 logit kimenet (BCEWithLogitsLoss)
#    - LeakyReLU a blokkokban
#    - Attention-alapú pooling (GlobalAttention)
# =========================
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GatedGraphConv, GlobalAttention, global_mean_pool

LEAKY_SLOPE = 0.1  # LeakyReLU meredekség

class GGNNBlockFeats(nn.Module):
    def __init__(self, channels: int, steps: int, num_edge_types: int = 3):
        super().__init__()
        self.num_edge_types = max(1, num_edge_types)
        self.convs = nn.ModuleList([GatedGraphConv(channels, num_layers=steps) for _ in range(self.num_edge_types)])
        self.norm = nn.LayerNorm(channels)
        self.act = nn.LeakyReLU(LEAKY_SLOPE)  
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
        return self.act(h) 

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

        # --- Attention-alapú pooling (GlobalAttention) ---
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.channels, max(1, self.channels // 2)),
                nn.LeakyReLU(LEAKY_SLOPE),
                nn.Linear(max(1, self.channels // 2), 1)
            )
        )

        # 1 logit a BCE-hez
        self.head = nn.Sequential(
            nn.Linear(self.channels,self.channels), nn.ReLU(), nn.Dropout(dropout),
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
        return self.head(hg).view(-1)  

# =========================
# 9) TRÉNING LOOP (BCEWithLogitsLoss)
# =========================
MODEL = 'ggnn'  # 'ggnn' | 'gine' 
num_edge_types = len(EDGE_TYPES)

model = GGNNClassifierFeatsNoEmb(
    num_types=vocab_size, tok_dim=TOK_DIM, small_dim=2,
    steps=10, blocks=5, num_edge_types=num_edge_types, dropout=0.3
).to(device)

lr = 4e-4
epochs = EPOCHS_GGNN
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# BCEWithLogitsLoss: pos_weight a pozitív osztály súlyozására
crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def run(loader, train=False):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        logits = model(batch)                   
        target = batch.y.float().view(-1)       
        loss = crit(logits, target)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
        loss_sum += loss.item() * batch.num_graphs
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        correct += int((pred == batch.y).sum())
        total += batch.num_graphs
    return (loss_sum/total if total else 0.0), (correct/total if total else 0.0)

def evaluate(loader):
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)               
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).long()
            y_true += batch.y.cpu().tolist()
            y_pred += pred.cpu().tolist()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

# =========================
# 10) TANÍTÁS + VALIDÁCIÓ + MENTÉS
# =========================
best_val, best_state = 0.0, None
for epoch in range(1, epochs+1):
    tr_loss, tr_acc = run(train_loader, train=True)
    va_acc, va_prec, va_rec, va_f1, _ = evaluate(val_loader)
    if va_acc > best_val:
        best_val, best_state = va_acc, model.state_dict()
    print(f"epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {va_acc:.3f} | val F1 {va_f1:.3f}")

if best_state is not None:
    model.load_state_dict(best_state)

te_acc, te_prec, te_rec, te_f1, te_cm = evaluate(test_loader)
print("TEST | acc:", te_acc, "| prec:", te_prec, "| rec:", te_rec, "| f1:", te_f1)
print("Confusion matrix:\n", te_cm)

out_name = f'cpp_augast_GGNN_bce_best.pt'
torch.save(model.state_dict(), out_name)
print('Mentve:', out_name)
