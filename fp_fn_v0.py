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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from tree_sitter import Language, Parser

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# =========================
# ALAP KONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SELECT_DATASET = 'draper_hf'     # 'code_x_glue' | 'draper_hf'
LANG = 'cpp'                     # 'c' | 'cpp'
MAX_SAMPLES = 20000              # 0 → mind (óvatosan RAM/VRAM miatt)
BATCH_TRAIN, BATCH_EVAL = 64, 128
EPOCHS_GGNN, EPOCHS_GINE = 30, 20

# one-hot hash bucket a levelek szövegéhez (normalizálás miatt lehet kicsi)
TOK_DIM = 128                    # korábban 1024; normalizálással bőven elé
TOK_SENTINEL = TOK_DIM           # "üres" token bucket ID

use_amp = (device.type == "cuda")


# =========================
# TREE-SITTER INIT
# =========================
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

# LANG változó maradhat, ahogy eddig volt: 'c' vagy 'cpp'

TS_LANG = Language(tsc.language()) if LANG.lower() == 'c' else Language(tscpp.language())
parser = Parser(TS_LANG)
print('Tree-Sitter OK, LANG =', LANG)



# =========================
# 1) DATASET BETÖLTÉS
# =========================
def _normalize_label(x):
    # különböző dataseteknél előfordulhat, hogy a target nem 0/1
    try:
        v = int(x)
        return 1 if v != 0 else 0
    except Exception:
        # ha string, akkor pl. 'buggy'/'clean'
        xs = str(x).lower()
        if xs in {'1', 'true', 'bug', 'buggy', 'vulnerable'}:
            return 1
        return 0

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
        # CSAK átnevezzük a kód/label oszlopot, a többi meta (pl. hibatípus/CWE) megmarad
        df = df.rename(columns={ccol: 'code', lcol: 'label'})
        df['label'] = df['label'].apply(_normalize_label)
        return df.dropna(subset=['code','label']).reset_index(drop=True)
    else:
        raise ValueError(select)

raw_df = load_any_dataset(SELECT_DATASET)

# --- Kiegyensúlyozás: pozitívak + mintavételezett negatívak ---
def make_about_10pct_pos(df: pd.DataFrame, seed: int = SEED, neg_per_pos: int = 9) -> pd.DataFrame:
    df = df.copy()
    df['label'] = df['label'].astype(int)
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    if len(df_pos) == 0:
        raise ValueError("Nincs pozitív minta a datasetben, nem lehet kiegyensúlyozni.")

    target_neg = min(len(df_neg), neg_per_pos * len(df_pos))   
    df_neg_sampled = df_neg.sample(target_neg, random_state=seed)

    df_bal = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_bal

raw_df = make_about_10pct_pos(raw_df, seed=SEED, neg_per_pos=3)

# MAX_SAMPLES alkalmazása STRATIFIKÁLTAN, hogy az arány megmaradjon
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
# =========================
def normalize_code(code: str) -> str:
    text = code.encode("utf8")
    try:
        tree = parser.parse(text)
        root = tree.root_node
    except Exception:
        # parser hiba esetén fallback: nagyon egyszerű normalizálás
        import re
        code = text.decode("utf8", "ignore")
        code = re.sub(r'\\"([^"\\\\]|\\\\.)*\\"', '<STR>', code)
        code = re.sub(r"\'([^'\\\\]|\\\\.)*\'", "<CHR>", code)
        return code

    counters = {"func": 0, "var": 0, "param": 0}
    scope_stack = [{"vars": {}, "params": {}, "func": None}]

    def enter_scope():
        scope_stack.append({"vars": {}, "params": {}, "func": scope_stack[-1].get("func")})

    def leave_scope():
        scope_stack.pop()

    def get_id(node):
        return text[node.start_byte:node.end_byte].decode("utf8", "ignore")

    def assign_var(name, scope):
        if name not in scope["vars"]:
            counters["var"] += 1
            scope["vars"][name] = f"VAR_{counters['var']}"

    def assign_param(name, scope):
        if name not in scope["params"]:
            counters["param"] += 1
            scope["params"][name] = f"PARAM_{counters['param']}"

    def first_pass(node):
        t = node.type
        if t in ("function_definition", "function_declarator"):
            enter_scope()
            func_decl = None
            for ch in node.children:
                if ch.type in ("function_declarator", "declarator"):
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
            for ch in node.children:
                first_pass(ch)
            leave_scope()
            return

        if t in ("compound_statement", "block"):
            enter_scope()
            for ch in node.children:
                first_pass(ch)
            leave_scope()
            return

        if t in ("declaration",):
            for ch in node.children:
                if ch.type == "init_declarator" or ch.type == "identifier":
                    if ch.type == "identifier":
                        assign_var(get_id(ch), scope_stack[-1])

        for ch in node.children:
            first_pass(ch)

    def lookup_identifier(name):
        for sc in reversed(scope_stack):
            if name in sc["params"]:
                return sc["params"][name]
            if name in sc["vars"]:
                return sc["vars"][name]
        return name

    def second_pass(node):
        t = node.type
        if t == "identifier":
            name = get_id(node)
            return lookup_identifier(name).encode("utf8")
        if len(node.children) == 0:
            return text[node.start_byte:node.end_byte]

        parts = []
        for ch in node.children:
            parts.append(second_pass(ch))
        return b"".join(parts)

    try:
        first_pass(root)
        out_bytes = second_pass(root)
        return out_bytes.decode("utf8", "ignore")
    except Exception:
        import re
        code = text.decode("utf8", "ignore")
        code = re.sub(r'\\"([^"\\\\]|\\\\.)*\\"', '<STR>', code)
        code = re.sub(r"\'([^'\\\\]|\\\\.)*\'", "<CHR>", code)
        return code


# =====================
# 3) STRATIFIKÁLT SPLIT
# =====================
df_train, df_tmp = train_test_split(raw_df, test_size=0.2, stratify=raw_df['label'], random_state=SEED)
df_val, df_test = train_test_split(df_tmp, test_size=0.5, stratify=df_tmp['label'], random_state=SEED)
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    print(f"{name}: {len(df)} | arány:\n", df['label'].value_counts(normalize=True).round(3))


# =========================
# 5) AUGMENTED AST ÉPÍTÉS
# =========================
@dataclass
class ASTGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, str]]
    label: int
    raw: str            # normalizált kód
    idx: int            # sor indexe a df-ben (hiba-típus visszakereséshez)

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
            'is_leaf': int(len(children) == 0),
            'depth': depth,
            'text': snippet.decode('utf8', 'ignore')
        })
        if parent_id is not None:
            edges.append((parent_id,my,'parent'))
        if last_sib is not None:
            edges.append((last_sib,my,'next_sibling'))
        prev = None
        for ch in children:
            ch_id = walk(ch, my, prev, depth+1)
            if prev is not None:
                edges.append((prev,ch_id,'next_token'))
            prev = ch_id
        return my
    walk(tree.root_node)
    return nodes, edges

EDGE_TYPES = {
    'parent': 0,
    'next_sibling': 1,
    'next_token': 2
}

def df_to_graphs(df: pd.DataFrame):
    out = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_code = str(row['code'])
        code = normalize_code(raw_code)           # <<< NORMALIZÁLT!
        y = int(row['label'])
        n, e = build_augmented_ast(code)
        out.append(ASTGraph(n, e, y, code, int(row.name)))
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
        if len(g.edges) == 0:
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
        data.x_type = x_type
        data.x_tok = x_tok
        data.x_small = x_small
        data.x = x_type.clone()  # kompatibilitás miatt
        # csak egy egész index, hogy a df-ben visszakeressük a hibatípust/nyers kódot
        data.sample_idx = torch.tensor([g.idx], dtype=torch.long)
        pyg.append(data)
    return pyg

pyg_train = to_pyg(graphs_train)
pyg_val   = to_pyg(graphs_val)
pyg_test  = to_pyg(graphs_test)

vocab_size = type_vocab.size()
print('PyG gráfok:', len(pyg_train), len(pyg_val), len(pyg_test), '| vocab_size =', vocab_size)

train_loader = DataLoader(pyg_train, batch_size=BATCH_TRAIN, shuffle=True)
val_loader   = DataLoader(pyg_val,   batch_size=BATCH_EVAL)
test_loader  = DataLoader(pyg_test,  batch_size=BATCH_EVAL)

y_train = np.array([int(g.y.item()) for g in pyg_train])
pos = (y_train==1).sum()
neg = (y_train==0).sum()
pos_weight = torch.tensor([max(1.0, neg/max(1,pos))], dtype=torch.float, device=device)
print('pos_weight:', pos_weight.item())


# =========================
# 8) MODELL
# =========================
class GGNNClassifierFeatsNoEmb(torch.nn.Module):
    def __init__(
        self,
        num_types: int,
        tok_dim: int,
        small_dim: int,
        steps: int,
        blocks: int,
        num_edge_types: int,
        hidden_dim: int = 64,
        proj_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_types = num_types
        self.tok_dim = tok_dim
        self.small_dim = small_dim
        self.steps = steps
        self.blocks = blocks
        self.num_edge_types = num_edge_types

        in_dim = 32 + 32 + small_dim
        self.type_emb = torch.nn.Embedding(num_types, 32)
        self.tok_emb  = torch.nn.Embedding(tok_dim+1, 32)

        self.msg_linears = torch.nn.ModuleList([
            torch.nn.Linear(in_dim, in_dim) for _ in range(num_edge_types)
        ])
        self.gru = torch.nn.GRU(
            input_size=in_dim,
            hidden_size=in_dim,
            batch_first=True
        )

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_dim, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(proj_dim, 1)
        )

    def forward(self, data: Data):
        x_type = data.x_type
        x_tok  = data.x_tok
        x_small = data.x_small
        edge_index = data.edge_index
        edge_type  = data.edge_type
        batch = data.batch

        type_emb = self.type_emb(x_type).squeeze(1)
        tok_emb  = self.tok_emb(x_tok).squeeze(1)
        x = torch.cat([type_emb, tok_emb, x_small], dim=-1)

        for _ in range(self.blocks):
            for _ in range(self.steps):
                msgs = []
                for et in range(self.num_edge_types):
                    mask = (edge_type == et)
                    if mask.sum() == 0:
                        msgs.append(torch.zeros_like(x))
                        continue
                    e_idx = edge_index[:,mask]
                    src, dst = e_idx[0], e_idx[1]
                    m = self.msg_linears[et](x[src])
                    agg = torch.zeros_like(x)
                    agg.index_add_(0, dst, m)
                    msgs.append(agg)
                h_in = x.unsqueeze(0)
                m_all = sum(msgs).unsqueeze(0)
                h_out, _ = self.gru(m_all, h_in)
                x = h_out.squeeze(0)

        out = torch_geometric.nn.global_mean_pool(x, batch)
        logits = self.proj(out).view(-1)
        return logits

MODEL = 'ggnn'
num_edge_types = len(EDGE_TYPES)
model = GGNNClassifierFeatsNoEmb(
    num_types=vocab_size,
    tok_dim=TOK_DIM,
    small_dim=2,
    steps=10,
    blocks=5,
    num_edge_types=num_edge_types,
    dropout=0.3
).to(device)

lr, epochs = 3e-4, EPOCHS_GGNN
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# =========================
# 9) TRÉNING / EVAL
# =========================
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(opt)
            scaler.update()

        loss_sum += loss.item() * batch.num_graphs
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        correct += int((pred == batch.y).sum())
        total   += batch.num_graphs
    return (loss_sum/total if total else 0.0), (correct/total if total else 0.0)

def find_best_threshold(loader):
    model.eval(); y_true, probs = [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                probs += torch.sigmoid(logits).cpu().tolist()
                y_true += batch.y.cpu().tolist()
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05,0.95,19):
        y_pred = [1 if p>=thr else 0 for p in probs]
        f1 = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)[2]
        if f1>best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

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


# --- FALSE NEGATIVE / FALSE POSITIVE GYŰJTÉSE + HIBATÍPUS STATISZTIKA ---

# lehetséges hibatípus/osztály oszlopnevek (pl. Draper-ben tipikusan a CWE mező)
ERROR_TYPE_CANDIDATES = [
    'CWE', 'cwe', 'bug_type', 'BugType', 'vuln_type', 'vul_type',
    'error_type', 'ErrorType', 'vulnerability', 'vul_category'
]

def guess_error_type_column(df: pd.DataFrame):
    for c in df.columns:
        if c in ERROR_TYPE_CANDIDATES:
            return c
        if c.lower() in [x.lower() for x in ERROR_TYPE_CANDIDATES]:
            return c
    return None

def collect_misclassified(loader, thr, df_split: pd.DataFrame):
    """Visszaadja a (FN, FP) DataFrame-eket az adott splithez."""
    model.eval(); rows_fn, rows_fp = [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                prob = torch.sigmoid(logits)
                pred = (prob >= thr).long()
                for i in range(batch.num_graphs):
                    y_true = int(batch.y[i].item())
                    y_pred = int(pred[i].item())
                    idx = int(batch.sample_idx[i].item())
                    row = df_split.loc[idx]
                    if y_true == 1 and y_pred == 0:
                        rows_fn.append(row)
                    elif y_true == 0 and y_pred == 1:
                        rows_fp.append(row)
    return pd.DataFrame(rows_fn), pd.DataFrame(rows_fp)


# --- TANÍTÁS ---
best_val, best_state = 0.0, None
for epoch in range(1, epochs+1):
    tr_loss, tr_acc = run(train_loader, train=True)
    va_acc, va_prec, va_rec, va_f1, _ = evaluate(val_loader, thr=0.5)
    if va_f1>best_val:
        best_val, best_state = va_f1, model.state_dict()
    print(f"epoch {epoch:02d} | train acc {tr_acc:.3f} | val acc {va_acc:.3f} | val F1 {va_f1:.3f}")

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

# --- False negative / false positive minták összegyűjtése a TEST halmazon ---
df_fn_test, df_fp_test = collect_misclassified(test_loader, best_thr, df_test)
print("False negative-ek száma (sebezhető, de nem detektált):", len(df_fn_test))
print("False positive-ok száma (nem sebezhető, de sebezhetőnek jelölt):", len(df_fp_test))

# Hibatípus szerinti statisztika (ha van ilyen oszlop)
err_col = guess_error_type_column(df_test)
if err_col is not None:
    print(f"Detektált hibatípus oszlop: {err_col}")
    if not df_fn_test.empty:
        print("\nFalse negative-ek hibatípus szerint (TEST):")
        print(df_fn_test[err_col].value_counts(dropna=False))
    if not df_fp_test.empty:
        print("\nFalse positive-ok hibatípus szerint (TEST):")
        print(df_fp_test[err_col].value_counts(dropna=False))
else:
    print("\nNincs egyértelmű hibatípus oszlop a DataFrame-ben, csak a darabszámok látszanak.")

# (opcionális) a nyers függvénykódokat is elmentheted, ha a 'code' oszlop a nyers kód:
if 'code' in df_fn_test.columns and len(df_fn_test) > 0:
    df_fn_test['code'].to_csv("test_false_negatives_rawcode.txt", index=False, header=False)
    print("FN kódok mentve: test_false_negatives_rawcode.txt")
if 'code' in df_fp_test.columns and len(df_fp_test) > 0:
    df_fp_test['code'].to_csv("test_false_positives_rawcode.txt", index=False, header=False)
    print("FP kódok mentve: test_false_positives_rawcode.txt")

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
