import os, json, hashlib, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from tree_sitter import Language, Parser
import tree_sitter_c as tsc
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Tree-Sitter C nyelv
C_LANG = Language(tsc.language())
parser = Parser(C_LANG)

@dataclass
class ASTGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, str]]  # (src, dst, edge_type)
    label: int
    raw: str

def parse_c(code: str):
    return parser.parse(code.encode('utf8'))

def build_augmented_ast(tree, source: bytes) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, str]]]:
    nodes, edges = [], []
    node_id = 0
    def walk(node, parent_id=None, last_sibling_id=None, depth=0):
        nonlocal node_id
        my_id = node_id; node_id += 1
        snippet = source[node.start_byte:node.end_byte]
        children = node.children
        nodes.append({
            'id': my_id,
            'type': node.type,
            'start_point': node.start_point,
            'end_point': node.end_point,
            'text': snippet.decode('utf8', 'ignore'),
            'is_leaf': int(len(children) == 0),
            'depth': depth
        })
        if parent_id is not None:
            edges.append((parent_id, my_id, 'parent'))
        if last_sibling_id is not None:
            edges.append((last_sibling_id, my_id, 'next_sibling'))
        prev_child_id = None
        for child in children:
            child_id = walk(child, my_id, prev_child_id, depth+1)
            if prev_child_id is not None:
                edges.append((prev_child_id, child_id, 'next_token'))
            prev_child_id = child_id
        return my_id
    walk(tree.root_node)
    return nodes, edges

# Devign (HF tükör) betöltés
ds = load_dataset('google/code_x_glue_cc_defect_detection')
train, valid, test = ds['train'], ds.get('validation') or ds.get('valid') or None, ds['test']
print(train); print('példa label:', train[0]['target']); print(train[0]['func'][:200])

# kiegyensúlyozott random részhalmaz (állítható)
N_TRAIN = 20000  # növelhető, ha fér a VRAM/ram
pos = train.filter(lambda x: x['target']==1)
neg = train.filter(lambda x: x['target']==0)
k = min(N_TRAIN//2, len(pos), len(neg))
subset = concatenate_datasets([
    pos.shuffle(seed=SEED).select(range(k)),
    neg.shuffle(seed=SEED+1).select(range(k)),
]).shuffle(seed=SEED+2)
len(subset)

# ASTGraph-okká alakítás\n
def sample_to_astgraph(sample) -> ASTGraph:
    code = sample['func']; label = int(sample['target'])
    tree = parse_c(code)
    nodes, edges = build_augmented_ast(tree, code.encode('utf8'))
    return ASTGraph(nodes=nodes, edges=edges, label=label, raw=code)

graphs_ast = [sample_to_astgraph(row) for row in tqdm(subset, total=len(subset))]
len(graphs_ast), graphs_ast[0].label, graphs_ast[0].nodes[0]['type']

# AugAST -> PyG konverzió (type + token + small)
EDGE_TYPES = {'parent':0, 'next_sibling':1, 'next_token':2}
TOK_DIM = 1024              # token-hash bucket; sentinel = TOK_DIM
TOK_SENTINEL = TOK_DIM

def _hash_bucket(s: str, D: int = TOK_DIM) -> int:
    if not s or not s.strip():
        return TOK_SENTINEL
    h = hashlib.md5(s.strip().encode('utf8')).hexdigest()
    return int(h, 16) % D

def to_pyg(g: ASTGraph) -> Data:
    # type vocab (globális)
    if not hasattr(to_pyg, 'type_vocab'):
        to_pyg.type_vocab = {}
    tv = to_pyg.type_vocab

    type_ids, tok_ids, small_feats = [], [], []
    max_depth = max([n.get('depth',0) for n in g.nodes] + [1])
    for n in g.nodes:
        t = n['type']
        if t not in tv: tv[t] = len(tv)
        type_ids.append([tv[t]])
        tok = _hash_bucket(n.get('text','') if n.get('is_leaf',0) else '', TOK_DIM)
        tok_ids.append([tok])
        d = float(n.get('depth',0))/float(max_depth)
        small_feats.append([float(n.get('is_leaf',0)), d])

    x_type  = torch.tensor(np.array(type_ids),  dtype=torch.long)
    x_tok   = torch.tensor(np.array(tok_ids),   dtype=torch.long)
    x_small = torch.tensor(np.array(small_feats), dtype=torch.float)

    if len(g.edges)==0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_type  = torch.empty((0,),  dtype=torch.long)
    else:
        src = [s for (s,_,_) in g.edges]; dst = [d for (_,d,_) in g.edges]
        et  = [EDGE_TYPES[t] for (_,_,t) in g.edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type  = torch.tensor(et, dtype=torch.long)

    data = Data(edge_index=edge_index, y=torch.tensor([g.label], dtype=torch.long))
    data.edge_type = edge_type
    data.x_type  = x_type
    data.x_tok   = x_tok
    data.x_small = x_small
    data.x = x_type.clone()  # kompat
    return data

pyg_graphs = [to_pyg(g) for g in tqdm(graphs_ast)]
torch.save(pyg_graphs, 'devign_augast_pyg.pt')
len(pyg_graphs), list(to_pyg.type_vocab)[:5], len(to_pyg.type_vocab)

# Betöltés + stratifikált split + DataLoaderek
graphs = torch.load('devign_augast_pyg.pt', weights_only=False)
ys = np.array([int(g.y.item()) for g in graphs])
pos_idx = np.where(ys==1)[0]; neg_idx = np.where(ys==0)[0]
rng = np.random.default_rng(SEED)
rng.shuffle(pos_idx); rng.shuffle(neg_idx)
def split3(idx):
    n=len(idx); n_tr=int(0.8*n); n_va=int(0.1*n); return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
p_tr,p_va,p_te = split3(pos_idx); n_tr,n_va,n_te = split3(neg_idx)
train_idx = np.concatenate([p_tr,n_tr]); rng.shuffle(train_idx)
val_idx   = np.concatenate([p_va,n_va]); rng.shuffle(val_idx)
test_idx  = np.concatenate([p_te,n_te]); rng.shuffle(test_idx)
train_set = [graphs[i] for i in train_idx]
val_set   = [graphs[i] for i in val_idx]
test_set  = [graphs[i] for i in test_idx]
class_weight = torch.tensor([1.0, max(1.0, len(n_tr)/max(1,len(p_tr)))], dtype=torch.float)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=128)
test_loader  = DataLoader(test_set,  batch_size=128)
print('Train/Val/Test:', len(train_set), len(val_set), len(test_set))

# Dimenziók kikövetkeztetése + sanity\n
def infer_vocab_size(gs):
    mx=0
    for g in gs:
        if hasattr(g,'x_type') and g.x_type.numel()>0:
            mx = max(mx, int(g.x_type.max().item()))
        elif hasattr(g,'x') and g.x.numel()>0:
            mx = max(mx, int(g.x.max().item()))
    return mx+1

def infer_tok_dim(gs):
    mx=-1
    for g in gs:
        if hasattr(g,'x_tok') and g.x_tok.numel()>0:
            mx = max(mx, int(g.x_tok.max().item()))
    return (mx if mx>=0 else TOK_DIM)

vocab_size = infer_vocab_size(train_set+val_set+test_set)
TOK_DIM_INFER = infer_tok_dim(train_set+val_set+test_set)
print('vocab_size =', vocab_size, '| TOK_DIM =', TOK_DIM_INFER)

# gyors sanity (CPU-n)
batch = next(iter(train_loader))
xt = getattr(batch,'x_type', getattr(batch,'x'))
xk = getattr(batch,'x_tok', None)
print('max xt:', int(xt.max()))
print('max xk:', int(xk.max()) if xk is not None and xk.numel()>0 else -1)

# ==== GGNN (no-emb) – type+token+small ====
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
            for t,conv in enumerate(self.convs):
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
            h_tok = torch.zeros((N,self.dim_tok), dtype=torch.float, device=h_type.device); h_tok[:,self.dim_tok-1]=1.0
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
    
# ==== GINE (no-emb) – type+token+small + edge_type ====
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
    
# ==== Modell választás, init, tréning ====#
MODEL = 'ggnn'   # 'ggnn' | 'gine'
num_edge_types = 3 if hasattr(train_set[0],'edge_type') and train_set[0].edge_type.numel()>0 else 1
tok_dim = TOK_DIM_INFER  # következtetett dimenzió a betöltött gráfokból

if MODEL=='ggnn':
    model = GGNNClassifierFeatsNoEmb(num_types=vocab_size, tok_dim=tok_dim, small_dim=2, steps=10, blocks=5, num_edge_types=num_edge_types, dropout=0.3).to(device)
else:
    model = GINEClassifierFeatsNoEmb(num_types=vocab_size, tok_dim=tok_dim, small_dim=2, num_layers=4, dropout=0.3, num_edge_types=num_edge_types).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if MODEL!='ggnn' else 3e-4, weight_decay=1e-4)
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
    return loss_sum/total, correct/total

best_val = 0.0
for epoch in range(1, 21 if MODEL!='ggnn' else 31):
    tr_loss, tr_acc = run(train_loader, train=True)
    va_loss, va_acc = run(val_loader,   train=False)
    if va_acc > best_val:
        best_val = va_acc
        torch.save(model.state_dict(), f'best_{MODEL}.pt')
    print(f'epoch {epoch:02d} | train {tr_acc:.3f} | val {va_acc:.3f}')

model.load_state_dict(torch.load(f'best_{MODEL}.pt', weights_only=False))
te_loss, te_acc = run(test_loader, train=False)
print('TEST acc:', te_acc)