# infer_vuln.py
# ----------------
# Használat:
#   BUNDLE checkpointtal (ajánlott):
#     python infer_vuln.py --lang cpp --bundle checkpoint_bundle_20250101_120000.pt --file snippet.cpp
#     python infer_vuln.py --lang cpp --bundle checkpoint_bundle_...pt --code "int foo(){ return 0; }"
#   Csak SÚLY fájllal (.pt):
#     python infer_vuln.py --lang cpp --weights cpp_augast_GGNN_bce_best.pt --vocab-size 231 --file snippet.cpp
#     # (opcionális) ha van mentett type_vocab.json:
#     python infer_vuln.py --lang cpp --weights ...pt --vocab-size 231 --vocab-json type_vocab.json --file snippet.cpp
#
# Kimenet: "prob_vulnerable: 0.6623" és (ha megadsz küszöböt) egy bináris címke is.

import os
import argparse
import json
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, GlobalAttention

# --- GPU beállítás (opcionális, de segít a fragmentáció ellen) ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# --- Tree-sitter C/C++ ---
from tree_sitter import Language, Parser
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

# ======== model & featurization paramok (a tréninggel egyezzenek!) ========
TOK_DIM = 128
TOK_SENTINEL = TOK_DIM
EDGE_TYPES = {'parent':0, 'next_sibling':1, 'next_token':2}
SMALL_DIM = 2
STEPS = 10
BLOCKS = 5
DROPOUT = 0.3

# ================== utils ==================
def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

device = pick_device()

@dataclass
class ASTGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, str]]

# ---------- normalizálás ----------
def make_parser(lang:str):
    ts_lang = Language(tsc.language()) if lang.lower()=='c' else Language(tscpp.language())
    p = Parser(ts_lang)
    return p

def normalize_code(code: str, parser: Parser) -> str:
    text = code.encode("utf8")
    try:
        tree = parser.parse(text)
        root = tree.root_node
    except Exception:
        import re
        code = code
        code = re.sub(r'\"([^"\\]|\\.)*\"', '<STR>', code)
        code = re.sub(r"\'([^'\\]|\\.)*\'", '<CHAR>', code)
        code = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', code)
        return code

    scope_stack = []; counters = {"func":0}
    def push(): scope_stack.append({"func":None,"params":{}, "vars":{}})
    def pop(): scope_stack.pop()
    def tid(node): return text[node.start_byte:node.end_byte].decode("utf8","ignore")
    def aparam(name, sc):
        if name in sc["params"]: return sc["params"][name]
        idx=len(sc["params"])+1; sc["params"][name]=f"PARAM_{idx}"; return sc["params"][name]
    def avar(name, sc):
        if name in sc["vars"]: return sc["vars"][name]
        idx=len(sc["vars"])+1; sc["vars"][name]=f"VAR_{idx}"; return sc["vars"][name]
    LITS={"number_literal":"<NUM>","string_literal":"<STR>","char_literal":"<CHAR>"}

    def first(node):
        t=node.type
        if t=="function_definition":
            push()
            func_decl=None
            for ch in node.children:
                if ch.type=="function_declarator":
                    func_decl=ch; break
            if func_decl is not None:
                for ch in func_decl.children:
                    if ch.type=="identifier":
                        counters["func"]+=1
                        scope_stack[-1]["func"]=f"FUNC_{counters['func']}"
                        break
                for ch in func_decl.children:
                    if ch.type=="parameter_list":
                        for p in ch.children:
                            if p.type=="parameter_declaration":
                                for gch in p.children:
                                    if gch.type=="identifier":
                                        aparam(tid(gch), scope_stack[-1])
            for ch in node.children:
                first(ch)
            pop(); return
        if scope_stack:
            if t in {"init_declarator","declarator"}:
                for ch in node.children:
                    if ch.type=="identifier":
                        avar(tid(ch), scope_stack[-1])
        for ch in node.children:
            first(ch)

    def lookup(name):
        for sc in reversed(scope_stack):
            if name in sc["params"]: return sc["params"][name]
            if name in sc["vars"]: return sc["vars"][name]
        return name

    def second(node):
        t=node.type
        if t=="function_definition":
            push(); out=[]
            for ch in node.children: out.append(second(ch))
            pop(); return "".join(out)
        if t in LITS: return LITS[t]
        if t=="identifier":
            if scope_stack and scope_stack[-1]["func"] is not None:
                parent=node.parent
                if parent and parent.type=="function_declarator":
                    return scope_stack[-1]["func"]
            return lookup(tid(node))
        if len(node.children)==0:
            return text[node.start_byte:node.end_byte].decode("utf8","ignore")
        parts=[second(ch) for ch in node.children]
        return "".join(parts)

    try:
        first(root)
        return second(root)
    except Exception:
        import re
        code=text.decode("utf8","ignore")
        code = re.sub(r'\"([^"\\]|\\.)*\"', '<STR>', code)
        code = re.sub(r"\'([^'\\]|\\.)*\'", '<CHAR>', code)
        code = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', code)
        return code

# ---------- AST → gráf ----------
def build_augmented_ast(code: str, parser: Parser):
    tree = parser.parse(code.encode('utf8'))
    nodes, edges = [], []; nid=0
    def walk(node, parent_id=None, last_sib=None, depth=0):
        nonlocal nid
        my=nid; nid+=1
        snippet = code.encode('utf8')[node.start_byte:node.end_byte]
        children = node.children
        nodes.append({'id':my,'type':node.type,'is_leaf':int(len(children)==0),'depth':depth,'text':snippet.decode('utf8','ignore')})
        if parent_id is not None: edges.append((parent_id,my,'parent'))
        if last_sib is not None: edges.append((last_sib,my,'next_sibling'))
        prev=None
        for ch in children:
            ch_id = walk(ch, my, prev, depth+1)
            if prev is not None:
                edges.append((prev, ch_id, 'next_token'))
            prev=ch_id
        return my
    walk(tree.root_node)
    return ASTGraph(nodes, edges)

# ---------- Vocab + featurization ----------
class FrozenVocab:
    """read-only string->int térkép; nem bővít automatikusan"""
    def __init__(self, mapping: Dict[str,int], default_id: int = 0):
        self.map = dict(mapping)
        self.default_id = int(default_id)
        self._size = int(max(mapping.values())+1 if mapping else max(1, self.default_id+1))
    def id(self, k:str)->int:
        return self.map.get(k, self.default_id)
    def size(self)->int:
        return self._size

def _hash_bucket(s: str, D: int = TOK_DIM) -> int:
    if not s or not s.strip(): return TOK_SENTINEL
    h = hashlib.md5(s.strip().encode('utf8')).hexdigest()
    return int(h, 16) % D

def graph_to_pyg(g: ASTGraph, type_vocab: FrozenVocab) -> Data:
    type_ids = [[type_vocab.id(n['type'])] for n in g.nodes]
    x_type = torch.tensor(np.array(type_ids), dtype=torch.long)

    tok_ids = [[_hash_bucket(n.get('text','') if n.get('is_leaf',0) else '', TOK_DIM)] for n in g.nodes]
    x_tok = torch.tensor(np.array(tok_ids), dtype=torch.long)

    max_depth = max([n.get('depth',0) for n in g.nodes] + [1])
    small = [[float(n.get('is_leaf',0)), float(n.get('depth',0))/float(max_depth)] for n in g.nodes]
    x_small = torch.tensor(np.array(small), dtype=torch.float)

    if len(g.edges)==0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    else:
        src = [s for s,_,_ in g.edges]; dst = [d for _,d,_ in g.edges]
        et  = [EDGE_TYPES[t] for *_,t in g.edges]
        edge_index = torch.tensor([src,dst], dtype=torch.long)
        edge_type  = torch.tensor(et, dtype=torch.long)

    data = Data(edge_index=edge_index, y=torch.tensor([0], dtype=torch.long))
    data.edge_type = edge_type
    data.x_type = x_type
    data.x_tok = x_tok
    data.x_small = x_small
    data.x = x_type.clone()
    return data

# ---------- Modell ----------
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
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.channels, self.channels // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(self.channels // 2, 1)
            )
        )
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

# ---------- fő logika ----------
def load_bundle(path:str):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError("A bundle nem tartalmaz 'state_dict' és 'config' kulcsokat.")
    cfg = ckpt["config"]
    model = GGNNClassifierFeatsNoEmb(
        num_types=cfg["num_types"], tok_dim=cfg["tok_dim"], small_dim=cfg["small_dim"],
        steps=cfg["steps"], blocks=cfg["blocks"], num_edge_types=cfg["num_edge_types"], dropout=cfg["dropout"]
    )
    model.load_state_dict(ckpt["state_dict"])
    vocab_map = ckpt.get("type_vocab", {})
    best_thr = float(ckpt.get("best_thr", 0.5))
    return model, FrozenVocab(vocab_map, default_id=0), best_thr

def load_weights(path:str, vocab_size:int, vocab_json:str|None):
    # vocab térkép (ha nincs, üres -> ismeretlen típusok 0-ra esnek)
    vmap = {}
    if vocab_json and os.path.isfile(vocab_json):
        with open(vocab_json, "r") as f:
            vmap = json.load(f)
    model = GGNNClassifierFeatsNoEmb(
        num_types=vocab_size, tok_dim=TOK_DIM, small_dim=SMALL_DIM,
        steps=STEPS, blocks=BLOCKS, num_edge_types=len(EDGE_TYPES), dropout=DROPOUT
    )
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model, FrozenVocab(vmap, default_id=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["c","cpp"], default="cpp")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="C/C++ fájl/függvény forrása")
    src.add_argument("--code", help="kód snippet sztringben")
    ck = ap.add_mutually_exclusive_group(required=True)
    ck.add_argument("--bundle", help="komplett checkpoint bundle (.pt)")
    ck.add_argument("--weights", help="csak súly fájl (.pt)")
    ap.add_argument("--vocab-size", type=int, help="ha --weights: tréningkori vocab_size (pl. 231)")
    ap.add_argument("--vocab-json", help="ha --weights: tréningkori type_vocab JSON (opcionális)")
    ap.add_argument("--threshold", type=float, default=None, help="opcionális döntési küszöb a labelhez")
    args = ap.parse_args()

    parser = make_parser(args.lang)

    if args.bundle:
        model, type_vocab, best_thr = load_bundle(args.bundle)
        if args.threshold is None:
            args.threshold = best_thr
    else:
        if args.vocab_size is None:
            raise SystemExit("--weights használatakor add meg a --vocab-size értéket (pl. 231)!")
        model, type_vocab = load_weights(args.weights, args.vocab_size, args.vocab_json)

    model.to(device).eval()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw_code = f.read()
    else:
        raw_code = args.code

    norm = normalize_code(raw_code, parser)
    g = build_augmented_ast(norm, parser)
    data = graph_to_pyg(g, type_vocab)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)  # egy gráf → batch index 0

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(device.type=="cuda")):
            logits = model(data.to(device))
            prob = torch.sigmoid(logits)[0].item()

    print(f"prob_vulnerable: {prob:.6f}")
    if args.threshold is not None:
        label = int(prob >= float(args.threshold))
        print(f"label@thr={float(args.threshold):.3f}: {label}")

if __name__ == "__main__":
    main()
