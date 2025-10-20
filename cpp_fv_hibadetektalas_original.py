#!/usr/bin/env python3
"""
Standalone inference script for your GGNN-based C/C++ vulnerability detector.

Given a C/C++ function snippet (e.g., from a .txt file), it:
  1) Normalizes the code (alpha-renaming, literal replacement)
  2) Builds the augmented AST graph (parent, next_sibling, next_token)
  3) Converts it to a PyG Data object using the *training-time* vocab + token hashing
  4) Loads the saved model checkpoint and runs a forward pass
  5) Returns the probability of being buggy (sigmoid output)

USAGE EXAMPLES
--------------
python infer_cpp_bugprob.py --model cpp_augast_GGNN_bce_best.pt --meta cpp_augast_meta.json --input snippet.txt
python infer_cpp_bugprob.py --model cpp_augast_GGNN_bce_best.pt --meta cpp_augast_meta.json --code "int f(int x){return x+1;}"

DEPENDENCIES
------------
- torch, torch_geometric, tree_sitter, tree_sitter_cpp (and/or tree_sitter_c)
- numpy, pandas (pandas not required strictly here), tqdm (optional)

NOTES
-----
- The meta JSON must contain the training vocabulary mapping for node types
  and the token hashing dimension. See the README section at the bottom of this
  file for how to produce it from your training code.
"""

import os
import json
import argparse
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data

# =========================
# 0) DEVICE PICKER
# =========================

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    try:
        import torch_directml  # type: ignore
        return torch_directml.device()
    except Exception:
        pass
    return torch.device("cpu")

DEVICE = pick_device()

# =========================
# 1) TREE-SITTER INIT (C/C++)
# =========================
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
# If you also want C support, import tree_sitter_c as tsc and switch by CLI flag

TS_LANG = Language(tscpp.language())  # default: C++
PARSER = Parser(TS_LANG)

EDGE_TYPES = {"parent": 0, "next_sibling": 1, "next_token": 2}

# =========================
# 2) NORMALIZATION (must mirror training)
# =========================

def normalize_code(code: str) -> str:
    text = code.encode("utf8")
    try:
        tree = PARSER.parse(text)
        root = tree.root_node
    except Exception:
        # Very simple fallback if parser fails
        import re
        code = re.sub(r'\"([^"\\]|\\.)*\"', '<STR>', code)
        code = re.sub(r"\'([^'\\]|\\.)*\'", '<CHAR>', code)
        code = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', code)
        return code

    scope_stack = []  # [{"func":str|None, "params":{}, "vars":{}}]
    counters = {"func": 0}

    def push_scope():
        scope_stack.append({"func": None, "params": {}, "vars": {}})

    def pop_scope():
        scope_stack.pop()

    def get_id(node):
        return text[node.start_byte:node.end_byte].decode("utf8", "ignore")

    def assign_param(name, scope):
        if name in scope["params"]:
            return scope["params"][name]
        idx = len(scope["params"]) + 1
        scope["params"][name] = f"PARAM_{idx}"
        return scope["params"][name]

    def assign_var(name, scope):
        if name in scope["vars"]:
            return scope["vars"][name]
        idx = len(scope["vars"]) + 1
        scope["vars"][name] = f"VAR_{idx}"
        return scope["vars"][name]

    LITS = {"number_literal": "<NUM>", "string_literal": "<STR>", "char_literal": "<CHAR>"}

    def first_pass(node):
        t = node.type
        if t == "function_definition":
            push_scope()
            func_decl = None
            for ch in node.children:
                if ch.type == "function_declarator":
                    func_decl = ch
                    break
            if func_decl is not None:
                for ch in func_decl.children:
                    if ch.type == "identifier":
                        counters["func"] += 1
                        scope_stack[-1]["func"] = f"FUNC_{counters['func']}"
                        break
                for ch in func_decl.children:
                    if ch.type == "parameter_list":
                        for p in ch.children:
                            if p.type == "parameter_declaration":
                                for gch in p.children:
                                    if gch.type == "identifier":
                                        assign_param(get_id(gch), scope_stack[-1])
            for ch in node.children:
                first_pass(ch)
            pop_scope()
            return

        if scope_stack:
            if t in {"init_declarator", "declarator"}:
                for ch in node.children:
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
            if scope_stack and scope_stack[-1]["func"] is not None:
                parent = node.parent
                if parent and parent.type == "function_declarator":
                    return scope_stack[-1]["func"]
            return lookup_identifier(get_id(node))
        if len(node.children) == 0:
            return text[node.start_byte:node.end_byte].decode("utf8", "ignore")
        parts = []
        for ch in node.children:
            parts.append(second_pass(ch))
        return "".join(parts)

    try:
        first_pass(root)
        return second_pass(root)
    except Exception:
        import re
        code = text.decode("utf8", "ignore")
        code = re.sub(r'\"([^"\\]|\\.)*\"', '<STR>', code)
        code = re.sub(r"\'([^'\\]|\\.)*\'", '<CHAR>', code)
        code = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', code)
        return code

# =========================
# 3) AUGMENTED AST
# =========================

@dataclass
class ASTGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int, str]]
    raw: str


def build_augmented_ast(code: str):
    tree = PARSER.parse(code.encode("utf8"))
    nodes, edges = [], []
    nid = 0

    def walk(node, parent_id=None, last_sib=None, depth=0):
        nonlocal nid
        my = nid
        nid += 1
        snippet = code.encode("utf8")[node.start_byte:node.end_byte]
        children = node.children
        nodes.append(
            {
                "id": my,
                "type": node.type,
                "is_leaf": int(len(children) == 0),
                "depth": depth,
                "text": snippet.decode("utf8", "ignore"),
            }
        )
        if parent_id is not None:
            edges.append((parent_id, my, "parent"))
        if last_sib is not None:
            edges.append((last_sib, my, "next_sibling"))
        prev = None
        for ch in children:
            ch_id = walk(ch, my, prev, depth + 1)
            if prev is not None:
                edges.append((prev, ch_id, "next_token"))
            prev = ch_id
        return my

    walk(tree.root_node)
    return nodes, edges

# =========================
# 4) TO PyG (single sample)
# =========================

TOK_SENTINEL = None  # will be initialized from meta["tok_dim"] as tok_dim


def _hash_bucket(s: str, D: int) -> int:
    if not s or not s.strip():
        return D  # sentinel id == D (like training)
    h = hashlib.md5(s.strip().encode("utf8")).hexdigest()
    return int(h, 16) % D


def graph_to_pyg(nodes, edges, vocab_map: Dict[str, int], tok_dim: int) -> Data:
    # type one-hot via training-time vocab size
    vocab_size = int(max(vocab_map.values()) + 1) if vocab_map else 1

    type_ids = [[vocab_map.get(n["type"], 0)] for n in nodes]
    x_type = torch.tensor(np.array(type_ids), dtype=torch.long)

    # token bucket only for leaves
    tok_ids = [[_hash_bucket(n.get("text", "") if n.get("is_leaf", 0) else "", tok_dim)] for n in nodes]
    x_tok = torch.tensor(np.array(tok_ids), dtype=torch.long)

    # small numeric features: [is_leaf, depth_norm]
    max_depth = max([n.get("depth", 0) for n in nodes] + [1])
    small = [[float(n.get("is_leaf", 0)), float(n.get("depth", 0)) / float(max_depth)] for n in nodes]
    x_small = torch.tensor(np.array(small), dtype=torch.float)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
    else:
        src = [s for s, _, _ in edges]
        dst = [d for _, d, _ in edges]
        et = [EDGE_TYPES[t] for *_, t in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type = torch.tensor(et, dtype=torch.long)

    data = Data(edge_index=edge_index, y=torch.tensor([0], dtype=torch.long))
    data.edge_type = edge_type
    data.x_type = x_type
    data.x_tok = x_tok
    data.x_small = x_small
    data.x = x_type.clone()
    data.num_nodes = x_type.size(0)

    # Keep vocab_size in data so model can be (safely) created
    data.vocab_size = vocab_size
    return data

# =========================
# 5) MODEL (must match training)
# =========================

class GGNNBlockFeats(nn.Module):
    def __init__(self, channels: int, steps: int, num_edge_types: int = 3):
        super().__init__()
        from torch_geometric.nn import GatedGraphConv
        self.num_edge_types = max(1, num_edge_types)
        self.convs = nn.ModuleList(
            [GatedGraphConv(channels, num_layers=steps) for _ in range(self.num_edge_types)]
        )
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
        return F.leaky_relu(h, negative_slope=0.01)


class GGNNClassifierFeatsNoEmb(nn.Module):
    def __init__(
        self,
        num_types: int,
        tok_dim: int,
        small_dim: int = 2,
        steps: int = 10,
        blocks: int = 5,
        num_edge_types: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dim_type = num_types
        self.dim_tok = tok_dim + 1  # +1 for sentinel bucket (unchanged)
        self.dim_small = small_dim
        self.channels = self.dim_type + self.dim_tok + self.dim_small
        self.blocks = nn.ModuleList(
            [GGNNBlockFeats(self.channels, steps, num_edge_types) for _ in range(blocks)]
        )
        self.drop = nn.Dropout(dropout)
        from torch_geometric.nn import GlobalAttention

        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(self.channels, self.channels // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(self.channels // 2, 1),
            )
        )
        self.head = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(self.channels, 1),
        )

    def build_features(self, data: Data):
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

    def forward(self, data: Data):
        h = self.build_features(data)
        et = getattr(data, "edge_type", None)
        for blk in self.blocks:
            h = blk(h, data.edge_index, et)
        h = self.drop(h)
        batch = getattr(data, "batch", torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        # If single graph, construct a zero-batch vector
        from torch_geometric.nn import global_mean_pool
        # We use attention pooling as in training
        hg = self.pool(h, batch)
        return self.head(hg).view(-1)  # [B]

# =========================
# 6) INFERENCE PIPELINE
# =========================

def load_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Required keys: vocab_map (dict str->int), tok_dim (int), num_edge_types (int), model_hparams (optional)
    assert "vocab_map" in meta and "tok_dim" in meta, "meta JSON must contain 'vocab_map' and 'tok_dim'"
    return meta


def build_single_data(code: str, vocab_map: Dict[str, int], tok_dim: int) -> Data:
    norm = normalize_code(code)
    nodes, edges = build_augmented_ast(norm)
    data = graph_to_pyg(nodes, edges, vocab_map, tok_dim)
    # Add a batch vector for a single-graph mini-batch
    data.batch = torch.zeros(data.x_type.size(0), dtype=torch.long)
    return data


def sigmoid_probability(logit_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        prob = torch.sigmoid(logit_tensor).flatten()[0].item()
    return float(prob)


def predict_probability(code: str, model_path: str, meta_path: str) -> float:
    meta = load_meta(meta_path)
    vocab_map: Dict[str, int] = {str(k): int(v) for k, v in meta["vocab_map"].items()}
    tok_dim: int = int(meta["tok_dim"])

    data = build_single_data(code, vocab_map, tok_dim)

    # Model hyperparams
    num_types = int(max(vocab_map.values()) + 1) if vocab_map else 1
    steps = int(meta.get("steps", 10))
    blocks = int(meta.get("blocks", 5))
    num_edge_types = int(meta.get("num_edge_types", 3))
    dropout = float(meta.get("dropout", 0.3))

    model = GGNNClassifierFeatsNoEmb(
        num_types=num_types,
        tok_dim=tok_dim,
        small_dim=2,
        steps=steps,
        blocks=blocks,
        num_edge_types=num_edge_types,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    data = data.to(DEVICE)
    with torch.no_grad():
        logits = model(data)  # shape [1]
    return sigmoid_probability(logits)

# =========================
# 7) CLI
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pt checkpoint (state_dict)")
    ap.add_argument("--meta", required=True, help="Path to meta JSON with vocab_map and tok_dim")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="Path to a text file containing a single C/C++ function snippet")
    g.add_argument("--code", help="Inline code string")
    args = ap.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        code = args.code

    prob = predict_probability(code, args.model, args.meta)
    # Print only the probability so the script can be used programmatically
    print(f"{prob:.6f}")

if __name__ == "__main__":
    main()


