# Data
import pandas as pd

# Graph Creation
import ast
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from py2cfg import CFGBuilder

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

import json




class FunctionGraphBuilder():

    node_id = 0
    cfg = None

    def __init__(self):
        self.nodes = pd.DataFrame(columns=['node_id', 'name', 'code', 'parent_id'])
        self.edges = pd.DataFrame(columns=['source_id', 'target_id'])


    def create_graph(self, code, graph_type="AST", package='ts', edge_set='default', visualize=False):
        """
        Parse the given code and return the AST.
        Parameters:
            - code (str): The Python code to parse.
            - graph_type (str): The type of graph to create. Can be "AST" or "CFG".
            - package (str): The package to use for parsing. Can be 'ts' for tree-sitter or 'ast' for python ast.
            - edge_set (str): The type of edges to create. Can be 'default' or 'extended'. Only applicable for AST.
        """
        
        cleaned_code = self._remove_docstring(code)

        # -------------------------------------
        # Create AST
        if graph_type in ["AST", "ast"]:

            if package == 'ts':
                tree = parser.parse(bytes(cleaned_code, "utf8"))
                self._TS_create_ast(tree, source_code=bytes(cleaned_code, "utf8"))
            else:
                tree = ast.parse(cleaned_code)
                self._AST_create_ast(tree, parent_id=None)
            
            self.edges = self.nodes[['parent_id', 'node_id']].rename(columns={'parent_id': 'source_id', 'node_id': 'target_id'}).dropna()

            if edge_set == 'extended':

                # Connect leaf consecutive leaf nodes
                leaf_nodes = self.nodes[self.nodes['is_leaf']]
                leaf_nodes['target_id'] = leaf_nodes['node_id'].shift(-1)
                leaf_nodes = leaf_nodes.dropna(subset=['target_id'])
                leaf_edges = leaf_nodes[['node_id', 'target_id']].rename(columns={'node_id': 'source_id', 'target_id': 'target_id'})
                self.edges = pd.concat([self.edges, leaf_edges], ignore_index=True).reset_index(drop=True)
        
        # -------------------------------------
        # Create Control Flow Graph (CFG)
        elif graph_type in ["CFG", "cfg"]:

            try:
                self._CFG_create_cfg(cleaned_code)
                if visualize:
                    self.cfg.build_visual("cfg_output", format='png')
            except:
                print("Error creating CFG. Returning empty DataFrames.")
                return pd.DataFrame(columns=['node_id', 'code', 'parent_id']), pd.DataFrame(columns=['source_id', 'target_id'])

        return self.nodes, self.edges
    

    def _remove_docstring(self, code):
        """
        Remove docstrings from the code.
        """
        tree = ast.parse(code)

        # Remove docstrings from the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 1:
                    node.body = node.body[1:]
        
        return ast.unparse(tree)

    def _AST_create_ast(self, node, parent_id=None):

        node_id = self.node_id
        node_name = type(node).__name__
        if node_name == 'Load' or node_name == 'Store':
            return
        
        node_code = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)

        new_row = pd.DataFrame([{
            'node_id': node_id,
            'name': node_name,
            'code': node_code,
            'parent_id': parent_id,
            'is_leaf': len(list(ast.iter_child_nodes(node))) == 0
        }])
        
        self.nodes = pd.concat([self.nodes, new_row], ignore_index=True).reset_index(drop=True)
        self.node_id += 1

        for child in ast.iter_child_nodes(node):
            self._create_ast(child, parent_id=node_id)

    def _TS_create_ast(self, tree, source_code):

        nodes = []
        node_id_counter = [0]

        def traverse(node, parent_id):
            current_id = node_id_counter[0]
            node_id_counter[0] += 1

            node_type = node.type
            code_snippet = source_code[node.start_byte:node.end_byte].decode("utf8")

            nodes.append({
                "node_id": current_id,
                "name": node_type,
                "code": code_snippet,
                "parent_id": parent_id,
                "is_leaf": len(node.children) == 0
            })

            for child in node.children:
                traverse(child, current_id)

        root_node = tree.root_node
        traverse(root_node, parent_id=None)

        self.nodes = pd.DataFrame(nodes)

    def _CFG_create_cfg(self, code):
        """
        Create a control flow graph (CFG) from the given code.
        """

        self.cfg = CFGBuilder().build_from_src("example", code)
        self.cfg.build_visual("cfg_output", format='json')
        
        with open("cfg_output.json") as f:
            data = json.load(f)

        # Csak azok a node-ok, amik "_ldraw_" alatt szöveget (text) tartalmaznak
        nodes = []
        for obj in data["objects"]:
            if "_ldraw_" in obj:
                text_parts = [entry["text"] for entry in obj["_ldraw_"] if entry["op"] == "T"]
                if text_parts:
                    nodes.append({
                        "node_id": int(obj["_gvid"]),
                        "code": "\\n".join(text_parts)
                    })

        self.nodes = pd.DataFrame(nodes)

        edges = []
        for edge in data["edges"]:
            source = edge["tail"]
            target = edge["head"]
            label = ""
            for item in edge.get("_ldraw_", []):
                if item["op"] == "T":
                    label = item["text"]
            edges.append({
                "source_id": source,
                "target_id": target,
                "label": label
            })

        self.edges = pd.DataFrame(edges)

        # Drop the isolated nodes
        self.nodes = self.nodes[self.nodes['node_id'].isin(self.edges['source_id']) | self.nodes['node_id'].isin(self.edges['target_id'])].reset_index(drop=True)
        self.nodes = self.nodes.iloc[:-8] # Remove the last 8 nodes, which are added by the package for plotting a legend to the visual output