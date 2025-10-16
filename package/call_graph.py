# Data
import pandas as pd

# Graph Creation
import ast
import networkx as nx
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from pyvis.network import Network

from package.adapters import LanguageAstAdapter

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# Torch
import torch
from torch_geometric.data import Data

# NLP
import torch


# Other
import copy, uuid, json
import os

from package.constants import REVERSE_EXTENSION_MAP
from package.adapters import LanguageAstAdapterRegistry

class CallGraphBuilder:

    imports = None
    classes = None
    functions = None
    calls = None

    nodes = None
    edges = None

    imp_id = 0
    cls_id = 0
    fnc_id = 0
    cll_id = 0


    def __init__(self):
        self.imports = pd.DataFrame(columns=['file_id', 'imp_id', 'name', 'from', 'as_name'])
        self.classes = pd.DataFrame(columns=['file_id', 'cls_id', 'name', 'base_classes'])
        self.functions = pd.DataFrame(columns=['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params'])
        self.calls = pd.DataFrame(columns=['file_id', 'cll_id', 'name', 'class', 'class_base_classes'])


    # Return type can be :
    #   - "pandas": for pandas DataFrames 
    #   - "original" for original pandas dataframes (imports, classes, functions, calls)
    #   - "networkx" for a NetworkX graph
    #   - "pytorch" for a PyTorch Geometric graph
    def build_call_graph(self, path, return_type="pandas", repo_functions_only=True):
        """
        Build a call graph from the given file path.
        Parameters:
            - path (str): The path to the directory containing Python files.
            - return_type (str): The type of the return value. Can be "pandas", "original", "networkx", or "pytorch".
            - repo_functions_only (bool): If True, only consider function calls within the repository.
        """

        # Reset IDs
        self.imp_id = 0
        self.cls_id = 0
        self.fnc_id = 0
        self.cll_id = 0

        filename_lookup = {}

        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                """
                name, file_extension = os.path.splitext(filename)
                if file_extension is None:
                    continue

                language: str = REVERSE_EXTENSION_MAP[file_extension]
                language_adapter:  LanguageAstAdapter = LanguageAstAdapterRegistry.get_adapter(language)

                """

                file_id = str(uuid.uuid1())
                file_name_and_path = os.path.join(dirpath, filename)

                filename_lookup[file_id] = file_name_and_path

                file_path = os.path.join(dirpath, filename)
                python_code = ''
                with open(file_path, 'r', encoding='utf-8') as f:
                    python_code += f.read()

                tree = ast.parse(python_code)
                self.process_file_ast(tree, return_dataframes=False, file_id=file_id)

        split_columns = self.calls['name'].str.split('.', n=1, expand=True)

        # Add combined name column to functions dataframe
        self.functions['combinedName'] = self.functions.apply(
            lambda x: (
                x["name"] if x["class"] == 'Global' else
                str(x["class"]) + '.' + str(x['name'])
            ), axis=1
        )

        self.functions['function_location'] = self.functions.apply(
            lambda x: (
                filename_lookup.get(x['file_id'], None) if pd.notnull(x['file_id']) else None
            ), axis=1
        )

        # Columns to store the split results
        self.calls['call_object'] = split_columns[0]
        self.calls['call_functiondot'] = split_columns[1]  # Automatically None if no dot is present

        # Resolve caller object
        self._resolve_caller_object()

        # Calls resolved combined name
        self.calls['combinedName'] = self.calls.apply(
            lambda x: (
                str(x["resolved_call_object"]) if x["call_functiondot"] is None else
                str(x["resolved_call_object"]) + '.' + str(x['call_functiondot'])
            ), axis=1
        )

        if return_type == "original":
            return self.imports, self.classes, self.functions, self.calls

        # Create nodes and edges for the call graph
        self.nodes = copy.deepcopy(self.functions)
        self.edges = copy.deepcopy(self.calls.loc[self.calls['func_id'].notnull()])

        # If we only want to consider function calls within the repository
        if repo_functions_only:
            self.edges = self.edges.merge(self.nodes[['fnc_id', 'combinedName']], left_on='combinedName', right_on='combinedName', how='inner')[['func_id', 'fnc_id']] \
                .rename(columns={'func_id': 'source_id', 'fnc_id': 'target_id'})
        
        # If we want to consider all function calls, including those not defined in the repository (e.g., external libraries)
        else:
            # Merge edges with nodes to find undefined functions
            self.edges = self.edges.merge(self.nodes[['fnc_id', 'combinedName']], left_on='combinedName', right_on='combinedName', how='left')

            # Identify new nodes that are not in the existing nodes and create dataframe for them
            new_nodes = self.edges.loc[self.edges['fnc_id'].isnull()].drop_duplicates(subset=['combinedName'])
            new_nodes['new_fnc_id'] = range(self.fnc_id, self.fnc_id + len(new_nodes))
            new_nodes = new_nodes[['new_fnc_id', 'combinedName']].rename(columns={'new_fnc_id': 'fnc_id'})
            new_nodes['file_id'] = None
            new_nodes['name'] = new_nodes['combinedName']
            new_nodes['docstring'] = None
            new_nodes['class_id'] = None
            new_nodes['class'] = None
            new_nodes['class_base_classes'] = '[]'
            new_nodes['params'] = '{}'
            new_nodes = new_nodes[['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params', 'docstring', 'class_id', 'combinedName']]
            
            # Update the function ID counter
            self.fnc_id += len(new_nodes)

            # Concatenate the new nodes with the existing nodes
            self.nodes = pd.concat([self.nodes, new_nodes], ignore_index=True).reset_index(drop=True)

            # Update the edges with the new function IDs
            self.edges = self.edges.merge(new_nodes[['fnc_id', 'combinedName']].rename(columns={'fnc_id': 'new_fnc_id'}), on='combinedName', how='left')
            self.edges['fnc_id'] = self.edges['fnc_id'].fillna(self.edges['new_fnc_id'])
            self.edges = self.edges.drop(columns=['new_fnc_id'])
            self.edges = self.edges.rename(columns={'func_id': 'source_id', 'fnc_id': 'target_id'})[['source_id', 'target_id']]
            self.edges['target_id'] = self.edges['target_id'].astype(int)

        if return_type == "pandas":
            return self.nodes, self.edges, self.imports
        
        elif return_type == "networkx":
            G = nx.from_pandas_edgelist(self.edges, source='source_id', target='target_id', create_using=nx.DiGraph())
            for _, row in self.nodes.iterrows():
                G.nodes[row['fnc_id']].update({
                    'file_id': row['file_id'],
                    'name': row['name'],
                    'class': row['class'],
                    'class_base_classes': row['class_base_classes'],
                    'params': row['params'],
                    'docstring': row['docstring']
                })
            return G
        
        else:
            x = torch.tensor(self.nodes['fnc_id'].values, dtype=torch.long)
            edge_index = torch.tensor(self.edges.values.T, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            return data
        


    def process_file_ast(self, ast_tree, return_dataframes=True, file_id=None):
        """
        Create a call graph from the given AST tree.
        """

        self._walk_ast(ast_tree, file_id=file_id)
        
        # Reset IDs for next use
        # self.imp_id = 0
        # self.cls_id = 0
        # self.fnc_id = 0
        # self.cll_id = 0

        if return_dataframes:
            return self.imports, self.classes, self.functions, self.calls



    def visualize_graph(self, output_path='graph.html'):
        # Create graph
        G = nx.Graph()

        # Add nodes
        for _, row in self.nodes.iterrows():
            node_id = str(row['fnc_id'])
            node_label = str(row['combinedName'])
            #node_docstring = str(row['docstring_embedding'])
            #node_label += f"\n {node_docstring}" if node_docstring else ''
            G.add_node(node_id, label=node_label)

        # Add edges
        for _, row in self.edges.iterrows():
            source = str(row['source_id'])
            target = str(row['target_id'])
            if source in G.nodes and target in G.nodes:
                G.add_edge(source, target)

        # Pyvis graph
        net = Network(height='1700px', width='100%', notebook=False)
        net.from_nx(G)
        net.force_atlas_2based()

        # Save
        net.save_graph(output_path)



    def _walk_ast(self, node, file_id=None, class_id=None, class_name=None, base_classes=None, func_id=None, func_name=None, func_params=None):

        # Handle Imports
        self._handle_imports(node, file_id=file_id)

        # Class definitions
        node_class_id, node_class_name, node_base_classes = self._handle_class_definitions(node=node, file_id=file_id)
        if node_class_id is not None:
            class_id = node_class_id
            class_name = node_class_name
            base_classes = node_base_classes
        
        # Function definitions
        node_func_id, node_func_name, node_func_params = self._handle_functions(
            node=node, 
            file_id=file_id, 
            class_id=class_id, 
            class_name=class_name, 
            base_classes=base_classes
        )
        if node_func_id is not None:
            func_id = node_func_id
            func_name = node_func_name
            func_params = node_func_params

        # Call expressions
        self._handle_call_expressions(
            node=node, 
            file_id=file_id,
            class_id=class_id, 
            class_name=class_name, 
            base_classes=base_classes, 
            func_id=func_id, 
            func_name=func_name, 
            func_params=func_params
        )

        for child in ast.iter_child_nodes(node):
            self._walk_ast(
                node=child, 
                file_id=file_id,
                class_id=class_id, 
                class_name=class_name, 
                base_classes=base_classes, 
                func_id=func_id, 
                func_name=func_name, 
                func_params=func_params
            )



    def _handle_imports(self, node, file_id=None):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    if file_id:
                        new_row = pd.DataFrame([{'file_id': file_id, 'imp_id': self.imp_id, 'name': alias.name, 'from': None, 'as_name': alias.asname}])
                    else:
                        new_row = pd.DataFrame([{'imp_id': self.imp_id, 'name': alias.name, 'from': None, 'as_name': alias.asname}])
                    self.imports = pd.concat([self.imports, new_row], ignore_index=True).reset_index(drop=True)
                    self.imp_id += 1
                else:
                    if file_id:
                        new_row = pd.DataFrame([{'file_id': file_id, 'imp_id': self.imp_id, 'name': alias.name, 'from': None, 'as_name': alias.name}])
                    else:
                        new_row = pd.DataFrame([{'imp_id': self.imp_id, 'name': alias.name, 'from': None, 'as_name': alias.name}])
                    self.imports = pd.concat([self.imports, new_row], ignore_index=True).reset_index(drop=True)
                    self.imp_id += 1
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname:
                    if file_id:
                        new_row = pd.DataFrame([{'file_id': file_id, 'imp_id': self.imp_id, 'name': alias.name, 'from': node.module, 'as_name': alias.asname}])
                    else:
                        new_row = pd.DataFrame([{'imp_id': self.imp_id, 'name': alias.name, 'from': node.module, 'as_name': alias.asname}])
                    self.imports = pd.concat([self.imports, new_row], ignore_index=True).reset_index(drop=True)
                    self.imp_id += 1
                else:
                    if file_id:
                        new_row = pd.DataFrame([{'file_id': file_id, 'imp_id': self.imp_id, 'name': alias.name, 'from': node.module, 'as_name': alias.name}])
                    else:
                        new_row = pd.DataFrame([{'imp_id': self.imp_id, 'name': alias.name, 'from': node.module, 'as_name': alias.name}])
                    self.imports = pd.concat([self.imports, new_row], ignore_index=True).reset_index(drop=True)
                    self.imp_id += 1



    def _handle_class_definitions(self, node, file_id=None):
        if isinstance(node, ast.ClassDef):
            node_class_id = self.cls_id
            node_class_name = node.name
            node_base_classes = [ast.unparse(base) for base in node.bases] if node.bases else []
            if file_id:
                new_row = pd.DataFrame([{'file_id': file_id, 'cls_id': node_class_id, 'name': node_class_name, 'docstring': ast.get_docstring(node),  'base_classes': node_base_classes}])
            else:
                new_row = pd.DataFrame([{'cls_id': node_class_id, 'name': node_class_name, 'docstring': ast.get_docstring(node),  'base_classes': node_base_classes}])
            self.classes = pd.concat([self.classes, new_row], ignore_index=True).reset_index(drop=True)
            self.cls_id += 1
            return node_class_id, node_class_name, node_base_classes
        else:
            return None, None, None
        


    def _handle_functions(self, node, file_id=None, class_id=None, class_name=None, base_classes=None):
        if isinstance(node, ast.FunctionDef):
            func_id = self.fnc_id
            func_name = node.name
            class_name = class_name if class_name else 'Global'
            base_classes = base_classes if base_classes else []

            param_types = {}
            for arg in node.args.args:
                if arg.annotation:
                    param_types[arg.arg] = ast.unparse(arg.annotation)
                else:
                    param_types[arg.arg] = 'Any'

            if file_id:
                new_row = pd.DataFrame([{
                    'file_id': file_id,
                    'fnc_id': func_id,
                    'name': func_name,
                    'docstring': ast.get_docstring(node),
                    'function_code': ast.unparse(node),
                    'class_id': class_id if class_id is not None else None,
                    'class': class_name,
                    'class_base_classes': base_classes,
                    'params': json.dumps(param_types)
                }])
            else:
                new_row = pd.DataFrame([{
                    'fnc_id': func_id,
                    'name': func_name,
                    'docstring': ast.get_docstring(node),
                    'function_code': ast.unparse(node),
                    'class_id': class_id if class_id is not None else None,
                    'class': class_name,
                    'class_base_classes': base_classes,
                    'params': json.dumps(param_types)
                }])
            self.functions = pd.concat([self.functions, new_row], ignore_index=True).reset_index(drop=True)
            self.fnc_id += 1

            return func_id, func_name, param_types
        else:
            return None, None, None
            
        

    def _handle_call_expressions(self, node, file_id=None, class_id=None, class_name=None, base_classes=None, func_id=None, func_name=None, func_params=None):
        if isinstance(node, ast.Call):
            call = ast.unparse(node.func)
            class_name = class_name if class_name else 'Global'
            base_classes = base_classes if base_classes else []
            if file_id:
                new_row = pd.DataFrame([{
                    'file_id': file_id,
                    'cll_id': self.cll_id,
                    'name': call,
                    'class_id': class_id if class_id is not None else None,
                    'class': class_name,
                    'class_base_classes': base_classes,
                    'func_id': func_id if func_id is not None else None,
                    'func_name': func_name if func_name is not None else None,
                    'func_params': func_params
                }])
            else:
                new_row = pd.DataFrame([{
                    'cll_id': self.cll_id,
                    'name': call,
                    'class_id': class_id if class_id is not None else None,
                    'class': class_name,
                    'class_base_classes': base_classes,
                    'func_id': func_id if func_id is not None else None,
                    'func_name': func_name if func_name is not None else None,
                    'func_params': func_params
                }])
            self.calls = pd.concat([self.calls, new_row], ignore_index=True).reset_index(drop=True)
            self.cll_id += 1



    def _resolve_caller_object(self):

        # Create a lookup table for the imports
        import_alias_map = {
            (row['file_id'], row['as_name']): row['name']
            for _, row in self.imports.dropna(subset=['as_name']).iterrows()
        }

        # Handle self and super calls
        self.calls['resolved_call_object'] = self.calls.apply(
            lambda x: (
                # Handle 'self' calls
                x["class"] if x["call_object"] == 'self' and pd.notnull(x["call_functiondot"]) else
                # Handle 'super' calls
                x["class_base_classes"][0] if isinstance(x["call_object"], str) and 'super' in x["call_object"] and pd.notnull(x["call_functiondot"]) and len(x["class_base_classes"]) > 0 else
                # Handle function parameters
                x["func_params"].get(x["call_object"], x["call_object"]) if isinstance(x["func_params"], dict) and x["call_object"] in x["func_params"] else
                # Handle imports - if no import alias is found, use the original call object
                import_alias_map.get((x["file_id"], x["call_object"]), x["call_object"])
            ), axis=1
        )