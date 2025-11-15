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

from package.constants import REVERSE_EXTENSION_MAP, extensions, special_filenames
from package.adapters import LanguageAstAdapterRegistry
from package.ast_processor import AstProcessor

from pathlib import Path

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
    fl_id = 0

    def __init__(self):
        self.imports = pd.DataFrame(columns=['file_id', 'imp_id', 'name', 'from', 'as_name'])
        self.classes = pd.DataFrame(columns=['file_id', 'cls_id', 'name', 'base_classes'])
        self.functions = pd.DataFrame(columns=['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params'])
        self.calls = pd.DataFrame(columns=['file_id', 'cll_id', 'name', 'class', 'class_base_classes'])
        self.files = pd.DataFrame(columns=['fl_id','file_id', 'name', 'path', 'is_folder', 'directory_id'])
        self.__README_NAMES = ["readme", "read me", "read"]

    def __concat_df(self, df1, df2):
        # Handle if df2 is a list or a single dataframe
        if isinstance(df2, list):
            df_combined = [df1] + df2
        else:
            df_combined = [df1, df2]
        return pd.concat(df_combined, ignore_index=True)

    def add_directory_node(self, path: str, dir_mapper: dict[str, str], fl_id: int) -> str:
        dirpath, basename = os.path.split(path)

        dirpath = dirpath.rstrip('/')
        path_normalized = path.rstrip('/')

        if len(dir_mapper) == 0:
            directory_id = None
        else:
            directory_id = dir_mapper.get(dirpath)

        own_id = str(uuid.uuid1())
        dir_mapper[path_normalized] = own_id  # Store normalized path
        df = pd.DataFrame([{'fl_id': fl_id, 'file_id': own_id, 'name': basename, 'path': path, 'is_folder': True,
                            'directory_id': directory_id}])

        return own_id, df

    def is_config_file(self, filename: str) -> bool:
        """
        Check if a file is a project-defining file for C++/Erlang/Python projects.

        """
        path = Path(filename)

        # Extract just the filename (without parent directories)
        file_name = path.name

        # Check for exact filename match
        if file_name in special_filenames:
            return True

        if file_name.startswith('.env'):
            return True

        # Check extension using pathlib
        if path.suffix in extensions:
            return True

        # Check for multiple suffixes (e.g., .app.src has suffix .src, but we want .app.src)
        for ext in extensions:
            if file_name.endswith(ext):
                return True

        return False

    # Return type can be :
    #   - "pandas": for pandas DataFrames 
    #   - "original" for original pandas dataframes (imports, classes, functions, calls)
    #   - "networkx" for a NetworkX graph
    #   - "pytorch" for a PyTorch Geometric graph
    def build_call_graph(self, path, return_type="pandas", repo_functions_only=True, project_language=None):
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
        self.fl_id = 0
        self.config_id = 1 # separate config file id-s from project files and directories, 
        readme_found = False
        
        filename_lookup = {}
        directory_mapper: dict[str, str] = {}

        for dirpath, _, filenames in os.walk(path):
            if ".git" in dirpath:
                continue #skip git related folder todo: add exclude list
            dir_id, dir_df = self.add_directory_node(dirpath, directory_mapper, self.fl_id)
            self.files = self.__concat_df(self.files, dir_df)
            self.fl_id += 1

            for filename in filenames:

                name, file_extension = os.path.splitext(filename)
                if file_extension is None:
                    continue

                language: str = REVERSE_EXTENSION_MAP.get(file_extension, None)
                is_config_file = self.is_config_file(filename)
                if language is None and not is_config_file:
                    continue

                if project_language is None:
                    project_language = language

                if project_language is not None and language != project_language and not is_config_file:
                    continue  # in case project language is set skip other files

                file_id = str(uuid.uuid1())
                file_name_and_path = os.path.join(dirpath, filename)
                filename_lookup[file_id] = file_name_and_path

                file_path = os.path.join(dirpath, filename)
                file_content = ''
                with open(file_path, 'rb') as f:
                    file_content = f.read()

                config_content = None
                current_file_id = self.fl_id
                temp_readme_found = readme_found
                if is_config_file:
                    config_content = file_content
                    if not readme_found:
                        if name.lower() in self.__README_NAMES:
                            readme_found = True
                            current_file_id = 0
                    if temp_readme_found == readme_found:
                        current_file_id = self.config_id
                        self.config_id += 1
                file_df = pd.DataFrame([{'fl_id': current_file_id, 'file_id': file_id, 'name': filename, 'path': path,
                                         'is_folder': False, 'directory_id': dir_id, "config": config_content}])
                self.fl_id += 1
                self.files = self.__concat_df(self.files, file_df)
                
                if is_config_file:
                    continue

                if language != "python":

                    language_adapter: LanguageAstAdapter = LanguageAstAdapterRegistry.get_adapter(language)
                    ast_processor = AstProcessor(language_adapter(), file_content)
                    imports, classes, functions, calls, id_dict = ast_processor.process_file_ast(file_id=file_id,
                                                                                                 id_dict={
                                                                                                     "imp_id": self.imp_id,
                                                                                                     "cls_id": self.cls_id,
                                                                                                     "fnc_id": self.fnc_id,
                                                                                                     "cll_id": self.cll_id
                                                                                                 })

                    self.imports = self.__concat_df(self.imports, imports)
                    self.classes = self.__concat_df(self.classes, classes)
                    self.functions = self.__concat_df(self.functions, functions)
                    self.calls = self.__concat_df(self.calls, calls)

                    self.imp_id = id_dict.get("imp_id")
                    self.cls_id = id_dict.get("cls_id")
                    self.fnc_id = id_dict.get("fnc_id")
                    self.cll_id = id_dict.get("cll_id")

                elif language == "python":
                    tree = ast.parse(file_content)
                    self.process_file_ast(tree, return_dataframes=False, file_id=file_id)
                else:
                    pass

        if project_language == "python":
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
        else:
            # Get C++ or Erlang adapter
            language_adapter = LanguageAstAdapterRegistry.get_adapter(project_language)
            language_adapter = language_adapter()

            # 1. Add combinedName to functions
            language_adapter.create_combined_name(self.functions, filename_lookup)

            # 2. Add function_location
            self.functions['function_location'] = self.functions.apply(
                lambda x: (
                    filename_lookup.get(x['file_id'], None) if pd.notnull(x['file_id']) else None
                ), axis=1
            )

            # 3. Resolve calls (adds combinedName to calls)
            language_adapter.resolve_calls(self.calls, self.functions, self.classes, self.imports, filename_lookup)

        if return_type == "original":
            return self.imports, self.classes, self.functions, self.calls, self.files, project_language

        # Create nodes and edges for the call graph
        self.nodes = copy.deepcopy(self.functions)
        self.edges = copy.deepcopy(self.calls.loc[self.calls['func_id'].notnull()])

        # If we only want to consider function calls within the repository
        if repo_functions_only:
            self.edges = self.edges.merge(self.nodes[['fnc_id', 'combinedName']], left_on='combinedName', right_on='combinedName',
                             how='inner')[['func_id', 'fnc_id']] \
                .rename(columns={'func_id': 'source_id', 'fnc_id': 'target_id'})
        # If we want to consider all function calls, including those not defined in the repository (e.g., external libraries)
        else:
            # Merge edges with nodes to find undefined functions
            self.edges = self.edges.merge(self.nodes[['fnc_id', 'combinedName']], left_on='combinedName',
                                          right_on='combinedName', how='left')

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
            new_nodes = new_nodes[
                ['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params', 'docstring', 'class_id',
                 'combinedName']]

            # Update the function ID counter
            self.fnc_id += len(new_nodes)

            # Concatenate the new nodes with the existing nodes
            self.nodes = pd.concat([self.nodes, new_nodes], ignore_index=True).reset_index(drop=True)

            # Update the edges with the new function IDs
            self.edges = self.edges.merge(
                new_nodes[['fnc_id', 'combinedName']].rename(columns={'fnc_id': 'new_fnc_id'}), on='combinedName',
                how='left')
            self.edges['fnc_id'] = self.edges['fnc_id'].fillna(self.edges['new_fnc_id'])
            self.edges = self.edges.drop(columns=['new_fnc_id'])
            self.edges = self.edges.rename(columns={'func_id': 'source_id', 'fnc_id': 'target_id'})[
                ['source_id', 'target_id']]
            self.edges['target_id'] = self.edges['target_id'].astype(int)

        if return_type == "pandas":
            return self.nodes, self.edges, self.imports, self.classes, self.files, project_language

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

            local_vars = self._extract_local_variables_python(node)

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
                    'params': json.dumps(param_types),
                    'local_vars': local_vars
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
                    'params': json.dumps(param_types),
                    'local_vars': local_vars
                }])
            self.functions = pd.concat([self.functions, new_row], ignore_index=True).reset_index(drop=True)
            self.fnc_id += 1

            return func_id, func_name, param_types
        else:
            return None, None, None
            
    def _extract_local_variables_python(self, func_node):
        """
        Extract local variables assigned to class constructors.
        
        Handles patterns like:
            - x = ClassName()
            - y = module.ClassName()
        
        Args:
            func_node: ast.FunctionDef node
            
        Returns:
            JSON string of dict mapping variable names to class names
        """
        local_vars = {}
        
        # Walk through all nodes in the function
        for node in ast.walk(func_node):
            # Look for assignment statements
            if isinstance(node, ast.Assign):
                # Only handle simple single assignments (x = ...)
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    
                    # Check if the assigned value is a function/constructor call
                    if isinstance(node.value, ast.Call):
                        class_name = None
                        
                        # Pattern 1: ClassName() - direct constructor call
                        if isinstance(node.value.func, ast.Name):
                            class_name = node.value.func.id
                        
                        # Pattern 2: module.ClassName() - qualified constructor call
                        elif isinstance(node.value.func, ast.Attribute):
                            class_name = node.value.func.attr
                        
                        # Store the mapping if we found a class name
                        if class_name:
                            local_vars[var_name] = class_name
        
        return json.dumps(local_vars)

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

        func_local_vars_lookup = {}
        if 'local_vars' in self.functions.columns:
            for _, row in self.functions.iterrows():
                if pd.notnull(row.get('local_vars')):
                    try:
                        func_local_vars_lookup[row['fnc_id']] = json.loads(row['local_vars'])
                    except (json.JSONDecodeError, TypeError):
                        func_local_vars_lookup[row['fnc_id']] = {}
                else:
                    func_local_vars_lookup[row['fnc_id']] = {}

        # Handle self and super calls
        self.calls['resolved_call_object'] = self.calls.apply(
            lambda x: (
                # Handle 'self' calls
                x["class"] if x["call_object"] == 'self' and pd.notnull(x["call_functiondot"]) else
                # Handle 'super' calls
                x["class_base_classes"][0] if isinstance(x["call_object"], str) and 'super' in x["call_object"] and pd.notnull(x["call_functiondot"]) and len(x["class_base_classes"]) > 0 else
                #Handle local variables
                func_local_vars_lookup.get(x["func_id"], {}).get(x["call_object"]) if x["func_id"] in func_local_vars_lookup and x["call_object"] in func_local_vars_lookup.get(x["func_id"], {}) else (
                    # Handle function parameters
                    x["func_params"].get(x["call_object"], x["call_object"]) if isinstance(x["func_params"], dict) and x["call_object"] in x["func_params"] else
                    # Handle imports - if no import alias is found, use the original call object
                    import_alias_map.get((x["file_id"], x["call_object"]), x["call_object"])
                )
            ), axis=1
        )