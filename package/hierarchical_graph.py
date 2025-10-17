# Data
import os
import pandas as pd
import numpy as np

# Graph Creation
import ast
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from py2cfg import CFGBuilder

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

# Torch
import torch
from torch_geometric.data import HeteroData

# NLP
from transformers import RobertaTokenizer, RobertaModel
import torch

# Text embedding
from sentence_transformers import SentenceTransformer

# Other
from tqdm import tqdm

from .call_graph import CallGraphBuilder
from .function_graph import FunctionGraphBuilder

# ignore warnings


class HierarchicalGraphBuilder:

    nodes = None
    edges = None

    subgraph_nodes = None
    subgraph_edges = None

    hierarchical_sub_to_main_edges = None
    hierarchical_main_to_sub_edges = None


    def __init__(self):
        pass


    def create_hierarchical_graph(
        self, 
        path, 
        return_type="pandas", 
        repo_functions_only=True, 
        graph_type="CFG", 
        package='ts', 
        edge_set='default', 
        remove_isolated=False,
        remove_subgraph_missing=True,
        batch_size=64,
        create_embedding=False,
        project_language=None
    ):
        """
        Create a hierarchical graph from the given code.
        Parameters:
            - path: Path to the code repository or file.
            - return_type: Type of the return value, either 'pandas' or 'pyg'.
            - repo_functions_only: Whether to include only functions from the repository.
            - graph_type: Type of the graph to create, either 'AST' or 'CFG'.
            - package: Package to use for the graph creation, default is 'ts' (tree-sitter). Only relevant for 'AST' graph_type.
            - edge_set: Edge set to use for the graph creation. Can be 'default' or 'extended'. Only relevant for 'AST' graph_type. Default is 'default'.
            - remove_isolated: Whether to remove isolated nodes from the call graph.
            - remove_subgraph_missing: Whether to remove call graph nodes that do not have a corresponding subgraph node.
            - batch_size: Batch size for embedding the nodes. Default is 128.
            - create_embedding: Whether to create embeddings for the nodes. Default is False.
        Returns:
            If pandas return type is specified:
            - nodes: DataFrame containing the call graph nodes.
            - edges: DataFrame containing the call graph edges.
            - subgraph_nodes: DataFrame containing the subgraph nodes for each function.
            - subgraph_edges: DataFrame containing the subgraph edges for each function.
            - hierarchical_sub_to_main_edges: DataFrame containing the hierarchical edges from subgraph nodes to main graph nodes.
            - hierarchical_main_to_sub_edges: DataFrame containing the hierarchical edges from main graph nodes to subgraph nodes.
            If pyg return type is specified:
            - A PyTorch Geometric HeteroData object containing the hierarchical graph.
        """
        print("Building CG...")
        # return the project language
        self.nodes, self.edges, self.imports, language = CallGraphBuilder().build_call_graph(path, return_type="pandas", repo_functions_only=repo_functions_only, project_language=project_language)

        # Convert function IDs to integers
        self.nodes['fnc_id'] = self.nodes['fnc_id'].astype(int)
        self.edges['source_id'] = self.edges['source_id'].astype(int)
        self.edges['target_id'] = self.edges['target_id'].astype(int)

        if create_embedding:
            print("CG build completed. Embedding CG nodes...")
            self._embed_graph_nodes(batch_size=batch_size)
            self._fix_missing_docstring_embeddings()
            print("CG nodes embedded.")

        print("Creating subgraphs for each function...")
        self.subgraph_nodes = pd.DataFrame(columns=['func_id', 'node_id', 'name', 'code', 'parent_id', 'is_leaf']) if \
            graph_type == "AST" else pd.DataFrame(columns=['func_id', 'node_id', 'code'])
        self.subgraph_edges = pd.DataFrame(columns=['func_id', 'source_id', 'target_id'])


        for _, row in tqdm(self.nodes.iterrows()):
            code = row['function_code']
            if pd.isna(code):
                continue
            
            subg_nodes, subg_edges = FunctionGraphBuilder().create_graph(
                code=code, 
                graph_type=graph_type, 
                package=package, 
                edge_set=edge_set,
                visualize=False,
                language=language
            )

            subg_nodes['func_id'] = row['fnc_id']
            subg_edges['func_id'] = row['fnc_id']

            subg_nodes = subg_nodes[['func_id', 'node_id', 'name', 'code', 'parent_id', 'is_leaf']] if graph_type == "AST" else subg_nodes[['func_id', 'node_id', 'code']]
            subg_edges = subg_edges[['func_id', 'source_id', 'target_id']]

            self.subgraph_nodes = pd.concat([self.subgraph_nodes, subg_nodes], ignore_index=True).reset_index(drop=True)
            self.subgraph_edges = pd.concat([self.subgraph_edges, subg_edges], ignore_index=True).reset_index(drop=True)

        # Convert node IDs to integers
        self.subgraph_nodes['node_id'] = self.subgraph_nodes['node_id'].astype(int)
        self.subgraph_edges['source_id'] = self.subgraph_edges['source_id'].astype(int)
        self.subgraph_edges['target_id'] = self.subgraph_edges['target_id'].astype(int)
        print("Subgraphs created.")

        if create_embedding:
            print("Embedding subgraph nodes...")
            self._embed_subgraph_nodes(batch_size=batch_size)
            print("Subgraph nodes embedded.")

        print("Filtering graph nodes...")
        self._filter_call_graph(remove_isolated, remove_subgraph_missing)
        self._filter_sub_graph(remove_isolated)

        print("Graph nodes filtered. Creating hierarchical edges...")
        self.hierarchical_sub_to_main_edges = self.subgraph_nodes[['node_id', 'func_id']].rename(columns={'node_id': 'source_id', 'func_id': 'target_id'})
        self.hierarchical_main_to_sub_edges = self.subgraph_nodes[['func_id', 'node_id']].rename(columns={'func_id': 'source_id', 'node_id': 'target_id'})

        try:
            os.remove('noname.gv.xdot')
            os.remove('cfg_output.json')
        except:
            pass
        
        print("Hierarchical graph building successful.")



        if return_type.lower() in ["pandas", "pandas_df", 'pd', 'df', 'dataframe']:
            return self.nodes, self.edges, self.subgraph_nodes, self.subgraph_edges, self.hierarchical_sub_to_main_edges, self.hierarchical_main_to_sub_edges, self.imports
        
        else:
            return self._create_hetero_data()
        




    def _embed_graph_nodes(self, batch_size=128):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.nodes['docstring'] = self.nodes['docstring'].fillna('')

        sentences = self.nodes['docstring'].tolist()
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=True
        )

        self.nodes['docstring_embedding'] = embeddings.tolist()


    def _embed_subgraph_nodes(self, batch_size=128):
        """ 
        Embed the subgraph nodes using a text embedding model.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        model.eval()

        code_list = self.subgraph_nodes['code'].tolist()

        embeddings = []

        for i in tqdm(range(0, len(code_list), batch_size)):
            batch = code_list[i:i+batch_size] if i + batch_size <= len(code_list) else code_list[i:]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            cls_embeds = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeds.cpu().numpy())

        self.subgraph_nodes["embedding"] = list(np.vstack(embeddings))


    def _fix_missing_docstring_embeddings(self, docstring_col='docstring', embedding_col='docstring_embedding', dim=384):
        zero_vec = [0.0] * dim
        zero_str = str(zero_vec)

        self.nodes.loc[self.nodes[docstring_col].isna(), embedding_col] = zero_str
        self.nodes.loc[self.nodes[docstring_col].isnull(), embedding_col] = zero_str
        self.nodes.loc[self.nodes[docstring_col] == "", embedding_col] = zero_str
        self.nodes.loc[self.nodes[docstring_col] == "NaN", embedding_col] = zero_str
        self.nodes.loc[self.nodes[docstring_col] == "null", embedding_col] = zero_str
        self.nodes.loc[self.nodes[docstring_col] == np.nan, embedding_col] = zero_str
        return self.nodes


    def _filter_call_graph(self, remove_isolated=False, remove_subgraph_missing=True):
        """
        Filter call graph nodes based on subgraph nodes.
        """
        # Copy the call graph nodes to avoid modifying the original dataframe
        filtered_cg_nodes = self.nodes.copy()

        self.edges = self.edges[self.edges['source_id'].isin(filtered_cg_nodes['fnc_id']) & self.edges['target_id'].isin(filtered_cg_nodes['fnc_id'])]

        if remove_subgraph_missing:
            # Remove nodes that are not present in the subgraph nodes as func_id
            filtered_cg_nodes = filtered_cg_nodes[filtered_cg_nodes['fnc_id'].isin(self.subgraph_nodes['func_id'])]
        if remove_isolated:
            # Remove nodes (by id) that are not present in the cg_edges dataframe
            filtered_cg_nodes = filtered_cg_nodes[filtered_cg_nodes['fnc_id'].isin(self.edges['source_id']) | filtered_cg_nodes['fnc_id'].isin(self.edges['target_id'])]
        # Filter CG edges, subgraph nodes and subgraph edges
        self.edges = self.edges[self.edges['source_id'].isin(filtered_cg_nodes['fnc_id']) & self.edges['target_id'].isin(filtered_cg_nodes['fnc_id'])]
        self.subgraph_nodes = self.subgraph_nodes[self.subgraph_nodes['func_id'].isin(filtered_cg_nodes['fnc_id'])]
        self.subgraph_edges = self.subgraph_edges[self.subgraph_edges['func_id'].isin(filtered_cg_nodes['fnc_id'])]


        # Filter columns
        col_mask = ['fnc_id', 'combinedName', 'function_code', 'docstring', 'docstring_embedding', 'function_location', 'file_id'] if 'docstring_embedding' in filtered_cg_nodes.columns else ['fnc_id', 'combinedName', 'function_code', 'docstring', 'function_location', 'file_id']
        filtered_cg_nodes = filtered_cg_nodes[col_mask]



        # Reset index of the filtered CG nodes
        self.nodes = filtered_cg_nodes.reset_index(drop=True)
        # Reset function ID-s
        self.nodes['RS_func_id'] = self.nodes.index

        # Change the source_id indexes in the CG edges to match the new func_id
        self.edges = self.edges.merge(self.nodes[['fnc_id', 'RS_func_id']], left_on='source_id', right_on='fnc_id', how='left')
        self.edges = self.edges.drop(columns=['source_id', 'fnc_id']).rename(columns={'RS_func_id': 'source_id'})

        # Change the target_id indexes in the CG edges to match the new func_id
        self.edges = self.edges.merge(self.nodes[['fnc_id', 'RS_func_id']], left_on='target_id', right_on='fnc_id', how='left')
        self.edges = self.edges.drop(columns=['target_id', 'fnc_id']).rename(columns={'RS_func_id': 'target_id'})

        # Edges as integers
        self.edges['source_id'] = self.edges['source_id'].round().astype(int)
        self.edges['target_id'] = self.edges['target_id'].round().astype(int)

        # Update the func_id column in the subgraph nodes accordinf to the new func_id col in the self.nodes (CG)
        self.subgraph_nodes = self.subgraph_nodes.merge(self.nodes[['fnc_id', 'RS_func_id']], left_on='func_id', right_on='fnc_id', how='left')
        self.subgraph_nodes = self.subgraph_nodes.drop(columns=['func_id', 'fnc_id']).rename(columns={'RS_func_id': 'func_id'})

        # Update the func_id column in the subgraph edges according to the new func_id col in the self.nodes (CG)
        self.subgraph_edges = self.subgraph_edges.merge(self.nodes[['fnc_id', 'RS_func_id']], left_on='func_id', right_on='fnc_id', how='left')
        self.subgraph_edges = self.subgraph_edges.drop(columns=['func_id', 'fnc_id']).rename(columns={'RS_func_id': 'func_id'})

        self.nodes = self.nodes.drop(columns=['fnc_id']).rename(columns={'RS_func_id': 'func_id'})


    def _filter_sub_graph(self, remove_isolated=False):
        """
        Filter subgraph nodes and edges based on the function IDs in the call graph.
        """

        filtered_subgraph_nodes = self.subgraph_nodes.copy()

        # Filter subgraph edges based on the filtered subgraph nodes
        tmp1 = self.subgraph_edges.merge(filtered_subgraph_nodes, left_on=['func_id', 'source_id'], right_on=['func_id', 'node_id'])
        tmp2 = tmp1.merge(filtered_subgraph_nodes, left_on=['func_id', 'target_id'], right_on=['func_id', 'node_id'])
        self.subgraph_edges = tmp2[['source_id', 'target_id', 'func_id']]

        # Remove nodes that are not present in the subgraph edges
        filtered_subgraph_nodes = filtered_subgraph_nodes[filtered_subgraph_nodes['node_id'].isin(self.subgraph_edges['source_id']) | filtered_subgraph_nodes['node_id'].isin(self.subgraph_edges['target_id'])]

        # Reset index of the filtered subgraph nodes
        self.subgraph_nodes = filtered_subgraph_nodes.reset_index(drop=True)
        # Reset node_id-s
        self.subgraph_nodes['RS_node_id'] = self.subgraph_nodes.index

        # Change the source_id indexes in the subgraph edges to match the new node_id
        self.subgraph_edges = self.subgraph_edges.merge(self.subgraph_nodes[['func_id', 'node_id', 'RS_node_id']], left_on=['func_id', 'source_id'], right_on=['func_id', 'node_id'], how='left')
        self.subgraph_edges = self.subgraph_edges.drop(columns=['source_id', 'node_id']).rename(columns={'RS_node_id': 'source_id'})

        # Change the target_id indexes in the subgraph edges to match the new node_id
        self.subgraph_edges = self.subgraph_edges.merge(self.subgraph_nodes[['func_id', 'node_id', 'RS_node_id']], left_on=['func_id', 'target_id'], right_on=['func_id', 'node_id'], how='left')
        self.subgraph_edges = self.subgraph_edges.drop(columns=['target_id', 'node_id']).rename(columns={'RS_node_id': 'target_id'})

        # Edges as integers
        self.subgraph_edges['source_id'] = self.subgraph_edges['source_id'].round().astype(int)
        self.subgraph_edges['target_id'] = self.subgraph_edges['target_id'].round().astype(int)

        # Rename node_id column to RS_node_id in the subgraph nodes
        self.subgraph_nodes = self.subgraph_nodes.drop(columns=['node_id']).rename(columns={'RS_node_id': 'node_id'})

        
    def _create_hetero_data(self):

        if 'docstring_embedding' not in self.nodes.columns or 'embedding' not in self.subgraph_nodes.columns:
            raise ValueError("Embeddings are missing in the nodes or subgraph_nodes dataframes. Please enable create_embedding=True when creating the hierarchical graph.")

        data = HeteroData()
        data['function'].x = torch.tensor(self.nodes['docstring_embedding'].tolist(), dtype=torch.float)
        data['expression'].x = torch.tensor(self.subgraph_nodes['embedding'].tolist(), dtype=torch.float)

        data['function', 'calls', 'function'].edge_index = torch.tensor(
            self.edges[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        data['expression', 'connected_to', 'expression'].edge_index = torch.tensor(
            self.subgraph_edges[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        data['expression', 'is_in', 'function'].edge_index = torch.tensor(
            self.hierarchical_sub_to_main_edges[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        data['function', 'contains', 'expression'].edge_index = torch.tensor(
            self.hierarchical_main_to_sub_edges[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        return data
    

    def create_hetero_data_from_df(
            self, 
            nodes_df, 
            edges_df, 
            subgraph_nodes_df, 
            subgraph_edges_df, 
            hier_sub_to_main_edges_df, 
            hier_main_to_sub_edges_df
    ):
        
        nodes_df['docstring_embedding'] = nodes_df['docstring_embedding'].apply(ast.literal_eval)
        subgraph_nodes_df['embedding'] = (
            subgraph_nodes_df['embedding']
            .str.replace("\n", "")
            .str.replace("  ", ",")
            .str.replace(" ", ",")
            .str.replace("[,", "[")
        ).apply(ast.literal_eval)

        data = HeteroData()
        data['function'].x = torch.tensor(nodes_df['docstring_embedding'].tolist(), dtype=torch.float)
        data['expression'].x = torch.tensor(subgraph_nodes_df['embedding'].tolist(), dtype=torch.float)

        data['function', 'calls', 'function'].edge_index = torch.tensor(
            edges_df[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        data['expression', 'connected_to', 'expression'].edge_index = torch.tensor(
            subgraph_edges_df[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        data['expression', 'is_in', 'function'].edge_index = torch.tensor(
            hier_sub_to_main_edges_df[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        data['function', 'contains', 'expression'].edge_index = torch.tensor(
            hier_main_to_sub_edges_df[['source_id', 'target_id']].values.T, 
            dtype=torch.long
        )

        return data
