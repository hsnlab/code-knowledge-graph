from package.adapters import CfgAdapter 
from package.adapters import LanguageAstAdapterRegistry
from package.adapters.erlang.ErlangCfgParser import ErlangCfgParser
from typing import Tuple, List
import pandas as pd



@LanguageAstAdapterRegistry.register_cfg('erlang')
class ErlangCfgAdapter(CfgAdapter):
    
    def extract_cfg(self, function: pd.DataFrame, language: str = "erlang") -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        parser = ErlangCfgParser()
        nodes_data, edges_data = parser.parse(function['function_code'])
        nodes_df = pd.DataFrame(nodes_data)
        if not nodes_df.empty:

            nodes_df['node_id'] = nodes_df['node_id'].astype(str)
        else:
            nodes_df = pd.DataFrame(columns=['node_id', 'code'])
        
        # Create edges DataFrame
        edges_df = pd.DataFrame(edges_data)
        if not edges_df.empty:

            edges_df['source_id'] = edges_df['source_id'].astype(str)
            edges_df['target_id'] = edges_df['target_id'].astype(str)
            
            # Deduplicate edges (remove duplicate source->target pairs)
            edges_df = edges_df.drop_duplicates(subset=['source_id', 'target_id']).reset_index(drop=True)
            
            if 'label' in edges_df.columns and edges_df['label'].str.strip().eq('').all():
                edges_df = edges_df[['source_id', 'target_id']]
        else:
            edges_df = pd.DataFrame(columns=['source_id', 'target_id'])
        
        return nodes_df, edges_df