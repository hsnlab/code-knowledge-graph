"""
Base CFG Adapter Class
Parent class for all CFG extraction adapters
"""

import pandas as pd
from abc import abstractmethod
from typing import Tuple, List, Optional


class CfgAdapter():
    """
    Abstract base class for CFG extraction adapters.
    
    All language-specific CFG adapters should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the CFG adapter.
        
        Args:
            api_url: Optional URL for API-based adapters (e.g., "http://localhost:5000")
        """
        self.api_url = api_url
    
    @abstractmethod
    def extract_cfg(self, function: pd.DataFrame, language: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract CFG from a single code snippet.
        
        Args:
            code: The source code to analyze
            language: The programming language
        
        Returns:
            Tuple of (nodes_df, edges_df):
            - nodes_df: DataFrame with columns ['node_id', 'code']
            - edges_df: DataFrame with columns ['source_id', 'target_id', 'label']
        
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def extract_cfg_batch(self, code_list: List[str], language: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract CFG from multiple code snippets in batch.
        
        Args:
            code_list: List of source code strings to analyze
            language: The programming language
        
        Returns:
            Tuple of (nodes_df, edges_df) with combined results:
            - nodes_df: DataFrame with columns ['node_id', 'code']
            - edges_df: DataFrame with columns ['source_id', 'target_id', 'label']
        
        Must be implemented by subclasses.
        """
        pass
    
    def check_health(self) -> bool:
        """
        Check if the adapter/service is healthy and available.
        
        Returns:
            bool: True if healthy, False otherwise
        
        Optional method - can be overridden by subclasses.
        Default implementation returns True.
        """
        return True
    
    def _create_empty_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper method to create empty DataFrames with correct schema.
        
        Returns:
            Tuple of (empty_nodes_df, empty_edges_df)
        """
        nodes_df = pd.DataFrame(columns=['node_id', 'code'])
        edges_df = pd.DataFrame(columns=['source_id', 'target_id'])
        return nodes_df, edges_df
    
    def _filter_isolated_nodes(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to filter out isolated nodes (nodes not in any edge).
        
        Args:
            nodes_df: DataFrame with nodes
            edges_df: DataFrame with edges
        
        Returns:
            Filtered nodes_df with only connected nodes
        """
        if nodes_df.empty or edges_df.empty:
            return nodes_df
        
        # Get all nodes that appear in edges
        connected_nodes = set(edges_df['source_id']) | set(edges_df['target_id'])
        
        # Filter nodes to only include connected ones
        filtered_nodes = nodes_df[nodes_df['node_id'].isin(connected_nodes)].reset_index(drop=True)
        
        return filtered_nodes
    
    def __repr__(self):
        """String representation of the adapter."""
        class_name = self.__class__.__name__
        if self.api_url:
            return f"{class_name}(api_url='{self.api_url}')"
        return f"{class_name}()"
