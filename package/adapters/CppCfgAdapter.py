"""
C++ CFG Adapter for FunctionGraphBuilder
This adapter calls the Joern Flask API to extract CFG from C++ code
"""

import requests
import pandas as pd
from typing import Tuple, List
from package.adapters import CfgAdapter  # Import the CLASS from the module
from package.adapters import LanguageAstAdapterRegistry


@LanguageAstAdapterRegistry.register_cfg('cpp')
class CppCfgAdapter(CfgAdapter):
    """
    Adapter to extract CFG from C++ code using Joern API service.
    Returns data in the same format as FunctionGraphBuilder's CFG extraction.
    
    Inherits from CfgAdapter base class.
    """
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        """
        Initialize the adapter.
        
        Args:
            api_url: The base URL of the Joern Flask API service
        """
        super().__init__(api_url)
        self.health_endpoint = f"{api_url}/health"
        self.cfg_endpoint = f"{api_url}/get_cfg"
    
    def check_health(self) -> bool:
        """
        Check if the Joern service is healthy and available.
        
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def extract_cfg(self, function: pd.DataFrame, language: str = "cpp") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract CFG from C++ code using Joern API.
        
        Args:
            code: The C/C++ source code to analyze
            language: The language ('cpp', 'c++', or 'c')
        
        Returns:
            Tuple of (nodes_df, edges_df):
            - nodes_df: DataFrame with columns ['node_id', 'code']
            - edges_df: DataFrame with columns ['source_id', 'target_id', 'label']
        """
        try:
            # Make API request
            response = requests.post(
                self.cfg_endpoint,
                json={
                    "language": language,
                    "code_snippet": function['function_code'],
                    "method_name": function['name']
                },
                timeout=120
            )
            
            # Check response
            if response.status_code != 200:
                error_data = response.json()
                raise Exception(f"API error: {error_data.get('error', 'Unknown error')}")
            
            # Parse response
            result = response.json()
            
            if not result.get('success'):
                raise Exception(f"CFG extraction failed: {result.get('error', 'Unknown error')}")
            
            # Convert to DataFrames - NO FILTERING, return everything Joern gives us
            nodes_data = result.get('nodes', [])
            edges_data = result.get('edges', [])
            
            # Create nodes DataFrame
            nodes_df = pd.DataFrame(nodes_data)
            if not nodes_df.empty:
                # Ensure correct column types
                nodes_df['node_id'] = nodes_df['node_id'].astype(str)
            else:
                nodes_df = pd.DataFrame(columns=['node_id', 'code'])
            
            # Create edges DataFrame
            edges_df = pd.DataFrame(edges_data)
            if not edges_df.empty:
                # Ensure correct column types
                edges_df['source_id'] = edges_df['source_id'].astype(str)
                edges_df['target_id'] = edges_df['target_id'].astype(str)
                
                # Deduplicate edges (remove duplicate source->target pairs)
                edges_df = edges_df.drop_duplicates(subset=['source_id', 'target_id']).reset_index(drop=True)
                
                # Drop label column if it's empty (to match your format)
                if 'label' in edges_df.columns and edges_df['label'].str.strip().eq('').all():
                    edges_df = edges_df[['source_id', 'target_id']]
            else:
                edges_df = pd.DataFrame(columns=['source_id', 'target_id'])
            
            return nodes_df, edges_df
            
        except requests.exceptions.Timeout:
            print("Request timeout - Joern service took too long to respond")
            return self._create_empty_dataframes()
        
        except requests.exceptions.ConnectionError:
            print(f"Connection error - Make sure Joern service is running at {self.api_url}")
            return self._create_empty_dataframes()
        
        except Exception as e:
            print(f"Error extracting CFG: {e}")
            return self._create_empty_dataframes()