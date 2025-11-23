import unittest
import json
import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from cfg_services.erlang.ErlangCfgParser import ErlangCfgParser

class TestErlangCfgParser(unittest.TestCase):
    """Test ErlangCfgParser CFG extraction"""
    
    @classmethod
    def setUpClass(cls):
        """Setup Phase - Load test configuration"""
        config_path = os.path.join(os.path.dirname(__file__), 'cfg_test_config.json')
        with open(config_path, 'r') as f:
            cls.test_config = json.load(f)
    
    def setUp(self):
        """Setup Phase - Initialize parser"""
        self.parser = ErlangCfgParser()
    
    def _test_specific_cfg_scenario(self, mock, expected_nodes, expected_edges):
        """
        Test a specific CFG scenario by checking graph connectivity.
        
        Args:
            mock: Test Erlang code snippet
            expected_nodes: List of expected node dicts
            expected_edges: List of expected edge dicts
        """
        parser = ErlangCfgParser()
        actual_nodes, actual_edges = parser.parse(mock)
        
        # Build node lookup by code (for matching)
        actual_nodes_by_code = {node['code']: node['node_id'] for node in actual_nodes}
        expected_nodes_by_code = {node['code']: node['node_id'] for node in expected_nodes}
        
        # Check that all expected nodes exist
        missing_nodes = set(expected_nodes_by_code.keys()) - set(actual_nodes_by_code.keys())
        extra_nodes = set(actual_nodes_by_code.keys()) - set(expected_nodes_by_code.keys())
        
        if missing_nodes:
            self.fail(f"Missing nodes in actual CFG: {missing_nodes}")
        if extra_nodes:
            self.fail(f"Extra nodes in actual CFG: {extra_nodes}")
        
        # Build adjacency lists for comparison
        actual_adjacency = {}
        for edge in actual_edges:
            src_id = edge['source_id']
            tgt_id = edge['target_id']
            # Find the code for these IDs
            src_code = next((n['code'] for n in actual_nodes if n['node_id'] == src_id), None)
            tgt_code = next((n['code'] for n in actual_nodes if n['node_id'] == tgt_id), None)
            
            if src_code not in actual_adjacency:
                actual_adjacency[src_code] = []
            actual_adjacency[src_code].append(tgt_code)
        
        expected_adjacency = {}
        for edge in expected_edges:
            src_id = edge['source_id']
            tgt_id = edge['target_id']
            # Find the code for these IDs
            src_code = next((n['code'] for n in expected_nodes if n['node_id'] == src_id), None)
            tgt_code = next((n['code'] for n in expected_nodes if n['node_id'] == tgt_id), None)
            
            if src_code not in expected_adjacency:
                expected_adjacency[src_code] = []
            expected_adjacency[src_code].append(tgt_code)
        
        # Sort adjacency lists for comparison
        for code in actual_adjacency:
            actual_adjacency[code] = sorted(actual_adjacency[code])
        for code in expected_adjacency:
            expected_adjacency[code] = sorted(expected_adjacency[code])
        
        # Compare adjacency lists
        if actual_adjacency != expected_adjacency:
            print("\n=== ACTUAL ADJACENCY ===")
            for src, targets in sorted(actual_adjacency.items()):
                print(f"{src} -> {targets}")
            print("\n=== EXPECTED ADJACENCY ===")
            for src, targets in sorted(expected_adjacency.items()):
                print(f"{src} -> {targets}")
            self.fail("Graph connectivity mismatch")


# Generate test methods dynamically
def generate_tests():
    config_path = os.path.join(os.path.dirname(__file__), 'cfg_test_config.json')
    with open(config_path, 'r') as f:
        test_config = json.load(f)
    
    cfg_tests = test_config.get("standalone_cfg_tests")
    if cfg_tests is not None:
        for test_name, test_data in cfg_tests.items():
            # Capture test_data in closure properly
            test_method = lambda self, \
                mock=test_data.get("mock"), \
                expected_nodes=test_data.get("expected_nodes"), \
                expected_edges=test_data.get("expected_edges"): \
                self._test_specific_cfg_scenario(mock, expected_nodes, expected_edges)
            
            test_method.__name__ = f'test_erlang_cfg_{test_name}'
            test_method.__doc__ = f'Test Erlang CFG {test_name}'
            setattr(TestErlangCfgParser, test_method.__name__, test_method)

generate_tests()

if __name__ == '__main__':
    unittest.main(verbosity=2)