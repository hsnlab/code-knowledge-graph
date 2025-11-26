from tree_sitter_language_pack import get_parser
from tree_sitter import Tree, Node
from typing import List, Dict, Tuple, Optional

from package.adapters.erlang.CfgNodeType import CfgNodeType, CfgNode

from package.adapters.erlang.branch_parsers.BranchParser import  BranchParser
from package.adapters.erlang.branch_parsers.CaseParser import CaseParser
from package.adapters.erlang.branch_parsers.IfParser import IfParser
from package.adapters.erlang.branch_parsers.ReceiveParser import ReceiveParser
from package.adapters.erlang.branch_parsers.TryParser import TryParser
from package.adapters.erlang.branch_parsers.MaybeParser import MaybeParser

class ErlangCfgParser:

    BRANCH_NODES = {
        'case_expr',
        'if_expr', 
        'receive_expr',
        'try_expr',
        'maybe_expr'
    }
    
    # Function/method calls
    CALL_NODES = {
        'call',      # Regular function call: foo(X)
        'remote'     # Remote call: module:function(X)
    }
    
    # Sequential/atomic expressions (basic blocks)
    BASIC_BLOCK_NODES = {
        # Assignment/matching
        'match_expr',           # A = B
        'cond_match_expr',      # A ?= B (maybe expr)
        
        # Operations
        'binary_op_expr',       # A + B, A andalso B, etc
        'unary_op_expr',        # -A, not B, etc
        
        # Literals
        'atom',
        'var',
        'integer',
        'float',
        'char',
        'string',
        
        # Data structures
        'list',                 # [1,2,3]
        'tuple',                # {1,2,3}
        'binary',               # <<1,2,3>>
        'map_expr',             # #{a => 1}
        'map_expr_update',      # Map#{a => 1}
        'record_expr',          # #record{field=val}
        'record_update_expr',   # Rec#record{field=val}
        'record_field_expr',    # Rec#record.field
        'record_index_expr',    # #record.field
        
        # Comprehensions
        'list_comprehension',   # [X || X <- List]
        'binary_comprehension', # <<X || X <= Binary>>
        'map_comprehension',    # #{K => V || K := V <- Map}
        
        # Grouping/blocks
        'paren_expr',           # (Expr)
        'block_expr',           # begin ... end
        
        # Function values
        'internal_fun',         # fun name/arity
        'external_fun',         # fun module:name/arity
        'anonymous_fun',        # fun(X) -> X end
        'fun_type',             # fun((...) -> type)
        
        # Type expressions
        'ann_type',             # Var :: Type
        'pipe',                 # Type1 | Type2
        'range_type',           # 1..10
        
        # Special
        'catch_expr',           # catch Expr
        'dotdotdot',            # ...
        'concatables',          # "str" "ing" (string concatenation)
        
        # Macros
        'macro_call_expr',      # ?MACRO or ?MACRO(Args)
        'macro_string',         # ??MACRO
        
    }
    
    # Binary operators that need special handling (short-circuit evaluation)
    SHORT_CIRCUIT_OPS = {'andalso', 'orelse'}
    
    # Binary op that's a call (message send)
    SEND_OP = '!'



    def __init__(self):
        self.nodes: List[CfgNode] = []
        self.edges: List[Dict] = []
        self.LANGUAGE = "erlang"
        self.parser = get_parser(self.LANGUAGE)
        self.node_id_counter = 0
        self.entry: Optional[CfgNode] = None
        self.exit: Optional[CfgNode] = None
        self.BRANCH_REGISTRY: dict[str, BranchParser] = {
            'case_expr': CaseParser(self),
            'if_expr': IfParser(self),
            'receive_expr': ReceiveParser(self),
            'try_expr': TryParser(self),
            'maybe_expr': MaybeParser(self)
        }

    def parse(self, function_code: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse Erlang function code and return CFG nodes and edges.
        
        Returns:
            (nodes, edges) where:
            - nodes: [{'node_id': 'node_0', 'code': 'Entry', 'type': 'Entry'}, ...]
            - edges: [{'source_id': 'node_0', 'target_id': 'node_1'}, ...]
        """

        self.nodes = []
        self.edges = []
        self.node_id_counter = 0
        self.entry = None
        self.exit = None
        
        # Parse the code
        tree: Tree = self.parser.parse(bytes(function_code, "utf8"))
        root: Node = tree.root_node
        
        # Build CFG
        self._build_cfg(root, function_code)
        
        # Convert nodes to dict format
        nodes_dict = [node.to_dict() for node in self.nodes]
        
        return nodes_dict, self.edges

    def _create_node(self, code: str, node_type: CfgNodeType) -> CfgNode:
        """Create a CFG node and return it."""
        node_id = self.node_id_counter
        self.node_id_counter += 1
        
        node = CfgNode(node_id, code, node_type)
        self.nodes.append(node)
        
        # Track entry and exit
        if node_type == CfgNodeType.ENTRY:
            self.entry = node
        elif node_type == CfgNodeType.EXIT:
            self.exit = node
        
        return node

    def _create_edge(self, source: CfgNode, target: CfgNode):
        """Create an edge between two nodes."""
        if source is None:
            pass
        self.edges.append({
            'source_id': source.node_id,
            'target_id': target.node_id
        })

    def _find_node_by_type(self, node: Node, target_type: str) -> Optional[Node]:
        """Recursively search for a node with the given type."""
        # Check current node
        if node.type == target_type:
            return node
        
        # Recursively search children
        for child in node.named_children:
            result = self._find_node_by_type(child, target_type)
            if result:
                return result
        
        return None

    def _build_cfg(self, root: Node, code: str):
        """Main CFG building logic."""
        # Step 1: Create entry node
        entry_node = self._create_node("Entry", CfgNodeType.ENTRY)
        
        # Step 2: Create exit node
        exit_node = self._create_node("Exit", CfgNodeType.EXIT)
         
        function_declaration_node: Node =  self._find_node_by_type(root, 'function_clause')
        full_text = function_declaration_node.text.decode('utf-8')
        signature = full_text.split('->')[0].strip()
        function_declaration_node: CfgNode = self._create_node(signature, CfgNodeType.FUNCTION_DECLARATION)
        
        self._create_edge(entry_node, function_declaration_node)
        # Step 4: find function body, and start building CFG from AST
        function_body: Node = self._find_node_by_type(root, 'clause_body')
        prev_node = function_declaration_node

        for child in function_body.named_children:
            if child.type == '->':  
                continue
            exit_node = self.process_expression(child, prev_node)  
            prev_node = exit_node  

        self._create_edge(prev_node, self.exit)


    def _handle_branch(self, node: Node, parent_node: CfgNode) -> CfgNode:
        """Handle branch expressions by delegating to appropriate parser"""
        parser = self.BRANCH_REGISTRY.get(node.type, None)
        if parser is None:
            raise NotImplementedError(f"Branch type '{node.type}' not implemented")
    
        return parser.parse(node, parent_node, "")
    
    def process_expression(self, node: Node, parent_node: CfgNode) -> CfgNode:
        """Process expression and return its exit node"""
        expression_type: str = node.type
        
        if expression_type in self.CALL_NODES:
            # Create call node
            call: CfgNode = self._create_node(
                node.text.decode('utf-8'),
                CfgNodeType.CALL
            )
            self._create_edge(parent_node, call)
            return call

        if expression_type in self.BASIC_BLOCK_NODES:
            # Create basic block
            block: CfgNode = self._create_node(
                node.text.decode('utf-8'),
                CfgNodeType.BASIC_BLOCK
            )
            self._create_edge(parent_node, block)
            return block 
        
        elif expression_type in self.BRANCH_NODES:
            # Handle branch and return its exit (merge node)
            branch_exit = self._handle_branch(node, parent_node)
            return branch_exit 
        
        else:
            # in case of other type handle it as Block
            block: CfgNode = self._create_node(
                node.text.decode('utf-8'),
                CfgNodeType.BASIC_BLOCK
            )
            self._create_edge(parent_node, block)
            return block 

