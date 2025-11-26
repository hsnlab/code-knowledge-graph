from tree_sitter import Node
from package.adapters.erlang.CfgNodeType import CfgNodeType, CfgNode
from package.adapters.erlang.branch_parsers.BranchParser import BranchParser

class ReceiveParser(BranchParser):
    """Parser for receive expressions"""
    
    def parse(self, receive_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        """Parse receive expression and return merge node"""
        
        # Get all cr_clauses (message patterns)
        clauses = [child for child in receive_node.named_children if child.type == 'cr_clause']
        
        # Process each clause
        clause_exits = []
        
        for i, clause in enumerate(clauses):
            
            pattern = self._get_pattern_from_clause(clause)
            pattern_node = self.parser._create_node(pattern, CfgNodeType.BRANCH)
            
            # Connect pattern directly from entry_node
            self.parser._create_edge(entry_node, pattern_node)
            
            clause_current = pattern_node
            
            # Check for guard
            guard = None
            for child in clause.named_children:
                if child.type == 'guard':
                    guard = child
                    break
            
            if guard:
                guard_code = guard.text.decode('utf-8')
                guard_node = self.parser._create_node(guard_code, CfgNodeType.BRANCH)
                self.parser._create_edge(clause_current, guard_node)
                clause_current = guard_node
            
            # Find clause body
            clause_body = self.parser._find_node_by_type(clause, 'clause_body')
            
            current = clause_current
            if clause_body:
                for expr in clause_body.named_children:
                    if expr.type == '->':
                        continue
                    exit_from_expr = self.parser.process_expression(expr, current)
                    current = exit_from_expr
            
            clause_exits.append(current)
        
        # Handle optional after clause
        after_clause = None
        for child in receive_node.named_children:
            if child.type == 'receive_after':
                after_clause = child
                break
        
        if after_clause:
            # Get timeout expression
            timeout_expr = self._get_timeout_from_after(after_clause)
            timeout_node = self.parser._create_node(timeout_expr, CfgNodeType.BRANCH)
            
            # Connect timeout directly from entry_node
            self.parser._create_edge(entry_node, timeout_node)
            
            # Process after body
            after_body = self.parser._find_node_by_type(after_clause, 'clause_body')
            current = timeout_node
            if after_body:
                for expr in after_body.named_children:
                    if expr.type == '->':
                        continue
                    exit_from_expr = self.parser.process_expression(expr, current)
                    current = exit_from_expr
            
            clause_exits.append(current)
        
        # Create merge
        merge = self.parser._create_node("merge", CfgNodeType.MERGE)
        
        # Connect all clause exits to merge
        for exit_node in clause_exits:
            self.parser._create_edge(exit_node, merge)
        
        return merge
    
    def _get_pattern_from_clause(self, clause: Node) -> str:
        """Get pattern from clause"""
        for child in clause.named_children:
            if child.type not in ['clause_body', 'guard']:
                return child.text.decode('utf-8')
        return "_"
    
    def _get_timeout_from_after(self, after_clause: Node) -> str:
        """Get timeout expression from after clause"""
        for child in after_clause.named_children:
            if child.type not in ['clause_body']:
                return f"after {child.text.decode('utf-8')}"
        return "after infinity"