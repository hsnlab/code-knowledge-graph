from tree_sitter import Node
from package.adapters.erlang.CfgNodeType import CfgNodeType, CfgNode
from package.adapters.erlang.branch_parsers.BranchParser import BranchParser

class IfParser(BranchParser):
    """Parser for if expressions"""
    
    def parse(self, if_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        """Parse if expression and return merge node"""
        clauses = [child for child in if_node.named_children if child.type == 'if_clause']
        clause_exits = []
        
        for i, clause in enumerate(clauses):
            guard = self._get_guard_from_clause(clause)
            guard_node = self.parser._create_node(guard, CfgNodeType.BRANCH)
            
            # Connect guard directly from entry_node
            self.parser._create_edge(entry_node, guard_node)
            
            clause_body = self.parser._find_node_by_type(clause, 'clause_body')
            current = guard_node
            
            if clause_body:
                for expr in clause_body.named_children:
                    if expr.type == '->':
                        continue
                    exit_from_expr = self.parser.process_expression(expr, current)
                    current = exit_from_expr
            
            clause_exits.append(current)
        
        # Create merge node
        merge = self.parser._create_node("merge", CfgNodeType.MERGE)
        
        # Connect all clause exits to merge
        for exit_node in clause_exits:
            self.parser._create_edge(exit_node, merge)
        
        return merge
    
    def _get_guard_from_clause(self, clause: Node) -> str:
        """Get guard expression from if_clause"""
        for child in clause.named_children:
            if child.type == 'guard':
                return child.text.decode('utf-8')
        return "true"