from tree_sitter import Node
from package.adapters.erlang.CfgNodeType import CfgNodeType, CfgNode
from package.adapters.erlang.branch_parsers.BranchParser import BranchParser

class MaybeParser(BranchParser):
    """Parser for maybe expressions"""
    
    def parse(self, maybe_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        """Parse maybe expression and return merge node"""
        children = list(maybe_node.named_children)
        
        else_start_idx = None
        for i, child in enumerate(children):
            if child.type == 'cr_clause':
                else_start_idx = i
                break
        
        # Step 1: Process maybe body expressions sequentially
        end_idx = else_start_idx if else_start_idx is not None else len(children)
        
        current = entry_node
        for i in range(end_idx):
            expr = children[i]
            exit_from_expr = self.parser.process_expression(expr, current)
            current = exit_from_expr
        
        maybe_body_exit = current
        
        # Step 2: Process else clauses
        if else_start_idx is not None:
            else_clauses = [child for child in children[else_start_idx:] if child.type == 'cr_clause']
            
            clause_exits = []
            
            for clause in else_clauses:
                pattern = self._get_pattern_from_clause(clause)
                pattern_node = self.parser._create_node(pattern, CfgNodeType.BRANCH)
                
                self.parser._create_edge(maybe_body_exit, pattern_node)
                
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
                
                # Process clause body
                clause_body = self.parser._find_node_by_type(clause, 'clause_body')
                current = clause_current
                
                if clause_body:
                    for expr in clause_body.named_children:
                        if expr.type == '->':
                            continue
                        exit_from_expr = self.parser.process_expression(expr, current)
                        current = exit_from_expr
                
                clause_exits.append((pattern_node, current))
            
            # Create merge
            merge = self.parser._create_node("merge", CfgNodeType.MERGE)
            
            # Connect maybe body success to merge
            self.parser._create_edge(maybe_body_exit, merge)
            
            # Connect all else patterns and their clause exits to merge
            for pattern_node, exit_node in clause_exits:
                # Failure edge: pattern doesn't match
                self.parser._create_edge(pattern_node, merge)
                # Success edge: pattern body completes
                self.parser._create_edge(exit_node, merge)
            
            return merge
        else:
            return maybe_body_exit
    
    def _get_pattern_from_clause(self, clause: Node) -> str:
        """Get pattern from else clause"""
        for child in clause.named_children:
            if child.type not in ['clause_body', 'guard']:
                return child.text.decode('utf-8')
        return "_"