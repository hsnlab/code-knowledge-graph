from tree_sitter import Tree, Node
from package.adapters.erlang.CfgNodeType import CfgNodeType, CfgNode
from package.adapters.erlang.branch_parsers.BranchParser import BranchParser

class CaseParser(BranchParser):
    """Parser for case expressions"""
    
    def parse(self, case_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        """Parse case expression and return merge node"""
        case_expr = self._get_case_expression(case_node)
        case_eval = self.parser._create_node(f"case {case_expr}", CfgNodeType.BRANCH)
        self.parser._create_edge(entry_node, case_eval)
        
        clauses = [child for child in case_node.named_children if child.type == 'cr_clause']
        clause_exits = []
        
        for i, clause in enumerate(clauses):
            pattern = self._get_pattern_from_clause(clause)
            pattern_node = self.parser._create_node(pattern, CfgNodeType.BRANCH)
            
            # Connect pattern directly from case_eval
            self.parser._create_edge(case_eval, pattern_node)
            
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
            
            clause_exits.append(current)
        
        # Create merge node
        merge = self.parser._create_node("merge", CfgNodeType.MERGE)
        
        # Connect all clause exits to merge
        for exit_node in clause_exits:
            self.parser._create_edge(exit_node, merge)
        
        return merge
    
    def _get_case_expression(self, case_node: Node) -> str:
        """Get the expression being evaluated"""
        for child in case_node.named_children:
            if child.type not in ['cr_clause']:
                return child.text.decode('utf-8')
        return "?"
    
    def _get_pattern_from_clause(self, clause: Node) -> str:
        """Get pattern from clause"""
        for child in clause.named_children:
            if child.type not in ['clause_body', 'guard']:
                return child.text.decode('utf-8')
        return "_"