from tree_sitter import Tree, Node
from cfg_services.erlang.CfgNodeType import CfgNodeType, CfgNode
from cfg_services.erlang.branch_parsers.BranchParser import BranchParser

class CaseParser(BranchParser):
    """Parser for case expressions"""
    
    def parse(self, case_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        """Parse case expression and return merge node"""
        # Step 1: Extract case expression
        case_expr = self._get_case_expression(case_node)
        
        # Step 2: Create case evaluation node
        case_eval = self.parser._create_node(f"case {case_expr}", CfgNodeType.BRANCH)
        self.parser._create_edge(entry_node, case_eval)
        
        # Step 3: Get all clauses
        clauses = [child for child in case_node.named_children if child.type == 'cr_clause']
        
        # Step 4: Process each clause
        clause_exits = []
        prev_node = case_eval
        
        for clause in clauses:
            # Extract pattern
            pattern = self._get_pattern_from_clause(clause)
            pattern_node = self.parser._create_node(pattern, CfgNodeType.BRANCH)
            self.parser._create_edge(prev_node, pattern_node)
            
            # Find clause body
            clause_body = self.parser._find_node_by_type(clause, 'clause_body')
            
            current = pattern_node
            if clause_body:
                for expr in clause_body.named_children:
                    if expr.type == '->':
                        continue
                    self.parser.process_expression(expr, current)
                    current = self.parser.nodes[-1]
            
            clause_exits.append(current)
            prev_node = pattern_node
        
        # Step 5: Create merge
        merge = self.parser._create_node("merge", CfgNodeType.MERGE)
        
        # Step 6: Connect all exits to merge
        for exit_node in clause_exits:
            self.parser._create_edge(exit_node, merge)
        
        # Connect last pattern failure to merge
        self.parser._create_edge(prev_node, merge)
        
        return merge  # ← RETURNS THE MERGE NODE
    
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