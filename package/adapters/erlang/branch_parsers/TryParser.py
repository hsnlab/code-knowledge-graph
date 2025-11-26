from tree_sitter import Node
from package.adapters.erlang.CfgNodeType import CfgNodeType, CfgNode
from package.adapters.erlang.branch_parsers.BranchParser import BranchParser

class TryParser(BranchParser):
    """Parser for try-catch-after expressions"""
    
    def parse(self, try_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        """Parse try expression and return merge node"""
        # Step 1: Separate children into categories
        children = list(try_node.named_children)
        
        try_body_exprs = []
        of_clauses = []
        catch_clauses = []
        after_clause = None
        
        for child in children:
            if child.type == 'cr_clause':
                of_clauses.append(child)
            elif child.type == 'catch_clause':
                catch_clauses.append(child)
            elif child.type == 'try_after':
                after_clause = child
            else:
                if not of_clauses and not catch_clauses and not after_clause:
                    try_body_exprs.append(child)
        
        # Step 2: Process try body sequentially
        current = entry_node
        try_body_nodes = []  # Track all try body expression nodes
        
        for expr in try_body_exprs:
            exit_from_expr = self.parser.process_expression(expr, current)
            try_body_nodes.append(exit_from_expr)
            current = exit_from_expr
        
        try_body_exit = current
        
        # Step 3: Process of clauses (if present)
        of_merge = None
        if of_clauses:
            clause_exits = []
            
            for clause in of_clauses:
                pattern = self._get_pattern_from_of_clause(clause)
                pattern_node = self.parser._create_node(pattern, CfgNodeType.BRANCH)
                
                # Connect pattern from try_body_exit
                self.parser._create_edge(try_body_exit, pattern_node)
                
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
                
                if clause_body:
                    for expr in clause_body.named_children:
                        if expr.type == '->':
                            continue
                        exit_from_expr = self.parser.process_expression(expr, clause_current)
                        clause_current = exit_from_expr
                
                clause_exits.append(clause_current)
            
            # Create merge for of clauses
            of_merge = self.parser._create_node("merge", CfgNodeType.MERGE)
            
            for exit_node in clause_exits:
                self.parser._create_edge(exit_node, of_merge)
        
        # Step 4: Process catch clauses (if any)
        catch_exits = []
        
        if catch_clauses:
            for catch_clause in catch_clauses:
                # Get exception pattern
                pattern = self._get_exception_pattern(catch_clause)
                pattern_node = self.parser._create_node(pattern, CfgNodeType.BRANCH)
                
                for try_node in try_body_nodes:
                    self.parser._create_edge(try_node, pattern_node)
                
                # Process handler body
                clause_body = self.parser._find_node_by_type(catch_clause, 'clause_body')
                clause_current = pattern_node
                
                if clause_body:
                    for expr in clause_body.named_children:
                        if expr.type == '->':
                            continue
                        exit_from_expr = self.parser.process_expression(expr, clause_current)
                        clause_current = exit_from_expr
                
                catch_exits.append(clause_current)
        
        # Step 5: Create final merge
        if catch_clauses:
            final_merge = self.parser._create_node("merge", CfgNodeType.MERGE)
            
            # Connect 'of' merge (or try body if no 'of') to final merge
            if of_merge:
                self.parser._create_edge(of_merge, final_merge)
            else:
                self.parser._create_edge(try_body_exit, final_merge)
            
            # Connect all catch exits to final merge
            for exit_node in catch_exits:
                self.parser._create_edge(exit_node, final_merge)
        else:
            final_merge = of_merge if of_merge else try_body_exit
        
        # Step 6: Handle 'after' clause
        if after_clause:
            current = final_merge
            
            for expr in after_clause.named_children:
                exit_from_expr = self.parser.process_expression(expr, current)
                current = exit_from_expr
            
            return current

        return final_merge
    
    def _get_pattern_from_of_clause(self, clause: Node) -> str:
        """Get pattern from 'of' clause (same as case clause)"""
        for child in clause.named_children:
            if child.type not in ['clause_body', 'guard']:
                return child.text.decode('utf-8')
        return "_"
    
    def _get_exception_pattern(self, catch_clause: Node) -> str:
        """
        Get exception pattern from catch_clause.
        Format: class:pattern or just pattern
        """
        parts = []
        
        for child in catch_clause.named_children:
            if child.type == 'try_class':
                class_name = child.named_children[0].text.decode('utf-8') if child.named_children else 'error'
                parts.append(class_name)
            elif child.type not in ['clause_body', 'guard', 'try_stack']:
                parts.append(child.text.decode('utf-8'))
        
        return ':'.join(parts) if len(parts) > 1 else (parts[0] if parts else "_:_")