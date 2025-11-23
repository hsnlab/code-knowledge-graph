from tree_sitter import Node
from cfg_services.erlang.CfgNodeType import CfgNodeType, CfgNode
from cfg_services.erlang.branch_parsers.BranchParser import BranchParser

class TryParser(BranchParser):
    """Parser for try-catch-after expressions"""
    
    def parse(self, try_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        """Parse try expression and return merge node"""
        # Step 1: Find boundaries - get all children types
        children = list(try_node.named_children)

        # Find where catch_clause starts and try_after
        catch_start_idx = None
        after_clause = None

        for i, child in enumerate(children):
            if child.type == 'catch_clause' and catch_start_idx is None:
                catch_start_idx = i
            elif child.type == 'try_after':
                after_clause = child
        
        # Step 2: Process try body (everything before catch_clause)
        end_idx = catch_start_idx if catch_start_idx is not None else len(children)
        if after_clause and catch_start_idx is None:
            # Find after_clause position
            for i, child in enumerate(children):
                if child.type == 'try_after':
                    end_idx = i
                    break
        
        current = entry_node
        for i in range(end_idx):
            expr = children[i]
            exit_from_expr = self.parser.process_expression(expr, current)
            current = exit_from_expr
        
        try_body_exit = current
        
        # Step 3: Process catch clauses
        catch_exits = []
        prev_catch = try_body_exit
        
        if catch_start_idx is not None:
            for i in range(catch_start_idx, len(children)):
                child = children[i]
                if child.type != 'catch_clause':
                    break
                
                # Get exception pattern
                pattern = self._get_exception_pattern(child)
                pattern_node = self.parser._create_node(pattern, CfgNodeType.BRANCH)
                
                # Connect from try body ONLY to first catch handler
                if len(catch_exits) == 0:
                    self.parser._create_edge(try_body_exit, pattern_node)
                
                # Connect failure edges
                if len(catch_exits) > 0:
                    self.parser._create_edge(prev_catch, pattern_node)
                
                # Process handler body
                clause_body = self.parser._find_node_by_type(child, 'clause_body')
                clause_current = pattern_node
                
                if clause_body:
                    for expr in clause_body.named_children:
                        if expr.type == '->':
                            continue
                        exit_from_expr = self.parser.process_expression(expr, clause_current)
                        clause_current = exit_from_expr
                
                catch_exits.append(clause_current)
                prev_catch = pattern_node
        
        # Step 4: Create merge
        merge = self.parser._create_node("merge", CfgNodeType.MERGE)
        
        # Connect try body success to merge
        self.parser._create_edge(try_body_exit, merge)
        
        # Connect all catch exits to merge
        for exit_node in catch_exits:
            self.parser._create_edge(exit_node, merge)
        
        # Connect last catch pattern failure to merge
        if len(catch_exits) > 0:
            self.parser._create_edge(prev_catch, merge)
        
        # Step 5: Handle 'after' clause
        if after_clause:
            
            current = merge
            
            for expr in after_clause.named_children:
                exit_from_expr = self.parser.process_expression(expr, current)
                current = exit_from_expr
            
            return current

        return merge
    
    def _get_exception_pattern(self, catch_clause: Node) -> str:
        """
        Get exception pattern from catch_clause.
        Format: class:pattern or just pattern
        """
        parts = []
        
        for child in catch_clause.named_children:
            if child.type == 'try_class':
                # Get class name (before the :)
                class_name = child.named_children[0].text.decode('utf-8') if child.named_children else 'error'
                parts.append(class_name)
            elif child.type not in ['clause_body', 'guard', 'try_stack']:
                # This is the pattern
                parts.append(child.text.decode('utf-8'))
        
        return ':'.join(parts) if len(parts) > 1 else (parts[0] if parts else "_:_")