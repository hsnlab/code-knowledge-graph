from package.adapters import LanguageAstAdapter
from package.adapters import NodeType
import json 
from tree_sitter import Node
import pandas as pd

from package.adapters.LanguageAstAdapterRegistry import LanguageAstAdapterRegistry

@LanguageAstAdapterRegistry.register('erlang')
class ErlangAstAdapter(LanguageAstAdapter):

    def __init__(self):
        super().__init__(language="erlang", mapper={
            "import_attribute": NodeType.IMPORT,  # For actual -import() statements
            "fun_decl": NodeType.FUNCTION,
            "call": NodeType.CALL,
            #"remote": NodeType.CALL,
        })

    def parse_import(self, top_import_node: Node, file_id: str, imp_id: int) -> list[pd.DataFrame]:
        imports: list = list()

        # Erlang doesn't have 'from' or 'as' - just the module name
        from_module = None
        as_name = None

        # For Erlang: -import(lists, [map/2, filter/2]).
        # We want to extract the module name (first atom child)
        for child in top_import_node.named_children:
            name = None

            # The module name is an 'atom' type
            if child.type == 'atom':
                name = child.text.decode('utf-8')

                if name:
                    as_name = name
                    new_row = pd.DataFrame([{
                        'file_id': file_id,
                        'imp_id': imp_id,
                        'name': name,
                        'from': from_module,
                        'as_name': as_name
                    }])
                    imports.append(new_row)
                    # Only get the first atom (module name), ignore the function list
                    break

        return imports

    def parse_class(self, top_class_node: Node, file_id: str, cls_id: int) -> list[pd.DataFrame]:
        return []

    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list,
                        file_id: str, fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        """Parse an Erlang function clause."""
        functions = []

        # Erlang fun_decl contains a function_clause
        function_clause = None
        for child in top_function_node.named_children:
            if child.type == 'function_clause':
                function_clause = child
                break

        if not function_clause:
            return []

        # Extract function name
        name = self._extract_function_name(function_clause)

        # Extract parameters
        params = self._extract_parameters(function_clause)

        # Erlang has no return types or docstrings
        return_type = None
        docstring = None

        # Function code is the entire fun_decl
        function_code = top_function_node.text.decode('utf-8')

        import json
        return [pd.DataFrame([{
            'file_id': file_id,
            'fnc_id': fnc_id,
            'name': name,
            'class': current_class_name,
            'class_base_classes': class_base_classes,
            'params': json.dumps(params),
            'docstring': docstring,
            'function_code': function_code,
            'class_id': class_id,
            'return_type': return_type
        }])]

    def _extract_function_name(self, function_clause: Node) -> str:
        """Extract function name from function_clause."""
        for child in function_clause.named_children:
            if child.type == 'atom':
                return child.text.decode('utf-8')
        return '<anonymous>'

    def _extract_parameters(self, function_clause: Node) -> dict[str, str]:
        """Extract parameters from expr_args in function_clause."""
        params = {}

        # Find expr_args node
        expr_args = None
        for child in function_clause.named_children:
            if child.type == 'expr_args':
                expr_args = child
                break

        if not expr_args:
            return params

        # Extract all variables from expr_args
        self._collect_vars_from_node(expr_args, params)

        return params

    def _collect_vars_from_node(self, node: Node, params: dict):
        """Recursively collect variables and pattern values from a node."""
        if node.type == 'var':
            var_name = node.text.decode('utf-8')
            # Skip ONLY the anonymous underscore '_', not variables like '_Age'
            if var_name != '_':  # ← Changed this line
                params[var_name] = 'Any'
        elif node.type in ['integer', 'atom', 'string']:
            # Include literal patterns as parameters
            literal_value = node.text.decode('utf-8')
            params[literal_value] = 'Any'

        # Recursively traverse children
        for child in node.named_children:
            self._collect_vars_from_node(child, params)

    def parse_calls(self, top_call_node: Node, file_id: str, cll_id: int,
                    current_class_name: str, class_base_classes: list, class_id: int,
                    fnc_id: int, func_name: str, func_params: dict) -> list[pd.DataFrame]:
        """
        Parse Erlang function calls from AST.
        Handles:
        - Local calls: say_hello()
        - Remote calls: io:format()
        - Spawn: spawn(Module, Function, Args) -> extract Module:Function
        - Apply: apply(Module, Function, Args) -> extract Module:Function
        """
        calls = []
        
        call_name = self._extract_call_name(top_call_node)
        
        if not call_name:
            return []
        
        # Handle special cases: spawn and apply
        if call_name == 'spawn':
            target_call = self._extract_spawn_target(top_call_node)
            if target_call:
                call_name = target_call
            else:
                return []  # Can't resolve spawn target
        
        elif call_name == 'apply':
            target_call = self._extract_apply_target(top_call_node)
            if target_call:
                call_name = target_call
            else:
                return []  # Can't resolve apply target (probably a fun variable)
        
        # Create the call entry
        new_row = pd.DataFrame([{
            'file_id': file_id,
            'cll_id': cll_id,
            'name': call_name,
            'call_position': top_call_node.start_byte,
            'class': current_class_name,
            'class_base_classes': class_base_classes,
            'class_id': class_id,
            'func_id': fnc_id,
            'func_name': func_name,
            'func_params': json.dumps(func_params)
        }])
        calls.append(new_row)
        
        return calls

    def _extract_call_name(self, call_node: Node) -> str | None:
        """
        Extract the call name from a call node.
        Returns:
        - Local call: "say_hello"
        - Remote call: "io:format"
        - spawn/apply: "spawn"/"apply" (special handling later)
        """
        # First check if this is a remote call
        remote_node = None
        atom_node = None
        
        for child in call_node.named_children:
            if child.type == 'remote':
                remote_node = child
            elif child.type == 'atom':
                atom_node = child
        
        # Priority: remote call first
        if remote_node:
            module_name = None
            func_name = None
            
            for remote_child in remote_node.named_children:
                if remote_child.type == 'remote_module':
                    # Extract module name
                    for mod_child in remote_child.named_children:
                        if mod_child.type == 'atom':
                            module_name = mod_child.text.decode('utf-8')
                            break
                
                elif remote_child.type == 'atom':
                    # Extract function name
                    func_name = remote_child.text.decode('utf-8')
            
            if module_name and func_name:
                return f"{module_name}:{func_name}"
        
        # If no remote call, check for local call
        if atom_node:
            return atom_node.text.decode('utf-8')
        
        return None

    def _extract_spawn_target(self, spawn_call_node: Node) -> str | None:
        """
        Extract the target function from spawn(Module, Function, Args).
        
        Examples:
        - spawn(?MODULE, server, []) -> "server"
        - spawn(other_module, handle, []) -> "other_module:handle"
        
        Returns the function name to be recorded as a call.
        """
        # Find expr_args node
        expr_args = None
        for child in spawn_call_node.named_children:
            if child.type == 'expr_args':
                expr_args = child
                break
        
        if not expr_args:
            return None
        
        # Extract first 2 arguments: Module, Function
        args = []
        for child in expr_args.named_children:
            if child.type == 'atom':
                args.append(child.text.decode('utf-8'))
            elif child.type == 'macro_call_expr':
                # ?MODULE expands to current module - treat as local call
                args.append('?MODULE')
            
            if len(args) >= 2:
                break
        
        if len(args) < 2:
            return None
        
        module_arg = args[0]
        function_arg = args[1]
        
        # If module is ?MODULE, it's a local call
        if module_arg == '?MODULE':
            return function_arg
        else:
            # Remote call
            return f"{module_arg}:{function_arg}"

    def _extract_apply_target(self, apply_call_node: Node) -> str | None:
        """
        Extract the target function from apply(Module, Function, Args).
        
        Examples:
        - apply(lists, reverse, []) -> "lists:reverse"
        - apply(Fun, []) -> None (can't resolve)
        
        Returns the function name to be recorded as a call, or None if variable.
        """
        # Find expr_args node
        expr_args = None
        for child in apply_call_node.named_children:
            if child.type == 'expr_args':
                expr_args = child
                break
        
        if not expr_args:
            return None
        
        # Check if first arg is a variable (fun) or atom (module)
        first_child = None
        for child in expr_args.named_children:
            if child.type in ['atom', 'var']:
                first_child = child
                break
        
        if not first_child:
            return None
        
        # If first arg is a variable, we can't resolve it statically
        if first_child.type == 'var':
            return None
        
        # First arg is atom - this is apply(Module, Function, Args)
        # Extract Module and Function
        args = []
        for child in expr_args.named_children:
            if child.type == 'atom':
                args.append(child.text.decode('utf-8'))
            
            if len(args) >= 2:
                break
        
        if len(args) < 2:
            return None
        
        module_name = args[0]
        function_name = args[1]
        
        return f"{module_name}:{function_name}"
    
    def should_skip_call_node(self, node: Node) -> bool:
        """Only process actual 'call' type nodes."""
        return node.type != 'call'