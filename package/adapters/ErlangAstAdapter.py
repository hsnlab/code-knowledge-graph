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
        
        arity = self._count_call_arity(top_call_node)

        # Handle special cases: spawn and apply
        if call_name == 'spawn':
            target_call = self._extract_spawn_target(top_call_node)
            if target_call:
                call_name = target_call

                arity = self._extract_spawn_arity(top_call_node)

            else:
                return []  # Can't resolve spawn target
        
        elif call_name == 'apply':
            target_call = self._extract_apply_target(top_call_node)
            if target_call:
                call_name = target_call

                arity = self._extract_apply_arity(top_call_node)

            else:
                return []  # Can't resolve apply target (probably a fun variable)
        
        call_name = f"{call_name}/{arity}"

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

    def _count_call_arity(self, call_node: Node) -> int:
        """Count the number of arguments in a call."""
        expr_args = None
        for child in call_node.named_children:
            if child.type == 'expr_args':
                expr_args = child
                break
        
        if not expr_args:
            return 0
        
        return len(expr_args.named_children)
    
    def _extract_spawn_arity(self, spawn_call_node: Node) -> int:
        """Extract arity from spawn's Args parameter (3rd argument)."""
        expr_args = None
        for child in spawn_call_node.named_children:
            if child.type == 'expr_args':
                expr_args = child
                break
        
        if not expr_args:
            return 0
        
        # The 3rd argument of spawn is the Args list
        arg_index = 0
        for child in expr_args.named_children:
            if arg_index == 2:  # 3rd argument (0-indexed)
                if child.type == 'list':
                    return len(child.named_children)
            arg_index += 1
        
        return 0
    
    def _extract_apply_arity(self, apply_call_node: Node) -> int:
        """Extract arity from apply's Args parameter (3rd or 2nd argument depending on form)."""
        expr_args = None
        for child in apply_call_node.named_children:
            if child.type == 'expr_args':
                expr_args = child
                break
        
        if not expr_args:
            return 0
        
        # Check if apply(Module, Function, Args) or apply(Fun, Args)
        args_list = list(expr_args.named_children)
        
        if len(args_list) >= 3:
            # apply(Module, Function, Args) - Args is 3rd argument
            if args_list[2].type == 'list':
                return len(args_list[2].named_children)
        elif len(args_list) >= 2:
            # apply(Fun, Args) - Args is 2nd argument
            if args_list[1].type == 'list':
                return len(args_list[1].named_children)
        
        return 0

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
    
    # Extract just function name (no module, no arity) for name column
    def __extract_function_name(self, combined_name):
        without_arity = combined_name.split('/')[0]  # Remove arity
        function_only = without_arity.split(':')[-1]  # Remove module
        return function_only

    def resolve_calls(self, calls: pd.DataFrame, functions: pd.DataFrame, 
                classes: pd.DataFrame, imports: pd.DataFrame, filename_lookup: dict[str, str] = None) -> None:
        """
        For Erlang: Extract function name from fully qualified name.
        - combinedName: Full qualified name with module and arity (e.g., "poolboy:checkout/3")
        - name: Just the function name (e.g., "checkout")
        """
        if calls.empty:
            return
        
        # Store full name as combinedName
        calls['combinedName'] = calls['name']
        
        # Extract just the function name (no module, no arity)

        calls['name'] = calls['combinedName'].apply(self.__extract_function_name)
        if not functions.empty:
            func_mapping = functions.set_index('combinedName')['fnc_id'].to_dict()        
            calls['func_id'] = calls['combinedName'].map(func_mapping)
        else:
            calls['func_id'] = None

    def create_combined_name(self, functions: pd.DataFrame, filename_lookup: dict[str, str] = None) -> None:
        """
        For Erlang: Add module prefix and arity to function names.
        Format: "module_name:function_name/arity"
        - combinedName: Full qualified name (e.g., "poolboy:checkout/3")
        - name: Just the function name (e.g., "checkout")
        """
        if functions.empty:
            return
        
        if not filename_lookup:
            functions['combinedName'] = functions['name']
            return
        
        def add_module_and_arity(row):
            file_id = row['file_id']
            func_name = row['name']
            
            # Count params to get arity
            params = row.get('params', {})
            if isinstance(params, str):
                params = json.loads(params)
            param_count = len(params) if params else 0
            
            # Get module name from filepath
            filepath = filename_lookup.get(file_id)
            if filepath:
                filename = filepath.split('/')[-1].split('\\')[-1]
                if filename.endswith('.erl'):
                    module_name = filename[:-4]
                    return f"{module_name}:{func_name}/{param_count}"
            
            return f"{func_name}/{param_count}"
        
        # Store full qualified name as combinedName
        functions['combinedName'] = functions.apply(add_module_and_arity, axis=1)
        
        functions['name'] = functions['combinedName'].apply(self.__extract_function_name)


    def create_import_edges(self, import_df: pd.DataFrame, cg_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create import nodes and edges."""
        if import_df.empty:
            imports = pd.DataFrame(columns=['ID', 'import_name', 'import_from', 'import_as_name', 'import_file_ids'])
            imp_edges = pd.DataFrame(columns=['source', 'target'])
            return imports, imp_edges
        
        # Group by name, from, and as_name to support both Python and Erlang
        imports_grouped = (
            import_df.groupby(['name', 'from', 'as_name'], dropna=False)['file_id']
            .apply(lambda x: list(set(x)))
            .reset_index()
        )

        imports_grouped.insert(0, 'import_id', range(1, len(imports_grouped) + 1))

        imports = imports_grouped.rename(columns={
            'name': 'import_name',
            'from': 'import_from',
            'as_name': 'import_as_name',
            'file_id': 'import_file_ids'
        })

        imp_edges = imports[['import_id', 'import_file_ids']].explode('import_file_ids')
        imp_edges = imp_edges.rename(columns={'import_file_ids': 'file_id'})

        imp_edges = imp_edges.merge(
            cg_nodes[['func_id', 'file_id']], 
            on='file_id', 
            how='left'
        )
        
        imp_edges = imp_edges[['import_id', 'func_id']].dropna().reset_index(drop=True)
        imp_edges = imp_edges.rename(columns={'import_id': 'source', 'func_id': 'target'})

        return imports, imp_edges