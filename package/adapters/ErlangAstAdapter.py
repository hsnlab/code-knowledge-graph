from package.adapters import LanguageAstAdapter
from package.adapters import NodeType

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
            "remote": NodeType.CALL,
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