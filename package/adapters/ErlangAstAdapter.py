from package.adapters import LanguageAstAdapter
from package.adapters import NodeType

from tree_sitter import Node
import pandas as pd

from package.adapters.LanguageAstAdapterRegistry import LanguageAstAdapterRegistry

@LanguageAstAdapterRegistry.register('erlang')
class ErlangAdapter(LanguageAstAdapter):

    def __init__(self):
        super().__init__(language="erlang", mapper={
            "import_attribute": NodeType.IMPORT,  # For actual -import() statements
            "fun_decl": NodeType.FUNCTION,
            "function_clause": NodeType.FUNCTION,
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

    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list, file_id: str,
                        fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        return []