from package.adapters import LanguageAdapter
import pandas as pd
from tree_sitter import Node, Tree, Parser

class AstProcessor:

    def __init__(self, adapter: LanguageAdapter, file_content):

        self.imports = pd.DataFrame(columns=['file_id', 'imp_id', 'name', 'from', 'as_name'])
        self.classes = pd.DataFrame(columns=['file_id', 'cls_id', 'name', 'base_classes'])
        self.functions = pd.DataFrame(columns=['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params'])
        self.calls = pd.DataFrame(columns=['file_id', 'cll_id', 'name', 'class', 'class_base_classes'])
        parser: Parser = adapter.get_tree_sitter_parser()
        self.tree: Tree = parser.parse(file_content)

    def process_file_ast(self, return_dataframes: bool = True, file_id: str | None=None) -> None | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        root_node: Node = self.tree.root_node
        if root_node is None:
            return None

        self.__walk_ast(root_node, file_id=file_id)
        if return_dataframes:
            return self.imports, self.classes, self.functions, self.calls

    def _handle_imports(self, node: Node, file_id: str):
        pass

    def __walk_ast(self, node: Node, file_id: str) -> None:
        self._handle_imports(node, file_id=file_id)