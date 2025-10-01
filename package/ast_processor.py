from package.adapters import LanguageAdapter, NodeType
import pandas as pd
from tree_sitter import Node, Tree, Parser

class AstProcessor:

    def __init__(self, adapter: LanguageAdapter, file_content):

        self.imports = pd.DataFrame(columns=['file_id', 'imp_id', 'name', 'from', 'as_name'])
        self.classes = pd.DataFrame(columns=['file_id', 'cls_id', 'name', 'base_classes'])
        self.functions = pd.DataFrame(columns=['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params'])
        self.calls = pd.DataFrame(columns=['file_id', 'cll_id', 'name', 'class', 'class_base_classes'])
        self.adapter = adapter
        parser: Parser = adapter.get_tree_sitter_parser()
        self.tree: Tree = parser.parse(file_content)

    def process_file_ast(self, file_id: str | None=None, id_dict: dict[str, int] = {}, return_dataframes: bool = True) -> None | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.id_dict = id_dict
        root_node: Node = self.tree.root_node
        if root_node is None:
            return None

        self.__walk_ast(root_node, file_id=file_id)
        if return_dataframes:
            return self.imports, self.classes, self.functions, self.calls

    def _handle_imports(self, node: Node, file_id: str, current_import_id):
        normalized_type: NodeType = self.adapter.map_node_type(node.type)
        if normalized_type is None or normalized_type != NodeType.IMPORT:
            return
        imports = self.adapter.parse_import(top_import_node=node, file_id=file_id, imp_id=current_import_id)
        if len(imports) == 0:
            return
        if self.id_dict.get("imp_id", None) is not None:
            self.id_dict["imp_id"] += len(imports)
        all_import_df = [self.imports] + imports
        self.imports = pd.concat(all_import_df, ignore_index=True)



    def __walk_ast(self, node: Node, file_id: str) -> None:
        self._handle_imports(node, file_id=file_id, current_import_id=self.id_dict.get("imp_id", None))

        for child in node.children:
            self.__walk_ast(child, file_id)