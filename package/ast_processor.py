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

    def __is_not_correct_type(self, node: Node, type_expected: NodeType):
        normalized_type: NodeType = self.adapter.map_node_type(node.type)
        return normalized_type is None or normalized_type != type_expected

    def __update_indexes_and_dataframe(self, parsed_data: list[pd.DataFrame], data_frame_to_update: pd.DataFrame, key_to_update: str) -> pd.DataFrame | None:
        if parsed_data is None or len(parsed_data) == 0:
            return
        if self.id_dict.get(key_to_update, None) is not None:
            self.id_dict[key_to_update] += len(parsed_data)
        all_import_df = [data_frame_to_update] + parsed_data
        return pd.concat(all_import_df, ignore_index=True)

    def _handle_imports(self, node: Node, file_id: str, current_import_id):
        if self.__is_not_correct_type(node, NodeType.IMPORT):
            return
        imports = self.adapter.parse_import(top_import_node=node, file_id=file_id, imp_id=current_import_id)
        self.imports = self.__update_indexes_and_dataframe(imports, self.imports, "imp_id")

    def _handle_class_definitions(self, node: Node, file_id: str, current_class_id):
        if self.__is_not_correct_type(node, NodeType.CLASS):
            return
        classes = self.adapter.parse_class(top_class_node=node, file_id=file_id, cls_id=current_class_id)

        self.classes = self.__update_indexes_and_dataframe(classes, self.classes, "cls_id")

    def __walk_ast(self, node: Node, file_id: str) -> None:
        # todo update the id mapper
        self._handle_imports(node, file_id=file_id, current_import_id=self.id_dict.get("imp_id", None))
        self._handle_class_definitions(node, file_id=file_id, current_class_id=self.id_dict.get("cls_id", None))

        for child in node.children:
            self.__walk_ast(child, file_id)