from package.adapters import LanguageAstAdapter, NodeType
import pandas as pd
from tree_sitter import Node, Tree, Parser

class AstProcessor:

    def __init__(self, adapter: LanguageAstAdapter, file_content):

        self.imports = pd.DataFrame(columns=['file_id', 'imp_id', 'name', 'from', 'as_name'])
        self.classes = pd.DataFrame(columns=['file_id', 'cls_id', 'name', 'base_classes'])
        self.functions = pd.DataFrame(columns=['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params', 'docstring', 'function_code', 'class_id', 'return_type'])
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
            return None
        classes = self.adapter.parse_class(top_class_node=node, file_id=file_id, cls_id=current_class_id)

        self.classes = self.__update_indexes_and_dataframe(classes, self.classes, "cls_id")
        return classes[0] if len(classes) > 0 else None

    def _handle_function_definitions(self, node: Node, current_class_name: str ,current_base_classes: list[str], file_id: str, fnc_id, class_id):
        if node.type == 'function_definition' and node.parent and node.parent.type == 'decorated_definition':
            return

        if self.__is_not_correct_type(node, NodeType.FUNCTION):
            return
        functions = self.adapter.parse_functions(top_function_node=node, current_class_name=current_class_name,
                    class_base_classes=current_base_classes, file_id=file_id, fnc_id=fnc_id, class_id=class_id)
        self.functions = self.__update_indexes_and_dataframe(functions, self.functions, "cls_id")


    def __walk_ast(self, node: Node, file_id: str, class_name='Global', class_base_classes=list(), class_id=None ) -> None:
        # todo update the id mapper
        self._handle_imports(node, file_id=file_id, current_import_id=self.id_dict.get("imp_id", None))
        current_class = self._handle_class_definitions(node, file_id=file_id, current_class_id=self.id_dict.get("cls_id", None))
        class_name=class_name if not None else "Global"
        class_base_classes=class_base_classes if not None else []
        class_id=class_id if not None else None
        if current_class is not None:
            current_class = current_class.iloc[0]
            class_name = current_class["name"]
            class_base_classes = current_class["base_classes"]
            class_id=current_class["cls_id"]
        self._handle_function_definitions(node, file_id=file_id, fnc_id=self.id_dict.get("fnc_id", None), current_class_name=class_name,
                                          current_base_classes=class_base_classes,class_id=class_id)
        for child in node.children:
            self.__walk_ast(child, file_id, class_name=class_name, class_base_classes=class_base_classes, class_id=class_id)