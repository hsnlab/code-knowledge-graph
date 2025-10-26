from package.adapters import LanguageAstAdapter, NodeType
import pandas as pd
from tree_sitter import Node, Tree, Parser
import json

class AstProcessor:

    def __init__(self, adapter: LanguageAstAdapter, file_content):

        self.imports = pd.DataFrame(columns=['file_id', 'imp_id', 'name', 'from', 'as_name'])
        self.classes = pd.DataFrame(columns=['file_id', 'cls_id', 'name', 'base_classes'])
        self.functions = pd.DataFrame(columns=['file_id', 'fnc_id', 'name', 'class', 'class_base_classes', 'params', 'docstring', 'function_code', 'class_id', 'return_type'])
        self.calls = pd.DataFrame(columns=['file_id', 'cll_id', 'name', 'call_position', 'class', 'class_base_classes', 'class_id', 'func_id', 'func_name', 'func_params'])
        self.adapter = adapter
        parser: Parser = adapter.get_tree_sitter_parser()
        self.tree: Tree = parser.parse(file_content)

    def process_file_ast(self, file_id: str | None=None, id_dict: dict[str, int] = {}, return_dataframes: bool = True) -> None | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
        self.id_dict = id_dict
        root_node: Node = self.tree.root_node
        if root_node is None:
            return None

        self.__walk_ast(root_node, file_id=file_id)
        if return_dataframes:
            return self.imports, self.classes, self.functions, self.calls, {
                "imp_id": self.id_dict.get("imp_id"),
                "cls_id": self.id_dict.get("cls_id"),
                "fnc_id": self.id_dict.get("fnc_id"),
                "cll_id": self.id_dict.get("cll_id")
            }

    def __is_not_correct_type(self, node: Node, type_expected: NodeType):
        normalized_type: NodeType = self.adapter.map_node_type(node.type)
        return normalized_type is None or normalized_type != type_expected

    def __update_indexes_and_dataframe(self, parsed_data: list[pd.DataFrame], data_frame_to_update: pd.DataFrame, key_to_update: str) -> pd.DataFrame | None:
        if parsed_data is None or len(parsed_data) == 0:
            return data_frame_to_update
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
        if self.adapter.should_skip_function_node(node):
            return None

        if self.__is_not_correct_type(node, NodeType.FUNCTION):
            return None
        functions = self.adapter.parse_functions(top_function_node=node, current_class_name=current_class_name,
                    class_base_classes=current_base_classes, file_id=file_id, fnc_id=fnc_id, class_id=class_id)
        self.functions = self.__update_indexes_and_dataframe(functions, self.functions, "fnc_id")
        return functions[0] if len(functions) > 0 else None

    def _handle_calls(self, node: Node, file_id: str, current_class_name: str, current_base_classes: list[str],
                      class_id: int, fnc_id: int, func_name: str, func_params: dict, cll_id: int):

        if self.adapter.should_skip_call_node(node):
            return

        if self.__is_not_correct_type(node, NodeType.CALL):
            return

        calls = self.adapter.parse_calls(top_call_node=node, current_class_name=current_class_name,
            class_base_classes=current_base_classes, file_id=file_id, fnc_id=fnc_id, class_id=class_id, cll_id=cll_id,
            func_name=func_name, func_params=func_params)

        self.calls = self.__update_indexes_and_dataframe(calls, self.calls, "cll_id")

    def __walk_ast(self, node: Node, file_id: str, class_name='Global', class_base_classes=None, class_id=None,
                   fnc_id=None, func_name=None, func_params=None) -> None:
        #todo update id mapper

        # Handle imports
        self._handle_imports(node, file_id=file_id, current_import_id=self.id_dict.get("imp_id", None))

        # Handle classes
        current_class = self._handle_class_definitions(node, file_id=file_id,
                                                       current_class_id=self.id_dict.get("cls_id", None))
        if class_base_classes is None:
            class_base_classes = []

        if current_class is not None:
            current_class = current_class.iloc[0]
            class_name = current_class["name"]
            class_base_classes = current_class["base_classes"]
            class_id = current_class["cls_id"]

        # Handle functions
        current_function = self._handle_function_definitions(node, file_id=file_id, fnc_id=self.id_dict.get("fnc_id", None),
            current_class_name=class_name, current_base_classes=class_base_classes, class_id=class_id)

        if current_function is not None:
            current_function = current_function.iloc[0]
            func_name = current_function["name"]
            func_params = json.loads(current_function["params"])
            fnc_id = current_function["fnc_id"]

        # Handle calls (using current function context)
        self._handle_calls(node, file_id=file_id, cll_id=self.id_dict.get("cll_id", None),
                           current_class_name=class_name, current_base_classes=class_base_classes,
                           class_id=class_id, fnc_id=fnc_id, func_name=func_name, func_params=func_params)

        # Recurse to children with updated context
        for child in node.children:
            self.__walk_ast(child, file_id, class_name=class_name, class_base_classes=class_base_classes,
                            class_id=class_id, fnc_id=fnc_id, func_name=func_name, func_params=func_params)