from tree_sitter_language_pack import get_parser
from tree_sitter import Parser
from enum import Enum
from tree_sitter import Node
import pandas as pd

class NodeType(Enum):
    IMPORT = "Import"
    CLASS = "Class"
    FUNCTION = "Function"
    CALL = "Call"

class LanguageAstAdapter:
    """
    Base class for mapping Tree-sitter nodes to normalized categories:
    Import, Class, Function, Call.
    """
    def __init__(self, language: str, mapper: dict[str, NodeType]):
        self.LANGUAGE = language
        self.NODE_MAPPER = mapper

    def map_node_type(self, node_type: str) -> None | NodeType:
        return self.NODE_MAPPER.get(node_type)

    def get_tree_sitter_parser(self) -> None | Parser:
        try:
            parser = get_parser(self.LANGUAGE)
            return parser
        except LookupError:
            return None

    def parse_import(self, top_import_node: Node, file_id: str, imp_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

    def parse_class(self, top_class_node: Node, file_id: str, cls_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

    def should_skip_function_node(self, node: Node) -> bool:
        """
        Check if this function node should be skipped during traversal.
        Override in language-specific adapters for custom logic.
        """
        return False

    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list, file_id: str,
                        fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

    def parse_calls(self, top_call_node: Node, file_id: str, cll_id: int,
                    current_class_name: str, class_base_classes: list, class_id: int,
                    fnc_id: int, func_name: str, func_params: dict) -> list[pd.DataFrame]:
        raise NotImplementedError