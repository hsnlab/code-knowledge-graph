from tree_sitter_language_pack import get_parser
from tree_sitter import Parser
from enum import Enum

class NodeType(Enum):
    IMPORT = "Import"
    CLASS = "Class"
    FUNCTION = "Function"
    CALL = "Call"

class LanguageAdapter:
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

"""
Mapper to store language-specific adapters and parsers.

Provides:
- Access to language-specific Tree-sitter parsers.
- Node type mappings for normalized categories (Import, Class, Function, Call).
"""

adapter_mapper: dict[str, LanguageAdapter] = {
    "python": LanguageAdapter(language="python", mapper={
        "import_statement": NodeType.IMPORT,
        "import": NodeType.IMPORT,
        "class_definition": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "call": NodeType.CALL,
    }),
    "cpp": LanguageAdapter(language="cpp", mapper={
        "preproc_include": NodeType.IMPORT,
        "class_specifier": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "call_expression": NodeType.CALL,
    }),
    "erlang": LanguageAdapter(language="erlang", mapper={
    "import_attribute": NodeType.IMPORT,  # For actual -import() statements
    "fun_decl": NodeType.FUNCTION,
    "function_clause": NodeType.FUNCTION,
    "call": NodeType.CALL,
    "remote": NodeType.CALL,
})
}


