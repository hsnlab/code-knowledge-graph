from tree_sitter_language_pack import get_parser
from tree_sitter import Parser

class LanguageAdapter:
    """
    Base class for mapping Tree-sitter nodes to normalized categories:
    Import, Class, Function, Call.
    """
    def __init__(self, language: str, mapper: dict[str, str]):
        self.LANGUAGE = language
        self.NODE_MAPPER = mapper

    def map_node_type(self, node_type: str) -> str:
        return self.NODE_MAPPER.get(node_type)

    def get_tree_sitter_parser(self) -> None | Parser:
        try:
            parser = get_parser(self.language)
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
        "import_statement": "Import",
        "import": "Import",
        "class_definition": "Class",
        "function_definition": "Function",
        "call": "Call",
    }),
    "cpp": LanguageAdapter(language="cpp", mapper={
        "preproc_include": "Import",
        "class_specifier": "Class",
        "function_definition": "Function",
        "call_expression": "Call",
    }),
    "erlang": LanguageAdapter(language="erlang", mapper={
        "module_attribute": "Import",
        "export_attribute": "Import",
        "fun_decl": "Function",
        "function_clause": "Function",
        "call": "Call",
        "remote": "Call",
    })
}


