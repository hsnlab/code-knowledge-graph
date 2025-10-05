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

    def parse_import(self, top_import_node: Node, file_id: str, imp_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

    def parse_class(self, top_class_node: Node, file_id: str, cls_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

    def parse_functions(self, top_function_node: Node, current_class: str, file_id: str, fnc_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

class PythonAdapter(LanguageAdapter):
    def __init__(self):
        super().__init__(language="python", mapper={
        "import_statement": NodeType.IMPORT,
        "import_from_statement": NodeType.IMPORT,
        "class_definition": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "call": NodeType.CALL,
    })

    def parse_import(self, top_import_node: Node, file_id: str, imp_id: int) -> list[pd.DataFrame]:
        imports: list = list()

        from_module = None
        as_name = None
        # Handle: from module import name
        # Handle: from module import name as n
        if top_import_node.type == 'import_from_statement':
            module_node = top_import_node.children_by_field_name('module_name')
            from_module = module_node[0].text.decode('utf-8') if module_node else None

        for child in top_import_node.named_children:
            name = child.text.decode('utf-8')
            if top_import_node.type == 'import_from_statement' and from_module == name:
                continue #edge case, we have to exclude the package name because we are importing specific parts of it not the whole


            if child.type == 'aliased_import':
                dotted = child.children_by_field_name('name')[0]
                name = dotted.text.decode('utf-8')
                alias = child.children_by_field_name('alias')[0]
                as_name = alias.text.decode('utf-8')

            if as_name is None:
                as_name = name
            new_row = pd.DataFrame([{
                'file_id': file_id,
                'imp_id': imp_id,
                'name': name,
                'from': from_module,
                'as_name': as_name
            }])
            imports.append(new_row)
            as_name = None # reset as_name
        return imports

    def parse_class(self, top_class_node: Node, file_id: str, cls_id: int) -> list[pd.DataFrame]:
        classes = []

        # Python class: class MyClass(BaseClass1, BaseClass2):
        # We need to extract:
        # - class name
        # - base classes (if any)

        name = None
        base_classes = []

        for child in top_class_node.named_children:
            # Get class name
            if child.type == 'identifier':
                name = child.text.decode('utf-8')

            # Get base classes from argument_list
            elif child.type == 'argument_list':
                for base in child.named_children:
                    if base.type == 'identifier':
                        base_classes.append(base.text.decode('utf-8'))
                    elif base.type == 'attribute':
                        # Handle cases like: class MyClass(module.BaseClass)
                        base_classes.append(base.text.decode('utf-8'))

        # Convert base_classes list to string (or keep as list, depends on your needs)
        base_classes_str = base_classes if base_classes else [ ]

        new_row = pd.DataFrame([{
            'file_id': file_id,
            'cls_id': cls_id,
            'name': name,
            'base_classes': base_classes_str
        }])
        classes.append(new_row)

        return classes

    def parse_functions(self, top_function_node: Node, current_class: str, file_id: str, fnc_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

class CppAdapter(LanguageAdapter):

    def __init__(self):
        super().__init__(language="cpp", mapper={
        "preproc_include": NodeType.IMPORT,
        "class_specifier": NodeType.CLASS,
        "struct_specifier": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "call_expression": NodeType.CALL,
    })


    def parse_import(self, top_import_node: Node, file_id: str, imp_id: int) -> list[pd.DataFrame]:
        imports: list = list()

        # C++ doesn't have 'from' or 'as' - just the include
        from_module = None
        as_name = None

        for child in top_import_node.named_children:
            name = None

            # System includes: #include <iostream>
            if child.type == 'system_lib_string':
                # Remove < and > brackets
                name = child.text.decode('utf-8').strip('<>')

            # User includes: #include "myheader.h"
            elif child.type == 'string_literal':
                # Remove quotes
                name = child.text.decode('utf-8').strip('"')
            if as_name is None:
                as_name = name
            if name:
                new_row = pd.DataFrame([{
                    'file_id': file_id,
                    'imp_id': imp_id,
                    'name': name,
                    'from': from_module,
                    'as_name': as_name
                }])
                imports.append(new_row)

        return imports

    def parse_class(self, top_class_node: Node, file_id: str, cls_id: int) -> list[pd.DataFrame]:
        classes = []

        # C++ class: class MyClass : public BaseClass1, private BaseClass2 { ... };
        # We need to extract:
        # - class name
        # - base classes (if any)

        name = None
        base_classes = []

        for child in top_class_node.named_children:
            # Get class name - it's a type_identifier in C++
            if child.type == 'type_identifier':
                name = child.text.decode('utf-8')

            # Get base classes from base_class_clause
            elif child.type == 'base_class_clause':
                for base in child.named_children:
                    # Each base class is in a base_class_specifier
                    if base.type in ['type_identifier', 'qualified_identifier']:
                        base_classes.append(base.text.decode('utf-8'))
                    elif base.type == 'base_class_specifier':
                        # Look for the actual type inside the specifier
                        for specifier_child in base.named_children:
                            if specifier_child.type in ['type_identifier', 'qualified_identifier']:
                                base_classes.append(specifier_child.text.decode('utf-8'))

        # Convert base_classes list to string
        base_classes_str = base_classes if base_classes else []

        new_row = pd.DataFrame([{
            'file_id': file_id,
            'cls_id': cls_id,
            'name': name,
            'base_classes': base_classes_str
        }])
        classes.append(new_row)

        return classes

    def parse_functions(self, top_function_node: Node, current_class: str, file_id: str, fnc_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

class ErlangAdapter(LanguageAdapter):

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

    def parse_functions(self, top_function_node: Node, current_class: str, file_id: str, fnc_id: int) -> list[pd.DataFrame]:
        raise NotImplementedError

"""
Mapper to store language-specific adapters and parsers.

Provides:
- Access to language-specific Tree-sitter parsers.
- Node type mappings for normalized categories (Import, Class, Function, Call).
"""

adapter_mapper: dict[str, LanguageAdapter] = {
    "python": PythonAdapter(),
    "cpp": CppAdapter(),
    "erlang": ErlangAdapter()
}


