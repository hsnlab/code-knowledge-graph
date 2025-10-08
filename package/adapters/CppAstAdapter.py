from package.adapters import LanguageAstAdapter
from package.adapters import NodeType

from tree_sitter import Node
import pandas as pd

from package.adapters.LanguageAstAdapterRegistry import LanguageAstAdapterRegistry

@LanguageAstAdapterRegistry.register('cpp')
class CppAstAdapter(LanguageAstAdapter):

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

    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list, file_id: str,
                        fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        return []