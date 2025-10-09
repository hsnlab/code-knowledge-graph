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
        "friend_declaration": NodeType.FUNCTION,
        "template_declaration": NodeType.FUNCTION,
       # "field_declaration": NodeType.FUNCTION,
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

    def should_skip_function_node(self, node: Node) -> bool:
        """Skip function nodes inside template_declaration (we process the wrapper instead)."""
        if node.parent and node.parent.type == 'template_declaration':
            # Skip function_definition or nested template_declaration inside a template
            return node.type in ['function_definition', 'template_declaration']
        return False

    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list,
                        file_id: str, fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        """Parse C++ function/method definition."""
        functions = []

        # Check if field_declaration is actually a function (pure virtual)
        if top_function_node.type == 'field_declaration':
            # Only process if it contains a function_declarator
            has_function_declarator = any(
                child.type == 'function_declarator'
                for child in top_function_node.named_children
            )
            if not has_function_declarator:
                return []  # Not a function, skip

        # Unwrap template declarations
        actual_function_node = top_function_node
        if top_function_node.type == 'template_declaration':
            # Find the function_definition inside (skip if it's a template class)
            for child in top_function_node.named_children:
                if child.type == 'function_definition':
                    actual_function_node = child
                    break
                elif child.type in ['class_specifier', 'struct_specifier']:
                    # This is a template class/struct, not a function template
                    return []  # Return empty, skip this node

        # Handle friend declarations
        if actual_function_node.type == 'friend_declaration':
            for child in actual_function_node.named_children:
                if child.type == 'declaration':
                    actual_function_node = child
                    break

        # Extract function components
        name = self._extract_function_name(actual_function_node, current_class_name)
        params = self._extract_parameters(actual_function_node)
        return_type = self._extract_return_type(actual_function_node)
        function_code = top_function_node.text.decode('utf-8')
        docstring = None

        import json
        return [pd.DataFrame([{
            'file_id': file_id,
            'fnc_id': fnc_id,
            'name': name,
            'class': current_class_name,
            'class_base_classes': class_base_classes,
            'params': json.dumps(params),
            'docstring': docstring,
            'function_code': function_code,
            'class_id': class_id,
            'return_type': return_type
        }])]

    def _extract_function_name(self, func_node: Node, current_class_name: str) -> str:
        """Extract function name from function_definition or declaration."""
        # Look for function_declarator
        for child in func_node.named_children:
            if child.type == 'function_declarator':
                # Extract name from inside declarator
                for subchild in child.named_children:
                    if subchild.type in ['identifier', 'field_identifier']:
                        return subchild.text.decode('utf-8')
                    elif subchild.type == 'destructor_name':
                        return subchild.text.decode('utf-8')
                    elif subchild.type == 'operator_name':
                        return subchild.text.decode('utf-8')

        return '<anonymous>'

    def _extract_parameters(self, func_node: Node) -> dict[str, str]:
        """Extract function parameters with types."""
        params = {}

        # Find function_declarator, then parameter_list
        func_declarator = None
        for child in func_node.named_children:
            if child.type == 'function_declarator':
                func_declarator = child
                break

        if not func_declarator:
            return params

        # Find parameter_list inside function_declarator
        param_list = None
        for child in func_declarator.named_children:
            if child.type == 'parameter_list':
                param_list = child
                break

        if not param_list:
            return params

        # Extract each parameter
        for param_child in param_list.named_children:
            if param_child.type in ['parameter_declaration', 'optional_parameter_declaration']:
                param_name, param_type = self._extract_param_info(param_child)
                if param_name:
                    params[param_name] = param_type

        return params

    def _extract_param_info(self, param_node: Node) -> tuple[str | None, str]:
        """Extract parameter name and type from a parameter declaration."""
        param_name = None
        param_type_parts = []

        for child in param_node.named_children:
            # Collect type information
            if child.type == 'type_qualifier':
                param_type_parts.append(child.text.decode('utf-8'))
            elif child.type == 'primitive_type':
                param_type_parts.append(child.text.decode('utf-8'))
            elif child.type == 'type_identifier':
                param_type_parts.append(child.text.decode('utf-8'))

            # Extract name from declarators
            elif child.type == 'identifier':
                param_name = child.text.decode('utf-8')
            elif child.type == 'pointer_declarator':
                # For pointers like "int* ptr"
                for subchild in child.named_children:
                    if subchild.type == 'identifier':
                        param_name = subchild.text.decode('utf-8')
                param_type_parts.append('*')
            elif child.type == 'reference_declarator':
                # For references like "int& ref"
                for subchild in child.named_children:
                    if subchild.type == 'identifier':
                        param_name = subchild.text.decode('utf-8')
                param_type_parts.append('&')

        # Construct full type string
        param_type = ' '.join(param_type_parts) if param_type_parts else 'void'

        return param_name, param_type

    def _extract_return_type(self, func_node: Node) -> str | None:
        """Extract return type - None for constructors/destructors."""
        for child in func_node.named_children:
            if child.type == 'primitive_type':
                return child.text.decode('utf-8')
            elif child.type == 'type_identifier':
                return child.text.decode('utf-8')

        # No return type found (constructor/destructor)
        return None