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
        "new_expression": NodeType.CALL,
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
    
    def _extract_class_from_qualified_name(self, func_node: Node) -> str | None:
        """Extract class name from qualified function definition like MyClass::method()."""
        for child in func_node.named_children:
            if child.type == 'function_declarator':
                for subchild in child.named_children:
                    if subchild.type in ['qualified_identifier', 'scoped_identifier']:
                        text = subchild.text.decode('utf-8')
                        
                        # MyClass::method -> MyClass
                        # Outer::Inner::method -> Outer::Inner
                        # Container<T>::add -> Container
                        if '::' in text:
                            # Remove template args if present
                            if '<' in text:
                                text = text.split('<')[0] + text.split('>')[-1]
                            
                            parts = text.split('::')
                            # Return everything except the last part (method name)
                            return '::'.join(parts[:-1]) if len(parts) > 1 else None
        return None

    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list,
                        file_id: str, fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        """Parse C++ function/method definition."""
        functions = []

        # Check if field_declaration is actually a function (pure virtual)
        if top_function_node.type == 'field_declaration':
            has_function_declarator = any(
                child.type == 'function_declarator'
                for child in top_function_node.named_children
            )
            if not has_function_declarator:
                return []

        # Unwrap template declarations
        actual_function_node = top_function_node
        if top_function_node.type == 'template_declaration':
            for child in top_function_node.named_children:
                if child.type == 'function_definition':
                    actual_function_node = child
                    break
                elif child.type in ['class_specifier', 'struct_specifier']:
                    return []

        # Handle friend declarations
        if actual_function_node.type == 'friend_declaration':
            for child in actual_function_node.named_children:
                if child.type == 'declaration':
                    actual_function_node = child
                    break

        # Extract function components
        name = self._extract_function_name(actual_function_node, current_class_name)
        
        # EXTRACT CLASS NAME FROM QUALIFIED IDENTIFIER
        extracted_class_name = self._extract_class_from_qualified_name(actual_function_node)
        if extracted_class_name:
            current_class_name = extracted_class_name
            # Also clear class_id since this is defined outside
            class_id = None
        
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
        
        for child in func_node.named_children:
            if child.type == 'function_declarator':
                # Extract name from inside declarator
                for subchild in child.named_children:
                    # ADD qualified_identifier and scoped_identifier:
                    if subchild.type in ['identifier', 'field_identifier', 'destructor_name', 
                                        'operator_name', 'qualified_identifier', 'scoped_identifier']:
                        text = subchild.text.decode('utf-8')
                        
                        # If qualified (MyClass::method or Outer::Inner::method), extract just the method name
                        if '::' in text:
                            return text.split('::')[-1]  # "MyClass::method" -> "method"
                        
                        return text
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

        param_type_types: list[str] = [
            'type_qualifier',
            'primitive_type',
            'type_identifier',
            'qualified_identifier'
        ]

        declarator_dict: dict[str, str] = {
            'pointer_declarator': '*',
            'reference_declarator': '&'
        }

        for child in param_node.named_children:
            child_type = child.type

            # Collect type information
            if child_type in param_type_types:
                param_type_parts.append(child.text.decode('utf-8'))

            # Extract name from identifier
            elif child_type == 'identifier':
                param_name = child.text.decode('utf-8')

            # Handle pointer and reference declarators
            elif child_type in declarator_dict:
                for subchild in child.named_children:
                    if subchild.type == 'identifier':
                        param_name = subchild.text.decode('utf-8')
                param_type_parts.append(declarator_dict[child_type])

        # Construct full type string
        param_type = ' '.join(param_type_parts) if param_type_parts else 'void'
        return param_name, param_type

    def _extract_return_type(self, func_node: Node) -> str | None:
        """Extract return type - None for constructors/destructors."""
        return_types: list[str] = [
            'primitive_type',
            'type_identifier'
        ]

        for child in func_node.named_children:
            if child.type in return_types :
                return child.text.decode('utf-8')

        # No return type found (constructor/destructor)
        return None

    def parse_calls(self, top_call_node: Node, file_id: str, cll_id: int,
                    current_class_name: str, class_base_classes: list, class_id: int,
                    fnc_id: int, func_name: str, func_params: dict) -> list[pd.DataFrame]:
        """Parse C++ function/method calls."""
        calls = []

        call_name = self._extract_call_name(top_call_node)

        if call_name:
            import json
            new_row = pd.DataFrame([{
                'file_id': file_id,
                'cll_id': cll_id,
                'name': call_name,
                'class': current_class_name,
                'class_base_classes': class_base_classes,
                'class_id': class_id,
                'func_id': fnc_id,
                'func_name': func_name,
                'func_params': json.dumps(func_params) if func_params else '{}'
            }])
            calls.append(new_row)

        return calls


    def _extract_call_name(self, call_node: Node) -> str | None:
        """Extract the call name from a call_expression node."""

        if call_node.type == 'new_expression':
            for child in call_node.named_children:
                if child.type == 'type_identifier':
                    return child.text.decode('utf-8')
            return None

        # Find the function being called (first named child that's not argument_list)
        func_expr = None
        for child in call_node.named_children:
            if child.type != 'argument_list':
                func_expr = child
                break

        if not func_expr:
            return None

        funct_types: set[str] = {
            'identifier', # Simple call: func()
            'field_expression', # Method call: obj.method() or ptr->method() or this->method()
            'qualified_identifier', # Qualified call: Class::method() or namespace::func()
            'template_function' # Template call: func<int>()
        }
        # Handle different call types
        if func_expr.type in funct_types:

            return func_expr.text.decode('utf-8')

        return None

    def resolve_calls(self, imports: pd.DataFrame, classes: pd.DataFrame, functions: pd.DataFrame,
                      calls: pd.DataFrame) -> None:
        """
        Resolve C++ call names to their targets and add 'combinedName' column.

        Handles:
        - this->method() -> ClassName.method()
        - obj.method() / obj->method() -> resolve obj from func_params
        - Class::method() -> Class.method (normalize :: to .)
        - Simple calls -> keep as-is
        """
        import json

        def resolve_single_call(row):
            """Resolve a single call name to its target."""
            call_name = row['name']
            class_name = row.get('class', 'Global')
            func_params_str = row.get('func_params', '{}')

            # Parse func_params from JSON string
            try:
                func_params = json.loads(func_params_str) if func_params_str else {}
            except (json.JSONDecodeError, TypeError):
                func_params = {}

            # Case 1: this->method() or this.method()
            if call_name.startswith('this->') or call_name.startswith('this.'):
                separator = '->' if '->' in call_name else '.'
                method_name = call_name.split(separator, 1)[1]

                if class_name and class_name != 'Global':
                    return f"{class_name}.{method_name}"
                return method_name

            # Case 2: obj->method() or obj.method() - try to resolve obj from parameters
            if '->' in call_name or ('.' in call_name and '::' not in call_name):
                separator = '->' if '->' in call_name else '.'
                parts = call_name.split(separator, 1)

                if len(parts) == 2:
                    obj_name, method_name = parts

                    # Try to resolve obj_name from function parameters
                    if obj_name in func_params:
                        param_type = func_params[obj_name]

                        # Clean up type: remove const, *, &, whitespace
                        param_type = (param_type
                                      .replace('const', '')
                                      .replace('*', '')
                                      .replace('&', '')
                                      .strip())

                        # Normalize :: to . in type
                        param_type = param_type.replace('::', '.')

                        return f"{param_type}.{method_name}"

                    # If not in params, keep original (local variable)
                    return call_name

            # Case 3: Class::method() or namespace::func() - normalize :: to .
            if '::' in call_name:
                return call_name.replace('::', '.')

            # Case 4: Template calls - strip template arguments
            # func<int>() -> func
            if '<' in call_name and '>' in call_name:
                # Extract base name before template
                base_name = call_name.split('<')[0]
                return base_name

            # Case 5: Simple function call or constructor - keep as-is
            return call_name

        # Apply resolution to all calls
        calls['combinedName'] = calls.apply(resolve_single_call, axis=1)