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

        local_vars = self.extract_local_variables(actual_function_node)

        import json
        return [pd.DataFrame([{
            'file_id': file_id,
            'fnc_id': fnc_id,
            'name': name,
            'class': current_class_name,
            'class_base_classes': class_base_classes,
            'params': json.dumps(params),
            'local_vars': json.dumps(local_vars), 
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

                    elif subchild.type == 'template_function':
                        for template_child in subchild.named_children:
                            if template_child.type == 'identifier':
                                return template_child.text.decode('utf-8')

            # ADD THIS BLOCK - Handle pointer_declarator for functions returning pointers
            elif child.type == 'pointer_declarator':
                for subchild in child.named_children:
                    if subchild.type == 'function_declarator':
                        for inner_child in subchild.named_children:
                            if inner_child.type == 'identifier':
                                return inner_child.text.decode('utf-8')

            if func_node.type == 'field_declaration':
                for child in func_node.named_children:
                    if child.type == 'function_declarator':
                        for subchild in child.named_children:
                            if subchild.type == 'identifier':
                                return subchild.text.decode('utf-8')

        return '<lambda>' if 'lambda' in func_node.text.decode('utf-8') else '<inline_function>'


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
    
        if 'local_vars' in functions.columns and not functions.empty:
            try:
                
                func_lookup = functions.set_index('fnc_id')['local_vars'].to_dict()
                
                
                
                calls['local_vars'] = calls['func_id'].map(func_lookup).fillna('{}')
                
                
                
            except Exception as e:
                
                import traceback
                traceback.print_exc()
                calls['local_vars'] = '{}'
        else:
            # No local_vars column or empty functions df
            calls['local_vars'] = '{}'


        def resolve_single_call(row):
            """Resolve a single call name to its target."""
            call_name = row['name']
            class_name = row.get('class', 'Global')
            func_params_str = row.get('func_params', '{}')
            local_vars_str = row.get('local_vars', '{}')

            # Parse func_params from JSON string
            try:
                func_params = json.loads(func_params_str) if func_params_str else {}
                local_vars = json.loads(local_vars_str) if local_vars_str else {}
                all_vars = {**func_params, **local_vars}
                
               
            except (json.JSONDecodeError, TypeError) as e:
             
                func_params = {}
                local_vars = {}
                all_vars = {}

            # Case 1: this->method() or this.method()
            if call_name.startswith('this->') or call_name.startswith('this.'):
                separator = '->' if '->' in call_name else '.'
                method_name = call_name.split(separator, 1)[1]

                if class_name and class_name != 'Global':
                    return f"{class_name}.{method_name}"
                return method_name

            # Case 2: obj->method() or obj.method() - resolve from ALL vars (params + locals)
            if '->' in call_name or ('.' in call_name and '::' not in call_name):
                separator = '->' if '->' in call_name else '.'
                parts = call_name.split(separator, 1)

                if len(parts) == 2:
                    obj_name, method_name = parts
                    
                    

                    
                    if obj_name in all_vars:
                        var_type = all_vars[obj_name]

                        if var_type.strip() == 'auto':
                            return call_name
                        
                        if 'unique_ptr<' in var_type or 'shared_ptr<' in var_type or 'weak_ptr<' in var_type:
                            # Extract what's between < and >
                            start = var_type.find('<')
                            end = var_type.rfind('>')
                            if start != -1 and end != -1 and end > start:
                                var_type = var_type[start + 1:end]

                        # Clean up type: remove const, *, &, whitespace
                        var_type = (var_type
                                    .replace('const', '')
                                    .replace('*', '')
                                    .replace('&', '')
                                    .strip())

                        # Normalize :: to . in type
                        var_type = var_type.replace('::', '.')

                        result = f"{var_type}.{method_name}"
                    
                        return result
                    

                    # If not in all_vars, keep original (unknown variable)
                    return call_name

            # Case 3: Class::method() or namespace::func() - normalize :: to .
            if '::' in call_name:
                return call_name.replace('::', '.')

            # Case 4: Template calls - strip template arguments
            if '<' in call_name and '>' in call_name:
                base_name = call_name.split('<')[0]
                return base_name

            # Case 5: Simple function call or constructor - keep as-is
            return call_name

        # Apply resolution to all calls
        calls['combinedName'] = calls.apply(resolve_single_call, axis=1)

    def create_import_edges(self, import_df: pd.DataFrame, cg_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates import nodes and edges for C++.
        C++ includes don't have 'from' module, so we group by (name, as_name) only.
        """
        if import_df.empty:
            imports = pd.DataFrame(columns=['ID', 'import_name', 'import_from', 'import_as_name', 'import_file_ids'])
            imp_edges = pd.DataFrame(columns=['source', 'target'])
            return imports, imp_edges
        
        # C++: Group by name and as_name (from is always None for #include)
        # Note: in C++, name = as_name for most cases
        imports_grouped = (
            import_df.groupby(['name', 'as_name'], dropna=False)['file_id']
            .apply(lambda x: list(set(x)))
            .reset_index()
        )

        # Add unique IDs for each import
        imports_grouped.insert(0, 'import_id', range(1, len(imports_grouped) + 1))

        # Format import nodes
        imports = imports_grouped.rename(columns={
            'name': 'import_name',
            'as_name': 'import_as_name',
            'file_id': 'import_file_ids'
        })
        imports['import_from'] = None  # C++ doesn't have 'from' concept

        # Create edges: Import -> Functions (via file_id)
        imp_edges = imports[['import_id', 'import_file_ids']].explode('import_file_ids')
        imp_edges = imp_edges.rename(columns={'import_file_ids': 'file_id'})

        # Connect imports to functions in the same file
        imp_edges = imp_edges.merge(
            cg_nodes[['func_id', 'file_id']], 
            on='file_id', 
            how='left'
        )
        
        # Keep only valid edges and rename columns
        imp_edges = imp_edges[['import_id', 'func_id']].dropna().reset_index(drop=True)
        imp_edges = imp_edges.rename(columns={'import_id': 'source', 'func_id': 'target'})

        return imports, imp_edges
    
    def extract_local_variables(self, func_node: Node) -> dict[str, str]:
        """
        Extract local variable declarations from function body.
        Walks entire function body tree, later definitions override earlier ones (simple shadowing).

        LIMITATION: Does not handle scope shadowing. If the same variable name
        is declared in nested scopes, only the last declaration is kept.

        Returns dict of {var_name: var_type}
        """
        local_vars = {}
        
        # Find function body (compound_statement)
        body = None
        for child in func_node.named_children:
            
            if child.type == 'compound_statement':
                body = child
                break
        
        if not body:
            
            return local_vars
        
        # Walk entire tree to find all variable declarations
        def walk_for_declarations(node: Node, depth: int = 0):
            # Limit recursion depth to avoid infinite loops
            if depth > 20:
                return
            
            # Found a declaration
            if node.type == 'declaration':
                 
                var_name, var_type = self._extract_var_from_declaration(node)
                
                if var_name and var_type:
                    # Later declarations override (simple shadowing handling)
                    local_vars[var_name] = var_type
            
            # Recurse into children
            for child in node.named_children:
                walk_for_declarations(child, depth + 1)
        
        walk_for_declarations(body)
        
        return local_vars


    def _extract_var_from_declaration(self, decl_node: Node) -> tuple[str | None, str]:
        """
        Extract variable name and type from a declaration node.
        Handles: Type var; Type* var; Type var = new Type(); etc.
        Returns: (var_name, var_type)
        """
        var_type_parts = []
        var_name = None
        
        for child in decl_node.named_children:
            # Collect type information
            if child.type in ['type_identifier', 'primitive_type', 'qualified_identifier', 'type_qualifier']:
                var_type_parts.append(child.text.decode('utf-8'))
            
            elif child.type == 'placeholder_type_specifier':
                var_type_parts.append('auto')

            elif child.type == 'template_type':
                # Check if it's a smart pointer template
                template_text = child.text.decode('utf-8')
                if 'unique_ptr' in template_text or 'shared_ptr' in template_text or 'weak_ptr' in template_text:
                    # Extract just the inner type
                    inner_type = self._extract_template_inner_type(child)
                    if inner_type:
                        var_type_parts.append(inner_type)
                    else:
                        var_type_parts.append(template_text)
                else:
                    # Not a smart pointer, keep full template type
                    var_type_parts.append(template_text)
            
            # Handle simple identifier (stack allocation: Type obj;)
            elif child.type == 'identifier':
                var_name = child.text.decode('utf-8')
            
            elif child.type == 'identifier' and not var_name:
            # Skip if this identifier is part of the type (shouldn't happen, but safe)
                var_name = child.text.decode('utf-8')

            # Handle declarators (where the variable name lives)
            elif child.type in ['init_declarator', 'pointer_declarator', 'reference_declarator']:
                name, extra_type = self._extract_from_declarator(child)
                if name:
                    var_name = name
                if extra_type:
                    var_type_parts.extend(extra_type)
        
        var_type = ' '.join(var_type_parts) if var_type_parts else 'auto'
        return var_name, var_type


    def _extract_from_declarator(self, declarator_node: Node) -> tuple[str | None, list[str]]:
        """
        Extract variable name and any additional type info (like * or &) from declarator.
        Returns: (var_name, [extra_type_parts])
        """
        var_name = None
        extra_type = []
        
        for child in declarator_node.named_children:
            # Direct identifier
            if child.type == 'identifier':
                var_name = child.text.decode('utf-8')
            
            # Pointer: Type* var
            elif child.type == 'pointer_declarator':
                extra_type.append('*')
                # Recurse to find the actual identifier
                nested_name, nested_type = self._extract_from_declarator(child)
                if nested_name:
                    var_name = nested_name
                extra_type.extend(nested_type)
            
            # Reference: Type& var
            elif child.type == 'reference_declarator':
                extra_type.append('&')
                nested_name, nested_type = self._extract_from_declarator(child)
                if nested_name:
                    var_name = nested_name
                extra_type.extend(nested_type)
            
            elif child.type == 'new_expression':
                # Skip: The type was already collected from the declaration's type_identifier
                # We only need the * from the pointer_declarator
                pass
            
            # Template type: std::unique_ptr<Target>
            elif child.type == 'template_type':
                pass
        
        return var_name, extra_type


    def _extract_type_from_new_expression(self, new_node: Node) -> str | None:
        """
        Extract type from new expression: new MyClass() -> MyClass
        """
        for child in new_node.named_children:
            if child.type in ['type_identifier', 'qualified_identifier']:
                return child.text.decode('utf-8')
            elif child.type == 'template_type':
                # new MyClass<T>() -> extract MyClass
                for subchild in child.named_children:
                    if subchild.type in ['type_identifier', 'qualified_identifier']:
                        return subchild.text.decode('utf-8')
        return None


    def _extract_template_inner_type(self, template_node: Node) -> str | None:
        """
        Extract inner type from template: std::unique_ptr<Target> -> Target
        """
        for child in template_node.named_children:
            if child.type == 'template_argument_list':
                for arg in child.named_children:
                    if arg.type in ['type_identifier', 'qualified_identifier']:
                        return arg.text.decode('utf-8')
                    elif arg.type == 'template_type':
                        # Nested template: std::vector<std::string>
                        return arg.text.decode('utf-8')
        return None