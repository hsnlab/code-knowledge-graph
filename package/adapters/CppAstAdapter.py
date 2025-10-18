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
        "new_expression": NodeType.CALL,
        "call_expression": NodeType.CALL,
    })


    def parse_import(self, top_import_node: Node, file_id: str, imp_id: int) -> list[pd.DataFrame]:
        """Parse C++ #include directives."""
        imports: list = list()
        from_module = None
        as_name = None

        for child in top_import_node.named_children:
            name = None

            if child.type == 'system_lib_string':
                name = child.text.decode('utf-8').strip('<>')
            elif child.type == 'string_literal':
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
        """Parse C++ class/struct definitions."""
        classes = []
        name = None
        base_classes = []

        for child in top_class_node.named_children:
            if child.type == 'type_identifier':
                name = child.text.decode('utf-8')

            elif child.type == 'base_class_clause':
                for base in child.named_children:
                    if base.type in ['type_identifier', 'qualified_identifier']:
                        base_classes.append(base.text.decode('utf-8'))
                    elif base.type == 'base_class_specifier':
                        for specifier_child in base.named_children:
                            if specifier_child.type in ['type_identifier', 'qualified_identifier']:
                                base_classes.append(specifier_child.text.decode('utf-8'))

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
        """Skip function nodes inside template_declaration to avoid duplicates."""
        if node.parent and node.parent.type == 'template_declaration':
            return node.type in ['function_definition', 'template_declaration']
        return False
    

    def _extract_class_from_qualified_name(self, func_node: Node) -> str | None:
        """Extract class name from qualified function definition (e.g. MyClass::method)."""
        for child in func_node.named_children:
            if child.type == 'function_declarator':
                for subchild in child.named_children:
                    if subchild.type in ['qualified_identifier', 'scoped_identifier']:
                        text = subchild.text.decode('utf-8')
                        
                        if '::' in text:
                            if '<' in text:
                                text = text.split('<')[0] + text.split('>')[-1]
                            
                            parts = text.split('::')
                            return '::'.join(parts[:-1]) if len(parts) > 1 else None
        
            elif child.type == 'pointer_declarator':
                for subchild in child.named_children:
                    if subchild.type == 'function_declarator':
                        for inner_child in subchild.named_children:
                            if inner_child.type in ['qualified_identifier', 'scoped_identifier']:
                                text = inner_child.text.decode('utf-8')
                                
                                if '::' in text:
                                    if '<' in text:
                                        text = text.split('<')[0] + text.split('>')[-1]
                                    
                                    parts = text.split('::')
                                    return '::'.join(parts[:-1]) if len(parts) > 1 else None

        return None


    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list,
                    file_id: str, fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        """Parse C++ function/method definitions."""
        functions = []

        if top_function_node.type == 'friend_declaration':
            has_function_def = any(
                child.type == 'function_definition'
                for child in top_function_node.named_children
            )
            if not has_function_def:
                return []

        if top_function_node.type == 'template_declaration':
            has_alias = any(
                child.type == 'alias_declaration'
                for child in top_function_node.named_children
            )
            if has_alias:
                return []
            
            for child in top_function_node.named_children:
                if child.type == 'declaration':
                    decl_text = child.text.decode('utf-8').strip()
                    if decl_text.endswith(';') and '{' not in decl_text:
                        return []

        if top_function_node.type == 'field_declaration':
            has_function_declarator = any(
                child.type == 'function_declarator'
                for child in top_function_node.named_children
            )
            if not has_function_declarator:
                return []

        actual_function_node = top_function_node
        if top_function_node.type == 'template_declaration':
            for child in top_function_node.named_children:
                if child.type == 'function_definition':
                    actual_function_node = child
                    break
                elif child.type in ['class_specifier', 'struct_specifier']:
                    return []

        if actual_function_node.type == 'friend_declaration':
            for child in actual_function_node.named_children:
                if child.type == 'declaration':
                    actual_function_node = child
                    break
                elif child.type == 'function_definition':
                    actual_function_node = child
                    break

        name = self._extract_function_name(actual_function_node, current_class_name)
        
        extracted_class_name = self._extract_class_from_qualified_name(actual_function_node)
        if extracted_class_name:
            current_class_name = extracted_class_name
            class_id = None
        
        if not current_class_name or current_class_name == 'None':
            current_class_name = 'Global'

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
        """Extract function name from AST node."""
        
        for child in func_node.named_children:
            if child.type == 'function_declarator':
                for subchild in child.named_children:
                    if subchild.type in ['identifier', 'field_identifier', 'destructor_name', 
                                        'operator_name', 'qualified_identifier', 'scoped_identifier']:
                        text = subchild.text.decode('utf-8')
                        
                        if '::' in text:
                            return text.split('::')[-1]
                        
                        return text

                    elif subchild.type == 'template_function':
                        for template_child in subchild.named_children:
                            if template_child.type == 'identifier':
                                return template_child.text.decode('utf-8')

            elif child.type == 'pointer_declarator':
                for subchild in child.named_children:
                    if subchild.type == 'pointer_declarator':
                        for nested_subchild in subchild.named_children:
                            if nested_subchild.type == 'function_declarator':
                                for inner_child in nested_subchild.named_children:
                                    if inner_child.type in ['identifier', 'field_identifier']:
                                        return inner_child.text.decode('utf-8')
                    
                    elif subchild.type == 'function_declarator':
                        for inner_child in subchild.named_children:
                            if inner_child.type in ['identifier', 'field_identifier']:
                                return inner_child.text.decode('utf-8')
                            
                            elif inner_child.type == 'operator_name':
                                return inner_child.text.decode('utf-8')
                            
                            elif inner_child.type in ['qualified_identifier', 'scoped_identifier']:
                                for qual_child in inner_child.named_children:
                                    if qual_child.type == 'identifier':
                                        return qual_child.text.decode('utf-8')
                                
                                text = inner_child.text.decode('utf-8')
                                if '::' in text:
                                    return text.split('::')[-1]
                                return text

            elif child.type == 'reference_declarator':
                for subchild in child.named_children:
                    if subchild.type == 'function_declarator':
                        for inner_child in subchild.named_children:
                            if inner_child.type in ['identifier', 'field_identifier']:
                                return inner_child.text.decode('utf-8')
                            
                            elif inner_child.type == 'operator_name':
                                return inner_child.text.decode('utf-8')
                            
                            elif inner_child.type in ['qualified_identifier', 'scoped_identifier']:
                                for qual_child in inner_child.named_children:
                                    if qual_child.type == 'identifier':
                                        return qual_child.text.decode('utf-8')
                                
                                text = inner_child.text.decode('utf-8')
                                if '::' in text:
                                    return text.split('::')[-1]
                                return text

            elif child.type == 'operator_cast':
                operator_text = child.text.decode('utf-8')
                operator_name = operator_text.split('(')[0].strip()
                return operator_name

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
        func_declarator = None
        
        for child in func_node.named_children:
            if child.type == 'function_declarator':
                func_declarator = child
                break
            elif child.type in ['pointer_declarator', 'reference_declarator']:
                for subchild in child.named_children:
                    if subchild.type == 'pointer_declarator':
                        for nested_subchild in subchild.named_children:
                            if nested_subchild.type == 'function_declarator':
                                func_declarator = nested_subchild
                                break
                    elif subchild.type == 'function_declarator':
                        func_declarator = subchild
                        break
                
                if func_declarator:
                    break
            elif child.type == 'operator_cast':
                return {}

        if not func_declarator:
            return params

        param_list = None
        for child in func_declarator.named_children:
            if child.type == 'parameter_list':
                param_list = child
                break

        if not param_list:
            return params

        for param_child in param_list.named_children:
            if param_child.type in ['parameter_declaration', 'optional_parameter_declaration']:
                param_name, param_type = self._extract_param_info(param_child)
                if param_name:
                    params[param_name] = param_type

        return params


    def _extract_param_info(self, param_node: Node) -> tuple[str | None, str]:
        """Extract parameter name and type."""
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

            if child_type in param_type_types:
                param_type_parts.append(child.text.decode('utf-8'))
            elif child_type == 'identifier':
                param_name = child.text.decode('utf-8')
            elif child_type in declarator_dict:
                for subchild in child.named_children:
                    if subchild.type == 'identifier':
                        param_name = subchild.text.decode('utf-8')
                param_type_parts.append(declarator_dict[child_type])

        param_type = ' '.join(param_type_parts) if param_type_parts else 'void'
        return param_name, param_type


    def _extract_return_type(self, func_node: Node) -> str | None:
        """Extract return type."""
        return_types: list[str] = ['primitive_type', 'type_identifier']

        for child in func_node.named_children:
            if child.type in return_types:
                return child.text.decode('utf-8')

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
        """Extract the function name being called."""
        if call_node.type == 'new_expression':
            for child in call_node.named_children:
                if child.type == 'type_identifier':
                    return child.text.decode('utf-8')
            return None

        func_expr = None
        for child in call_node.named_children:
            if child.type != 'argument_list':
                func_expr = child
                break

        if not func_expr:
            return None

        funct_types: set[str] = {
            'identifier',
            'field_expression',
            'qualified_identifier',
            'template_function'
        }
        
        if func_expr.type in funct_types:
            return func_expr.text.decode('utf-8')

        return None


    def resolve_calls(self, imports: pd.DataFrame, classes: pd.DataFrame, functions: pd.DataFrame,
                      calls: pd.DataFrame) -> None:
        """Resolve C++ call names to their targets."""
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
            calls['local_vars'] = '{}'

        def resolve_single_call(row):
            call_name = row['name']
            class_name = row.get('class', 'Global')
            func_params_str = row.get('func_params', '{}')
            local_vars_str = row.get('local_vars', '{}')

            try:
                func_params = json.loads(func_params_str) if func_params_str else {}
                local_vars = json.loads(local_vars_str) if local_vars_str else {}
                all_vars = {**func_params, **local_vars}
            except (json.JSONDecodeError, TypeError):
                func_params = {}
                local_vars = {}
                all_vars = {}

            if call_name.startswith('this->') or call_name.startswith('this.'):
                separator = '->' if '->' in call_name else '.'
                method_name = call_name.split(separator, 1)[1]

                if class_name and class_name != 'Global':
                    return f"{class_name}.{method_name}"
                return method_name

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
                            start = var_type.find('<')
                            end = var_type.rfind('>')
                            if start != -1 and end != -1 and end > start:
                                var_type = var_type[start + 1:end]

                        var_type = (var_type
                                    .replace('const', '')
                                    .replace('*', '')
                                    .replace('&', '')
                                    .strip())

                        var_type = var_type.replace('::', '.')
                        result = f"{var_type}.{method_name}"
                        return result

                    return call_name

            if '::' in call_name:
                return call_name.replace('::', '.')

            if '<' in call_name and '>' in call_name:
                base_name = call_name.split('<')[0]
                return base_name

            if class_name and class_name != 'Global' and '::' not in call_name and '->' not in call_name and '.' not in call_name:
                potential_method = f"{class_name}.{call_name}"
                
                if not functions.empty and 'combinedName' in functions.columns:
                    if potential_method in functions['combinedName'].values:
                        return potential_method
                
                return call_name

            return call_name

        calls['combinedName'] = calls.apply(resolve_single_call, axis=1)


    def create_import_edges(self, import_df: pd.DataFrame, cg_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create import nodes and edges."""
        if import_df.empty:
            imports = pd.DataFrame(columns=['ID', 'import_name', 'import_from', 'import_as_name', 'import_file_ids'])
            imp_edges = pd.DataFrame(columns=['source', 'target'])
            return imports, imp_edges
        
        imports_grouped = (
            import_df.groupby(['name', 'as_name'], dropna=False)['file_id']
            .apply(lambda x: list(set(x)))
            .reset_index()
        )

        imports_grouped.insert(0, 'import_id', range(1, len(imports_grouped) + 1))

        imports = imports_grouped.rename(columns={
            'name': 'import_name',
            'as_name': 'import_as_name',
            'file_id': 'import_file_ids'
        })
        imports['import_from'] = None

        imp_edges = imports[['import_id', 'import_file_ids']].explode('import_file_ids')
        imp_edges = imp_edges.rename(columns={'import_file_ids': 'file_id'})

        imp_edges = imp_edges.merge(
            cg_nodes[['func_id', 'file_id']], 
            on='file_id', 
            how='left'
        )
        
        imp_edges = imp_edges[['import_id', 'func_id']].dropna().reset_index(drop=True)
        imp_edges = imp_edges.rename(columns={'import_id': 'source', 'func_id': 'target'})

        return imports, imp_edges
    

    def extract_local_variables(self, func_node: Node) -> dict[str, str]:
        """Extract local variable declarations from function body."""
        local_vars = {}
        
        body = None
        for child in func_node.named_children:
            if child.type == 'compound_statement':
                body = child
                break
        
        if not body:
            return local_vars
        
        def walk_for_declarations(node: Node, depth: int = 0):
            if depth > 20:
                return
            
            if node.type == 'declaration':
                var_name, var_type = self._extract_var_from_declaration(node)
                
                if var_name and var_type:
                    local_vars[var_name] = var_type
            
            for child in node.named_children:
                walk_for_declarations(child, depth + 1)
        
        walk_for_declarations(body)
        
        return local_vars


    def _extract_var_from_declaration(self, decl_node: Node) -> tuple[str | None, str]:
        """Extract variable name and type from declaration."""
        var_type_parts = []
        var_name = None
        
        for child in decl_node.named_children:
            if child.type in ['type_identifier', 'primitive_type', 'qualified_identifier', 'type_qualifier']:
                var_type_parts.append(child.text.decode('utf-8'))
            
            elif child.type == 'placeholder_type_specifier':
                var_type_parts.append('auto')

            elif child.type == 'template_type':
                template_text = child.text.decode('utf-8')
                if 'unique_ptr' in template_text or 'shared_ptr' in template_text or 'weak_ptr' in template_text:
                    inner_type = self._extract_template_inner_type(child)
                    if inner_type:
                        var_type_parts.append(inner_type)
                    else:
                        var_type_parts.append(template_text)
                else:
                    var_type_parts.append(template_text)
            
            elif child.type == 'identifier':
                var_name = child.text.decode('utf-8')

            elif child.type in ['init_declarator', 'pointer_declarator', 'reference_declarator']:
                name, extra_type = self._extract_from_declarator(child)
                if name:
                    var_name = name
                if extra_type:
                    var_type_parts.extend(extra_type)
        
        var_type = ' '.join(var_type_parts) if var_type_parts else 'auto'
        return var_name, var_type


    def _extract_from_declarator(self, declarator_node: Node) -> tuple[str | None, list[str]]:
        """Extract variable name and type modifiers from declarator."""
        var_name = None
        extra_type = []
        
        for child in declarator_node.named_children:
            if child.type == 'identifier':
                var_name = child.text.decode('utf-8')
            
            elif child.type == 'pointer_declarator':
                extra_type.append('*')
                nested_name, nested_type = self._extract_from_declarator(child)
                if nested_name:
                    var_name = nested_name
                extra_type.extend(nested_type)
            
            elif child.type == 'reference_declarator':
                extra_type.append('&')
                nested_name, nested_type = self._extract_from_declarator(child)
                if nested_name:
                    var_name = nested_name
                extra_type.extend(nested_type)
        
        return var_name, extra_type


    def _extract_type_from_new_expression(self, new_node: Node) -> str | None:
        """Extract type from new expression."""
        for child in new_node.named_children:
            if child.type in ['type_identifier', 'qualified_identifier']:
                return child.text.decode('utf-8')
            elif child.type == 'template_type':
                for subchild in child.named_children:
                    if subchild.type in ['type_identifier', 'qualified_identifier']:
                        return subchild.text.decode('utf-8')
        return None


    def _extract_template_inner_type(self, template_node: Node) -> str | None:
        """Extract inner type from template."""
        for child in template_node.named_children:
            if child.type == 'template_argument_list':
                for arg in child.named_children:
                    if arg.type in ['type_identifier', 'qualified_identifier']:
                        return arg.text.decode('utf-8')
                    elif arg.type == 'template_type':
                        return arg.text.decode('utf-8')
        return None