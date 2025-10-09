from package.adapters import LanguageAstAdapter
from package.adapters import NodeType

from tree_sitter import Node
import pandas as pd

from package.adapters.LanguageAstAdapterRegistry import LanguageAstAdapterRegistry

@LanguageAstAdapterRegistry.register('python')
class PythonAdapter(LanguageAstAdapter):
    def __init__(self):
        super().__init__(language="python", mapper={
        "import_statement": NodeType.IMPORT,
        "import_from_statement": NodeType.IMPORT,
        "class_definition": NodeType.CLASS,
        "function_definition": NodeType.FUNCTION,
        "decorated_definition": NodeType.FUNCTION,
        "async_function_definition": NodeType.FUNCTION,
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

    def should_skip_function_node(self, node: Node) -> bool:
        """Skip function_definition nodes inside decorated_definition (we process the wrapper instead)."""
        return (node.type in ['function_definition', 'async_function_definition'] and
                node.parent and
                node.parent.type == 'decorated_definition')

    def parse_functions(self, top_function_node: Node, current_class_name: str, class_base_classes: list,
                        file_id: str, fnc_id: int, class_id: int) -> list[pd.DataFrame]:
        # Unwrap decorated functions to get actual function node
        func_node = top_function_node
        if top_function_node.type == 'decorated_definition':
            for child in top_function_node.named_children:
                if child.type in ['function_definition', 'async_function_definition']:
                    func_node = child
                    break

        # Extract all components
        name = self._get_child_text(func_node, 'identifier') or '<anonymous>'
        params = self._parse_params(func_node)
        return_type = self._get_child_text(func_node, 'type')
        docstring = self._get_docstring(func_node)

        import json
        return [pd.DataFrame([{
            'file_id': file_id,
            'fnc_id': fnc_id,
            'name': name,
            'class': current_class_name,
            'class_base_classes': class_base_classes,
            'params': json.dumps(params),
            'docstring': docstring,
            'function_code': top_function_node.text.decode('utf-8'),
            'class_id': class_id,
            'return_type': return_type
        }])]

    def _get_child_text(self, node: Node, child_type: str) -> str | None:
        """Get decoded text of first child matching type."""
        for child in node.named_children:
            if child.type == child_type:
                return child.text.decode('utf-8')
        return None

    def _parse_params(self, func_node: Node) -> dict[str, str]:
        """Extract parameters, skipping positional-only, keyword-only, *args, **kwargs."""
        params_node = next((c for c in func_node.named_children if c.type == 'parameters'), None)
        if not params_node:
            return {}

        params = {}
        has_pos_sep = any(c.type == 'positional_separator' for c in params_node.named_children)
        before_pos_sep, after_kw_sep = has_pos_sep, False

        for child in params_node.named_children:
            if child.type == 'positional_separator':
                before_pos_sep = False
            elif child.type == 'keyword_separator':
                after_kw_sep = True
            elif not (before_pos_sep or after_kw_sep or
                      child.type in ['list_splat_pattern', 'dictionary_splat_pattern']):
                name, typ = self._get_param_info(child)
                if name:
                    params[name] = typ
        return params

    def _get_param_info(self, node: Node) -> tuple[str | None, str]:
        """Extract parameter name and type."""
        if node.type == 'identifier':
            return node.text.decode('utf-8'), 'Any'
        if node.type in ['typed_parameter', 'typed_default_parameter']:
            name = self._get_child_text(node, 'identifier')
            typ = self._get_child_text(node, 'type') or 'Any'
            return name, typ
        return None, 'Any'

    def _get_docstring(self, func_node: Node) -> str | None:
        """Extract docstring from first statement."""
        block = next((c for c in func_node.named_children if c.type == 'block'), None)
        if not block or not block.named_children:
            return None

        first = block.named_children[0]
        if first.type == 'expression_statement':
            string = next((c for c in first.named_children if c.type == 'string'), None)
            if string:
                text = string.text.decode('utf-8')
                # Check for string prefix BEFORE stripping quotes
                if text and text[0] in 'frFR':
                    text = text[1:]
                # Now strip quotes
                return text.strip('"""').strip("'''").strip('"').strip("'")
        return None
