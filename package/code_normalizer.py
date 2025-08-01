import ast
import textwrap


class CodeNormalizer():

    def __init__(self):
        pass


    def remove_docstring(self, code: str) -> str:
        """
        Removes the docstring from a Python function (if present).
        
        Args:
            code (str): A string containing a Python function.
        
        Returns:
            str: The code with the docstring removed.
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Modify the AST: remove docstrings
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                    if (node.body and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)):
                        # Remove the first statement if it's a docstring
                        node.body.pop(0)

            # Convert AST back to code
            new_code = ast.unparse(tree)
            return textwrap.dedent(new_code).strip()

        except Exception as e:
            raise ValueError(f"Could not process the given code: {e}")


    def normalize(self, code: str) -> str:

        code = self.remove_docstring(code)

        tree = ast.parse(code)
        norm_tree = ASTCodeNormalizer().visit(tree)
        norm_code = ast.unparse(norm_tree)
    
        return textwrap.dedent(norm_code).strip()






class ASTCodeNormalizer(ast.NodeTransformer):

    def __init__(self):
        self.global_var_counter = 0
        self.func_counter = 0
        self.class_counter = 0
        self.scopes = []
        self.self_attrs = {}
        self.self_methods = {}  # <--- új: self.metodusok


    def _enter_scope(self):
        self.scopes.append({})

    def _exit_scope(self):
        self.scopes.pop()

    def _normalize_name(self, name, prefix):
        current_scope = self.scopes[-1]
        if name not in current_scope:
            norm_name = f"{prefix}_{len(current_scope)}"
            current_scope[name] = norm_name
        return current_scope[name]

    def _lookup_name(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return name
    
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            attr = node.func
            if isinstance(attr.value, ast.Name) and attr.value.id == "self":
                if attr.attr not in self.self_methods:
                    self.self_methods[attr.attr] = f"CLASS_FUNC_{len(self.self_methods)}"
                attr.attr = self.self_methods[attr.attr]
        return node


    def visit_ClassDef(self, node):
        class_name = f"CLASS_{self.class_counter}"
        self.class_counter += 1
        node.name = class_name
        self._enter_scope()
        self.generic_visit(node)
        self._exit_scope()
        return node

    def visit_FunctionDef(self, node):
        func_name = f"FUNC_{self.func_counter}"
        self.func_counter += 1
        node.name = func_name

        self._enter_scope()

        for i, arg in enumerate(node.args.args):
            if arg.arg == "self":
                self.scopes[-1][arg.arg] = "self"  # keep "self"
            else:
                arg.arg = self._normalize_name(arg.arg, "ARG")

        self.generic_visit(node)
        self._exit_scope()
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            if node.id != "self":
                node.id = self._normalize_name(node.id, "VAR")
        else:
            node.id = self._lookup_name(node.id)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            # ha már metódusként normalizáltuk, ne írjuk át
            if node.attr in self.self_methods.values():
                return node
            if node.attr not in self.self_attrs:
                self.self_attrs[node.attr] = f"CLASS_ATTR_{len(self.self_attrs)}"
            node.attr = self.self_attrs[node.attr]
        return node


    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        # elif isinstance(node.value, (int, float)):
        #     return ast.copy_location(ast.Constant(value=0), node)
        return node

    def visit_Str(self, node):  # Python <3.8
        return ast.copy_location(ast.Str(s="STR"), node)

    def visit_Num(self, node):  # Python <3.8
        return ast.copy_location(ast.Num(n=0), node)