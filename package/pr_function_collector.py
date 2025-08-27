import ast
import hashlib
from typing import Dict, List, Optional
from github import Github
from difflib import unified_diff

class PRFunctionCollector(ast.NodeVisitor):
    def __init__(self, lines: List[str]):
        self.lines = lines
        self.stack = []
        self.functions = {}  # name -> (start, end)

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        full_name = '.'.join(self.stack + [node.name])
        start = node.lineno - 1
        end = self._find_end_lineno(node)
        self.functions[full_name] = (start, end)
        self.generic_visit(node)

    def _find_end_lineno(self, node):
        # Try to find the maximum lineno in this function
        max_lineno = node.lineno
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                max_lineno = max(max_lineno, child.lineno)
        return max_lineno

def hash_function_body(lines: List[str]) -> str:
    return hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest()

def extract_function_bodies(code: str) -> Dict[str, str]:
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    collector = PRFunctionCollector(lines)
    collector.visit(tree)

    func_bodies = {}
    for name, (start, end) in collector.functions.items():
        body_lines = lines[start:end]
        func_bodies[name] = hash_function_body(body_lines)

    return func_bodies

def get_changed_functions(base_code: str, head_code: str) -> List[str]:
    base_funcs = extract_function_bodies(base_code)
    head_funcs = extract_function_bodies(head_code)

    changed = []

    all_keys = set(base_funcs.keys()) | set(head_funcs.keys())

    for name in all_keys:
        base_hash = base_funcs.get(name)
        head_hash = head_funcs.get(name)
        if base_hash != head_hash:
            changed.append(name)

    return sorted(changed)

def get_changed_python_files(pr):
    return [
        f for f in pr.get_files()
        if f.filename.endswith(".py") and f.status in {"modified", "renamed"}
    ]

def get_file_content(repo, path, ref) -> str:
    try:
        return repo.get_contents(path, ref=ref).decoded_content.decode()
    except:
        return ""  # May happen for deleted/renamed files

def extract_changed_functions_from_pr(pr) -> List[str]:
    repo = pr.base.repo
    base_sha = pr.base.sha
    head_sha = pr.head.sha

    changed_funcs = []

    for f in get_changed_python_files(pr):
        file_path = f.filename

        base_code = get_file_content(repo, file_path, base_sha)
        head_code = get_file_content(repo, file_path, head_sha)

        if not base_code or not head_code:
            continue

        funcs = get_changed_functions(base_code, head_code)
        qualified_funcs = [f"{file_path}:{fn}" for fn in funcs]
        changed_funcs.extend(qualified_funcs)

    return sorted(changed_funcs)