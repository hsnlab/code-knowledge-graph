"""
Data Flow Graph (DFG) builder for Python functions.

A DFG shows how data moves through a function via variables:
  - DEF node: where a variable is assigned / defined
  - USE node: where a variable's value is read
  - DEF->USE edge: the data defined at DEF flows into USE

Only Python is supported (Tree-sitter grammar).
"""

import pandas as pd
from collections import defaultdict
from tree_sitter_language_pack import get_parser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# These identifiers are always present and carry no interesting data flow
_SKIP_NAMES = frozenset({"True", "False", "None", "self", "cls", "__name__"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DFGBuilder:
    """
    Builds an intra-procedural Data Flow Graph for a single Python function.

    Usage
    -----
    builder = DFGBuilder()
    dfg_nodes, dfg_edges, dfg_function_edges = builder.build_dfg(
        function_code="def foo(x):\n    y = x + 1\n    return y",
        func_id=42,
        start_node_id=0,
    )
    """

    def __init__(self):
        self._parser = get_parser("python")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build_dfg(
        self,
        function_code: str,
        func_id: int,
        start_node_id: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Analyse *function_code* and return three DataFrames.

        Parameters
        ----------
        function_code  : source text of the whole function (def ... : body)
        func_id        : ID of the parent FUNCTION node in the knowledge graph
        start_node_id  : first dfg_id to use (avoids collisions when called
                         for many functions in a loop)

        Returns
        -------
        dfg_nodes : columns [dfg_id, func_id, name, node_type, line, code]
                    node_type is either 'DEF' or 'USE'
        dfg_edges : columns [source_id, target_id]
                    DEF -> USE data-flow edges (both IDs are dfg_ids)
        dfg_function_edges : columns [dfg_id, func_id]
                    links every DFG node back to its parent FUNCTION node
        """
        # Parse source into a Tree-sitter AST
        tree = self._parser.parse(function_code.encode("utf-8"))
        lines = function_code.splitlines()

        # Find the root function node so nested functions can be skipped
        root_func = None
        for child in tree.root_node.children:
            if child.type in ("function_definition", "async_function_definition"):
                root_func = child
                break

        # Collect raw (name, line_number, code_snippet, kind) tuples
        raw_defs: list[tuple[str, int, str]] = []   # variable definitions
        raw_uses: list[tuple[str, int, str]] = []   # variable uses

        self._walk(tree.root_node, function_code, lines, raw_defs, raw_uses,
                   root_func=root_func)

        # Ha semmi nem található, adjunk vissza üres DataFrame-eket helyes oszlopokkal
        if not raw_defs and not raw_uses:
            return (
                pd.DataFrame(columns=["dfg_id", "func_id", "name", "node_type", "line", "code"]),
                pd.DataFrame(columns=["source_id", "target_id"]),
                pd.DataFrame(columns=["dfg_id", "func_id"]),
            )

        # Build node rows
        node_rows: list[dict] = []
        nid = start_node_id

        # --- DEF nodes ---
        # Maps variable name -> list of dfg_ids of its DEF nodes
        def_map: dict[str, list[int]] = defaultdict(list)

        for name, line, code_snip in raw_defs:
            node_rows.append({
                "dfg_id":     nid,
                "func_id":    func_id,
                "name":       name,
                "node_type":  "DEF",
                "line":       line,
                "code":       code_snip,
            })
            def_map[name].append(nid)
            nid += 1

        # --- USE nodes ---
        # Maps variable name -> list of dfg_ids of its USE nodes
        use_map: dict[str, list[int]] = defaultdict(list)

        for name, line, code_snip in raw_uses:
            node_rows.append({
                "dfg_id":     nid,
                "func_id":    func_id,
                "name":       name,
                "node_type":  "USE",
                "line":       line,
                "code":       code_snip,
            })
            use_map[name].append(nid)
            nid += 1

        # Build DEF->USE edge rows
        # For every variable that has both a DEF and at least one USE,
        # connect each DEF to each USE (Cartesian product within one variable).
        edge_rows: list[dict] = []
        for name, def_ids in def_map.items():
            if name in use_map:
                for def_id in def_ids:
                    for use_id in use_map[name]:
                        edge_rows.append({
                            "source_id": def_id,
                            "target_id": use_id,
                        })

        # Build FUNCTION->DFG linking edge rows
        func_link_rows = [
            {"dfg_id": row["dfg_id"], "func_id": func_id}
            for row in node_rows
        ]

        dfg_nodes          = pd.DataFrame(node_rows)
        dfg_edges          = pd.DataFrame(edge_rows)
        dfg_function_edges = pd.DataFrame(func_link_rows)

        return dfg_nodes, dfg_edges, dfg_function_edges

    # ------------------------------------------------------------------
    # AST traversal – dispatch on node type
    # ------------------------------------------------------------------

    def _walk(self, node, src: str, lines: list[str], defs, uses, root_func=None):
        """
        Recursively walk the Tree-sitter AST.

        We handle specific node types explicitly so we can tell whether an
        identifier is being *defined* or *used*.  For everything else we just
        recurse into the children.
        """
        t = node.type

        # ---- variable assignment:  x = expr  -------------------------
        if t == "assignment":
            lhs = node.child_by_field_name("left")
            rhs = node.child_by_field_name("right")
            if lhs:
                self._extract_targets(lhs, src, lines, defs)
            if rhs:
                self._extract_uses(rhs, src, lines, uses)
            return  # children already handled

        # ---- augmented assignment:  x += expr  -----------------------
        # x is both used (old value) and re-defined (new value)
        if t == "augmented_assignment":
            lhs = node.child_by_field_name("left")
            rhs = node.child_by_field_name("right")
            if lhs and lhs.type == "identifier":
                name = _text(lhs, src)
                line, snip = _line_snip(lhs, lines)
                uses.append((name, line, snip))   # reads old value
                defs.append((name, line, snip))   # writes new value
            if rhs:
                self._extract_uses(rhs, src, lines, uses)
            return

        # ---- walrus operator:  (x := expr)  -------------------------
        if t == "named_expression":
            name_node = node.child_by_field_name("name")
            value     = node.child_by_field_name("value")
            if name_node:
                name = _text(name_node, src)
                line, snip = _line_snip(name_node, lines)
                defs.append((name, line, snip))
            if value:
                self._extract_uses(value, src, lines, uses)
            return

        # ---- for loop variable:  for x in iterable  -----------------
        if t == "for_statement":
            loop_var  = node.child_by_field_name("left")
            iterable  = node.child_by_field_name("right")
            body      = node.child_by_field_name("body")
            if loop_var:
                self._extract_targets(loop_var, src, lines, defs)
            if iterable:
                self._extract_uses(iterable, src, lines, uses)
            if body:
                self._walk(body, src, lines, defs, uses, root_func)
            return

        # ---- with statement:  with expr as x  -----------------------
        if t == "with_statement":
            for child in node.children:
                if child.type == "with_clause":
                    for item in child.children:
                        if item.type == "with_item":
                            val   = item.child_by_field_name("value")
                            alias = item.child_by_field_name("alias")
                            if val:
                                self._extract_uses(val, src, lines, uses)
                            if alias:
                                self._extract_targets(alias, src, lines, defs)
                elif child.type == "block":
                    self._walk(child, src, lines, defs, uses, root_func)
            return

        # ---- except clause:  except Exception as e  -----------------
        if t == "except_clause":
            # The alias after 'as' is a new DEF
            as_seen = False
            for child in node.children:
                if child.type == "as":
                    as_seen = True
                    continue
                if as_seen and child.type == "identifier":
                    name = _text(child, src)
                    line, snip = _line_snip(child, lines)
                    defs.append((name, line, snip))
                    as_seen = False
                else:
                    self._extract_uses(child, src, lines, uses)
            # Process the handler body
            for child in node.children:
                if child.type == "block":
                    self._walk(child, src, lines, defs, uses, root_func)
            return

        # ---- function / method definition  ---------------------------
        # Csak a gyökér függvény paramétereit és body-ját dolgozzuk fel.
        # Beágyazott függvény esetén kihagyjuk a body-t, hogy a belső
        # paraméterek ne kerüljenek a külső scope DEF-jei közé.
        if t in ("function_definition", "async_function_definition"):
            params = node.child_by_field_name("parameters")
            body   = node.child_by_field_name("body")
            if params:
                self._extract_params(params, src, lines, defs)
            if body and node is root_func:
                self._walk(body, src, lines, defs, uses, root_func)
            return

        # ---- list / dict / set comprehension  -----------------------
        # The iteration variable inside [ x for x in ... ] is a DEF.
        if t in ("list_comprehension", "set_comprehension",
                 "dictionary_comprehension", "generator_expression"):
            for child in node.children:
                if child.type == "for_in_clause":
                    lv = child.child_by_field_name("left")
                    rv = child.child_by_field_name("right")
                    if lv:
                        self._extract_targets(lv, src, lines, defs)
                    if rv:
                        self._extract_uses(rv, src, lines, uses)
                else:
                    self._walk(child, src, lines, defs, uses, root_func)
            return

        # ---- everything else: just recurse --------------------------
        for child in node.children:
            self._walk(child, src, lines, defs, uses, root_func)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_targets(self, node, src, lines, defs):
        """
        Pull DEF names out of the LEFT side of an assignment.
        Handles simple identifiers, tuple/list unpacking, and starred targets.
        """
        t = node.type
        if t == "identifier":
            name = _text(node, src)
            if name not in _SKIP_NAMES:
                line, snip = _line_snip(node, lines)
                defs.append((name, line, snip))
        elif t in ("tuple_pattern", "list_pattern",
                   "pattern_list", "tuple", "list"):
            for child in node.children:
                self._extract_targets(child, src, lines, defs)
        elif t == "starred_expression":
            inner = node.children[-1] if node.children else None
            if inner:
                self._extract_targets(inner, src, lines, defs)
        # attribute access (self.x = ...) is intentionally ignored:
        # we only track local variable flow

    def _extract_params(self, params_node, src, lines, defs):
        """
        Extract function parameter names as DEF nodes.
        Handles plain params, defaults, type annotations, *args, **kwargs.
        """
        for child in params_node.children:
            t = child.type
            if t == "identifier":
                name = _text(child, src)
                if name not in _SKIP_NAMES:
                    line, snip = _line_snip(child, lines)
                    defs.append((name, line, snip))
            elif t in ("default_parameter", "typed_parameter",
                       "typed_default_parameter", "list_splat_pattern",
                       "dictionary_splat_pattern"):
                name_node = child.child_by_field_name("name")
                if name_node is None and child.children:
                    # fallback: first identifier child
                    for c in child.children:
                        if c.type == "identifier":
                            name_node = c
                            break
                if name_node:
                    name = _text(name_node, src)
                    if name not in _SKIP_NAMES:
                        line, snip = _line_snip(name_node, lines)
                        defs.append((name, line, snip))

    def _extract_uses(self, node, src, lines, uses):
        """
        Collect every identifier that is *read* inside an expression.
        We skip identifiers that are function/attribute names in a call
        because those are structural, not data-flow references.
        """
        t = node.type
        if t == "identifier":
            name = _text(node, src)
            if name not in _SKIP_NAMES:
                line, snip = _line_snip(node, lines)
                uses.append((name, line, snip))
        elif t == "call":
            # For  foo(a, b)  only record a and b as USEs, not foo.
            # For  obj.method(a)  record obj and a, not method.
            func_node = node.child_by_field_name("function")
            args_node = node.child_by_field_name("arguments")
            if func_node:
                # If it's   identifier(...)  skip the function name itself
                # If it's   expr.attr(...)   recurse into the receiver (expr)
                if func_node.type == "attribute":
                    obj = func_node.child_by_field_name("object")
                    if obj:
                        self._extract_uses(obj, src, lines, uses)
                elif func_node.type != "identifier":
                    self._extract_uses(func_node, src, lines, uses)
            if args_node:
                for arg in args_node.children:
                    self._extract_uses(arg, src, lines, uses)
        elif t == "attribute":
            # obj.field  → only record obj as USE, not field
            obj = node.child_by_field_name("object")
            if obj:
                self._extract_uses(obj, src, lines, uses)
        elif t == "string":
            # F-stringek belsejét be kell járni: f"{x}" -> interpolation -> identifier
            for child in node.children:
                if child.type == "interpolation":
                    self._extract_uses(child, src, lines, uses)
        elif t not in ("comment", ":", ",", "(", ")", "[", "]"):
            for child in node.children:
                self._extract_uses(child, src, lines, uses)


# ---------------------------------------------------------------------------
# Small utility functions
# ---------------------------------------------------------------------------

def _text(node, src: str) -> str:
    """Return the raw source text of a Tree-sitter node."""
    return src[node.start_byte:node.end_byte]


def _line_snip(node, lines: list[str]) -> tuple[int, str]:
    """
    Return (1-based line number, stripped source line) for a node.
    Used as a human-readable context for each DEF/USE node.
    """
    row = node.start_point[0]          # 0-based
    snip = lines[row].strip() if row < len(lines) else ""
    return row + 1, snip              # convert to 1-based
