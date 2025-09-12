import ast
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from typing import Optional



import pandas as pd
from git import Repo


# -----------------------
# Segédfüggvények
# -----------------------

def _normalize_code(s: str) -> str:
    """Docstringek, kommentek és felesleges whitespace kiszűrése a stabil hash-hez."""
    # Több soros docstring
    s = re.sub(r'"""[\s\S]*?"""', "", s)
    s = re.sub(r"'''[\s\S]*?'''", "", s)
    # Sorvégi kommentek
    s = re.sub(r"#.*", "", s)
    # Whitespace redukció
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class FuncDef:
    rel_path: str
    qualified_name: str           # pl. Class.method vagy fuggveny
    start_line: int
    end_line: int
    code: str
    body_hash: str

    @property
    def symbol_id(self) -> str:
        return f"{self.rel_path}::{self.qualified_name}"


class _PyFuncVisitor(ast.NodeVisitor):
    """AST visitor: minősített nevek és kódrészletek kinyerése."""
    def __init__(self, code: str, rel_path: str):
        self.code = code
        self.rel_path = rel_path
        self.stack: List[str] = []
        self.results: List[FuncDef] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _add_func(self, node: ast.AST, name: str):
        qname = ".".join(self.stack + [name]) if self.stack else name
        try:
            code_seg = ast.get_source_segment(self.code, node) or ""
        except Exception:
            code_seg = ""
        norm = _normalize_code(code_seg)
        h = _sha256(norm)
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        self.results.append(
            FuncDef(
                rel_path=self.rel_path,
                qualified_name=qname,
                start_line=start,
                end_line=end,
                code=code_seg,
                body_hash=h,
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._add_func(node, node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._add_func(node, node.name)
        self.generic_visit(node)


# -----------------------
# Fő osztály
# -----------------------

class FunctionVersioning:
    """
    Használat:
        fv = FunctionVersioning(repo_path="/abszolút/út/a/klónozott/repo-hoz")
        out = fv.build(function_nodes_df=cg_nodes)   # cg_nodes a CallGraphBuilder outputja

    Visszatér:
        {
          "function_version_nodes": DataFrame[ ID, function_id, symbol_id, commit_sha, authored_datetime,
                                               file_path, qualified_name, body_hash, start_line, end_line, code ],
          "function_version_edges": DataFrame[ source, target ]               # NEXT_VERSION (version -> version)
          "functionversion_function_edges": DataFrame[ source, target ]       # version(ID) -> function(ID)
          "commit_nodes": DataFrame[ ID, sha, authored_datetime, author_name, author_email, message ]
          "functionversion_commit_edges": DataFrame[ source, target ]         # version(ID) -> commit(ID)
        }
    """

    def __init__(self, repo_path: str, log_errors: bool = False):
        if not os.path.isdir(repo_path):
            raise ValueError(f"Nem található a repo: {repo_path}")
        self.repo_path = os.path.abspath(repo_path)
        self.repo = Repo(self.repo_path)
        self.log_errors = log_errors

    # ---------- Public API ----------

    def build(
        self,
        function_nodes_df: pd.DataFrame,
        branch_ref: str = "HEAD",
        only_changed_files: bool = True,
        max_commits: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Commitonként kinyer minden érintett .py fájlból függvényeket, és verzióláncokat épít.
        :param function_nodes_df: a CallGraphBuilder().build_call_graph(...) által visszaadott nodes DF (cg_nodes)
                                  (legalább: ['func_id' vagy 'ID', 'combinedName', 'function_location'])
        :param branch_ref: melyik ágat kövessük (HEAD alapértelmezés)
        :param only_changed_files: ha True, csak a commit által érintett .py fájlokat nézzük;
                                   ha False, minden .py fájlt a fában (lassabb).
        """
        # 1) commitok időrendben
        commits = list(self.repo.iter_commits(branch_ref))
        commits.reverse()  # legrégebbitől a legfrissebbig
        if max_commits:
            commits = commits[-max_commits:]  # csak a legutóbbi N commit
        
        if not commits:
            empty = pd.DataFrame
            return {
                "function_version_nodes": empty(columns=["ID","function_id","symbol_id","commit_sha","authored_datetime","file_path","qualified_name","body_hash","start_line","end_line","code"]),
                "function_version_edges": empty(columns=["source","target"]),
                "functionversion_function_edges": empty(columns=["source","target"]),
                "commit_nodes": empty(columns=["ID","sha","authored_datetime","author_name","author_email","message"]),
                "functionversion_commit_edges": empty(columns=["source","target"]),
            }
        
        # --- fájlok a CG-ből ---
        fn_df = function_nodes_df.copy()
        if "ID" in fn_df.columns and "func_id" not in fn_df.columns:
            fn_df = fn_df.rename(columns={"ID": "func_id"})
        fn_df["function_location"] = fn_df["function_location"].fillna("").astype(str)
        # ugyanaz a normalizálás, mint _project_relpath + replace
        allowed_rel = set(fn_df["function_location"]
                        .apply(self._project_relpath)
                        .str.replace("\\\\", "/", regex=True)
                        .dropna().unique())

        
        # 2) verziók gyűjtése szimbólumonként
        versions_by_symbol: Dict[str, List[dict]] = {}

        for idx, c in enumerate(commits):
            try:
                changed_paths = self._changed_python_files(c) if only_changed_files else self._all_python_files(c)
            except Exception as e:
                if self.log_errors:
                    print(f"[warn] változott fájlok listázása sikertelen {c.hexsha[:7]}: {e}")
                changed_paths = []

            for rel_path, status in changed_paths:
                if status == "D":
                    continue
                if allowed_rel and rel_path not in allowed_rel:
                    continue
                try:
                    code = self._read_file_at_commit(c, rel_path)
                    if code is None:
                        continue
                    for fdef in self._extract_functions(rel_path, code):
                        entry = {
                            "commit_sha": c.hexsha,
                            "authored_datetime": c.authored_datetime.isoformat() if hasattr(c, "authored_datetime") else "",
                            "file_path": rel_path,
                            "qualified_name": fdef.qualified_name,
                            "symbol_id": fdef.symbol_id,
                            "body_hash": fdef.body_hash,
                            "start_line": fdef.start_line,
                            "end_line": fdef.end_line,
                            "code": fdef.code,
                            "author_name": getattr(c.author, "name", ""),
                            "author_email": getattr(c.author, "email", ""),
                            "message": (c.message or "").strip().splitlines()[0] if hasattr(c, "message") else "",
                        }
                        versions_by_symbol.setdefault(fdef.symbol_id, []).append(entry)
                except Exception as e:
                    if self.log_errors:
                        print(f"[err] {c.hexsha[:7]} {rel_path}: {e}")
                    continue

        # 3) időrend szerinti rendezés és csomópontok/élek építése
        return self._build_dfs(versions_by_symbol, function_nodes_df)

    # ---------- Belso segédek ----------

    def _project_relpath(self, abspath: str) -> str:
        abspath = os.path.abspath(abspath)
        if abspath.startswith(self.repo_path):
            relp = abspath[len(self.repo_path):].lstrip(os.sep)
            return relp.replace("\\", "/")
        return abspath.replace("\\", "/")

    def _read_file_at_commit(self, commit, rel_path: str) -> Optional[str]:
        try:
            blob = (commit.tree / rel_path)
            data = blob.data_stream.read()
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _changed_python_files(self, commit) -> List[Tuple[str, str]]:
        """(rel_path, status) listát ad vissza az előző commit(hoz) képest."""
        out: List[Tuple[str, str]] = []

        if commit.parents:
            diffs = commit.parents[0].diff(commit, create_patch=False)
            for d in diffs:
                p = getattr(d, "b_path", None) or getattr(d, "a_path", None)
                if not p or not p.endswith(".py"):
                    continue
                if getattr(d, "deleted_file", False):
                    status = "D"
                elif getattr(d, "new_file", False):
                    status = "A"
                else:
                    status = "M"
                out.append((p, status))
        else:
            # első commit: minden blob új fájlnak számít
            for blob in commit.tree.traverse():
                if getattr(blob, "type", None) == "blob" and blob.path.endswith(".py"):
                    out.append((blob.path, "A"))

        # egyedisítés
        seen, uniq = set(), []
        for p, s in out:
            if p not in seen:
                uniq.append((p, s))
                seen.add(p)
        return uniq    


    def _all_python_files(self, commit) -> List[Tuple[str, str]]:
        paths = []
        for blob in commit.tree.traverse():
            if getattr(blob, "type", None) == "blob" and blob.path.endswith(".py"):
                paths.append((blob.path, "M"))
        return paths

    def _extract_functions(self, rel_path: str, code: str) -> List[FuncDef]:
        try:
            tree = ast.parse(code)
        except Exception:
            return []
        v = _PyFuncVisitor(code, rel_path)
        v.visit(tree)
        return v.results

    def _build_dfs(self, versions_by_symbol: Dict[str, List[dict]], function_nodes_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # 3/a) commit-táblázat
        commit_rows: Dict[str, dict] = {}
        for vers in versions_by_symbol.values():
            for rec in vers:
                sha = rec["commit_sha"]
                if sha not in commit_rows:
                    commit_rows[sha] = {
                        "sha": sha,
                        "authored_datetime": rec["authored_datetime"],
                        "author_name": rec["author_name"],
                        "author_email": rec["author_email"],
                        "message": rec["message"],
                    }
        commit_nodes = pd.DataFrame(list(commit_rows.values())).reset_index(drop=True)
        commit_nodes.insert(0, "ID", range(1, len(commit_nodes) + 1))
        commit_id_map = {row["sha"]: int(row["ID"]) for _, row in commit_nodes.iterrows()}

        # 3/b) FunctionVersion node-ok és NEXT_VERSION élek
        fv_nodes_rows: List[dict] = []
        next_edges_rows: List[dict] = []

        # Szimbolumonként időrend és egymás utáni hash-változások figyelése
        def _sort_key(x):
            # authored_datetime lehet üres -> akkor a commit sha alapján rendezünk
            return (x.get("authored_datetime") or "", x["commit_sha"])

        fv_id_counter = 1
        version_key_to_id: Dict[Tuple[str, str, str], int] = {}  # (symbol_id, body_hash, commit_sha) -> version_id

        for symbol_id, items in versions_by_symbol.items():
            items_sorted = sorted(items, key=_sort_key)
            prev_ver_id: Optional[int] = None
            prev_hash: Optional[str] = None

            for rec in items_sorted:
                # ugyanabban a szimbólumban csak akkor új verzió, ha megváltozik a body_hash
                if rec["body_hash"] == prev_hash:
                    continue

                v_id = fv_id_counter
                fv_id_counter += 1
                version_key_to_id[(symbol_id, rec["body_hash"], rec["commit_sha"])] = v_id

                fv_nodes_rows.append({
                    "ID": v_id,
                    "function_id": None,  # később kitöltjük a CG-hez kötve
                    "symbol_id": symbol_id,
                    "commit_sha": rec["commit_sha"],
                    "authored_datetime": rec["authored_datetime"],
                    "file_path": rec["file_path"],
                    "qualified_name": rec["qualified_name"],
                    "body_hash": rec["body_hash"],
                    "start_line": rec["start_line"],
                    "end_line": rec["end_line"],
                    "code": rec["code"],
                })

                if prev_ver_id is not None:
                    next_edges_rows.append({"source": prev_ver_id, "target": v_id})

                prev_ver_id = v_id
                prev_hash = rec["body_hash"]

        function_version_nodes = pd.DataFrame(fv_nodes_rows, columns=[
            "ID", "function_id", "symbol_id", "commit_sha", "authored_datetime", "file_path",
            "qualified_name", "body_hash", "start_line", "end_line", "code"
        ])
        function_version_edges = pd.DataFrame(next_edges_rows, columns=["source", "target"])

        # 3/c) Version -> Commit élek
        fv_commit_edges = function_version_nodes[["ID", "commit_sha"]].copy()
        fv_commit_edges["target"] = fv_commit_edges["commit_sha"].map(commit_id_map)
        fv_commit_edges = fv_commit_edges.drop(columns=["commit_sha"]).rename(columns={"ID": "source"})
        fv_commit_edges = fv_commit_edges.dropna().reset_index(drop=True).astype({"source": int, "target": int})

        # 3/d) Version -> Function élek (CG-hez kötés)
        # function_nodes_df oszlopok: a repo CG-ben 'func_id' vagy 'ID' néven azonosított
        fn_df = function_nodes_df.copy()
        if "ID" in fn_df.columns and "func_id" not in fn_df.columns:
            fn_df = fn_df.rename(columns={"ID": "func_id"})
        # relatív fájlút a CG-ben
        fn_df["function_location"] = fn_df["function_location"].fillna("").astype(str)
        fn_df["rel_path"] = fn_df["function_location"].apply(self._project_relpath)
        # néha a CG-ben teljes path + repo gyökér is benne lehet → normalizálás
        # (ha már relatív, ez nem változtat)
        fn_df["rel_path"] = fn_df["rel_path"].str.replace("\\\\", "/", regex=True)

        # Symbol_id formátum: "{rel_path}::{qualified_name}"
        name_col = "combinedName" if "combinedName" in fn_df.columns else None
        if name_col is None:
            # nincs név oszlop – akkor nem tudunk Function-hoz kötni, de a verziók így is létrejönnek
            fn_df["symbol_id"] = None
        else:
            fn_df["symbol_id"] = fn_df.apply(
                lambda r: f"{r['rel_path']}::{r[name_col]}" if pd.notna(r.get("rel_path")) and pd.notna(r.get(name_col))
                else None,
                axis=1
            )

        symbol_to_funcid = (
            fn_df.dropna(subset=["symbol_id", "func_id"])[["symbol_id", "func_id"]]
            .drop_duplicates()
            .set_index("symbol_id")["func_id"].to_dict()
        )

        function_version_nodes["function_id"] = function_version_nodes["symbol_id"].map(symbol_to_funcid)
        fv_func_edges = function_version_nodes.dropna(subset=["function_id"])[["ID", "function_id"]].copy()
        fv_func_edges = fv_func_edges.rename(columns={"ID": "source", "function_id": "target"}).astype({"source": int, "target": int})

        # 3/e) commit_nodes ID-k már ki vannak számolva
        # rendezett, típushelyes DF-ek visszaadása
        commit_nodes = commit_nodes[["ID", "sha", "authored_datetime", "author_name", "author_email", "message"]].copy()

        return {
            "function_version_nodes": function_version_nodes.reset_index(drop=True),
            "function_version_edges": function_version_edges.reset_index(drop=True),
            "functionversion_function_edges": fv_func_edges.reset_index(drop=True),
            "commit_nodes": commit_nodes.reset_index(drop=True),
            "functionversion_commit_edges": fv_commit_edges.reset_index(drop=True),
        }
