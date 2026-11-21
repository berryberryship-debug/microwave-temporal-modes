
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QUANTUM-LAB — HYBRID GRAPH RAG
==============================

Moteur indépendant d'analyse du dépôt Quantum-Lab.

Fonctions :
    - Extraction AST des modules, classes et fonctions.
    - Cartographie des dépendances avec NetworkX MultiDiGraph.
    - Résolution prudente des imports et appels internes.
    - Index lexical BM25.
    - Recherche sémantique SentenceTransformer facultative.
    - Fusion des classements par Reciprocal Rank Fusion.
    - Extraction du contexte de code pour audit assisté par LLM.
    - Interface CLI autonome.

Séparation des responsabilités :
    Ce module analyse le code source.
    Il ne participe pas aux calculs physiques.
    Il ne valide pas les résultats des simulations quantiques.

Installation :
    pip install numpy networkx rank-bm25 sentence-transformers

Utilisation :
    python hybrid_graph_rag.py --repo . --build-only
    python hybrid_graph_rag.py --repo . --query "calibration SLH"
    python hybrid_graph_rag.py --repo . --query "wavepackets" --top-k 8

Sans SentenceTransformer :
    python hybrid_graph_rag.py --repo . --query "calibration" --lexical-only
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import logging
import re

from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

LOGGER = logging.getLogger("QuantumLab.HybridGraphRAG")

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    ".idea",
    ".vscode",
}


# ============================================================
# OUTILS AST
# ============================================================

def call_name(func: ast.AST) -> str | None:
    """Retourne un nom d'appel syntaxique sans inférence arbitraire."""

    if isinstance(func, ast.Name):
        return func.id

    if isinstance(func, ast.Attribute):
        parts = []
        current = func

        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))

        if isinstance(current, ast.Call):
            if (
                isinstance(current.func, ast.Name)
                and current.func.id == "super"
            ):
                return "super()." + ".".join(reversed(parts))

    return None


class ScopeVisitor(ast.NodeVisitor):
    """
    Collecte les appels et imports avec leur portée lexicale.

    Les décorateurs, annotations et valeurs par défaut sont visités
    dans la portée extérieure à la fonction ou à la classe.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.scope_stack = []
        self.calls = []
        self.imports = []

    @property
    def current_scope(self):
        if self.scope_stack:
            return ".".join(self.scope_stack)
        return f"{self.module_name}::module"

    def _visit_function_header(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)

        args = node.args

        for default in args.defaults:
            self.visit(default)

        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)

        all_args = (
            args.posonlyargs
            + args.args
            + args.kwonlyargs
        )

        for arg in all_args:
            if arg.annotation:
                self.visit(arg.annotation)

        if args.vararg and args.vararg.annotation:
            self.visit(args.vararg.annotation)

        if args.kwarg and args.kwarg.annotation:
            self.visit(args.kwarg.annotation)

        if node.returns:
            self.visit(node.returns)

    def _visit_function(self, node):
        self._visit_function_header(node)

        self.scope_stack.append(node.name)

        for statement in node.body:
            self.visit(statement)

        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node)

    def visit_ClassDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)

        for base in node.bases:
            self.visit(base)

        for keyword in node.keywords:
            self.visit(keyword.value)

        self.scope_stack.append(node.name)

        for statement in node.body:
            self.visit(statement)

        self.scope_stack.pop()

    def visit_Call(self, node):
        self.calls.append({
            "source": self.current_scope,
            "name": call_name(node.func),
            "line": getattr(node, "lineno", None),
        })
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "kind": "import",
                "module": alias.name,
                "name": None,
                "alias": alias.asname,
                "level": 0,
                "line": getattr(node, "lineno", None),
            })

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append({
                "kind": "from",
                "module": node.module or "",
                "name": alias.name,
                "alias": alias.asname,
                "level": node.level,
                "line": getattr(node, "lineno", None),
            })


# ============================================================
# MOTEUR PRINCIPAL
# ============================================================

class RepoHybridGraphRAG:

    def __init__(
        self,
        repo_path=".",
        embed_model="all-MiniLM-L6-v2",
        excluded_dirs=None,
        batch_size=32,
        lazy_encoder=True,
    ):
        self.repo_path = Path(repo_path).resolve()

        if not self.repo_path.is_dir():
            raise NotADirectoryError(
                f"Dépôt introuvable : {self.repo_path}"
            )

        if batch_size < 1:
            raise ValueError("batch_size doit être supérieur à zéro.")

        self.embed_model_name = embed_model
        self.batch_size = batch_size
        self.excluded_dirs = IGNORED_DIRS | set(excluded_dirs or [])

        # Plusieurs relations distinctes peuvent relier deux nœuds.
        self.graph = nx.MultiDiGraph()

        self.nodes_data = {}
        self.node_ids = []

        self.bm25 = None
        self.embeddings = None

        self.file_asts = {}
        self.module_paths = {}
        self.symbol_index = defaultdict(list)
        self.import_maps = {}

        self.parse_errors = []
        self.indexed = False
        self.semantic_indexed = False

        self._encoder = None

        if not lazy_encoder:
            self._init_encoder()

    # ========================================================
    # ENCODEUR
    # ========================================================

    def _init_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SentenceTransformer est requis pour la recherche "
                    "sémantique. Installer avec : "
                    "pip install sentence-transformers"
                ) from exc

            LOGGER.info(
                "Chargement du modèle : %s",
                self.embed_model_name,
            )

            self._encoder = SentenceTransformer(
                self.embed_model_name
            )

        return self._encoder

    @property
    def encoder(self):
        return self._init_encoder()

    # ========================================================
    # TOKENISATION
    # ========================================================

    def _tokenize(self, text):
        if not isinstance(text, str):
            return []

        # La casse originale est préservée pour détecter CamelCase.
        raw_tokens = re.findall(r"[A-Za-z0-9_]+", text)
        expanded = []

        for token in raw_tokens:
            expanded.append(token.lower())

            if "_" in token:
                expanded.extend(
                    part.lower()
                    for part in token.split("_")
                    if part
                )

            camel_parts = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\b)|"
                r"[A-Z]?[a-z]+|"
                r"[0-9]+",
                token,
            )

            expanded.extend(
                part.lower()
                for part in camel_parts
            )

        return [
            token for token in expanded
            if len(token) > 1
        ]

    # ========================================================
    # DÉCOUVERTE DES FICHIERS
    # ========================================================

    def _discover_files(self):
        py_files = []
        md_files = []

        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue

            relative = path.relative_to(self.repo_path)

            if any(
                part in self.excluded_dirs
                for part in relative.parts
            ):
                continue

            if path.suffix == ".py":
                py_files.append(path)

            elif path.suffix.lower() == ".md":
                md_files.append(path)

        return sorted(py_files), sorted(md_files)

    def _module_name(self, path):
        relative = path.relative_to(self.repo_path)
        parts = list(relative.with_suffix("").parts)

        if parts and parts[-1] == "__init__":
            parts.pop()

        return ".".join(parts)

    @staticmethod
    def _node_id(module_name, qualified_name=None):
        if (
            qualified_name is None
            or qualified_name == f"{module_name}::module"
        ):
            return f"{module_name}::module"

        return f"{module_name}::{qualified_name}"

    # ========================================================
    # GRAPHE ET NŒUDS
    # ========================================================

    def _add_edge(self, source, target, relation, **attributes):
        """Ajoute une relation sans écraser les relations existantes."""

        if source not in self.graph or target not in self.graph:
            return

        key = (
            relation,
            attributes.get("line"),
            attributes.get("source_is_test", False),
        )

        self.graph.add_edge(
            source,
            target,
            key=key,
            relation=relation,
            **attributes,
        )

    def _add_node(
        self,
        nid,
        file,
        name,
        node_type,
        code,
        doc="",
        signature="",
        identifiers=None,
        line=None,
    ):
        if nid in self.nodes_data:
            return

        identifiers = identifiers or []
        ident_str = " ".join(identifiers)

        dense_text = (
            f"Nom: {name}\n"
            f"Type: {node_type}\n"
            f"Signature: {signature}\n"
            f"Docstring: {doc}\n"
            f"Identifiants: {ident_str}"
        ).strip()

        bm25_text = (
            f"{file} {name} {node_type} "
            f"{signature} {doc} {ident_str} {code}"
        )

        self.graph.add_node(
            nid,
            type=node_type,
            file=file,
            name=name,
        )

        self.nodes_data[nid] = {
            "id": nid,
            "file": file,
            "name": name,
            "type": node_type,
            "code": code,
            "doc": doc,
            "signature": signature,
            "line": line,
            "dense_text": dense_text,
            "bm25_text": bm25_text,
        }

    # ========================================================
    # EXTRACTION AST
    # ========================================================

    @staticmethod
    def _extract_identifiers(node):
        identifiers = set()

        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name):
                identifiers.add(subnode.id)

            elif isinstance(subnode, ast.Attribute):
                identifiers.add(subnode.attr)

            elif isinstance(subnode, ast.arg):
                identifiers.add(subnode.arg)

        return sorted(identifiers)

    @staticmethod
    def _extract_signature(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args

            names = (
                [a.arg for a in args.posonlyargs]
                + [a.arg for a in args.args]
            )

            if args.vararg:
                names.append("*" + args.vararg.arg)

            names.extend(a.arg for a in args.kwonlyargs)

            if args.kwarg:
                names.append("**" + args.kwarg.arg)

            prefix = (
                "async def"
                if isinstance(node, ast.AsyncFunctionDef)
                else "def"
            )

            return f"{prefix} {node.name}({', '.join(names)})"

        if isinstance(node, ast.ClassDef):
            bases = [
                ast.unparse(base)
                for base in node.bases
            ]
            return f"class {node.name}({', '.join(bases)})"

        return ""

    def _extract_ast_nodes(self, tree, source, module_name, filename):
        module_nid = self._node_id(module_name)

        self._add_node(
            nid=module_nid,
            file=filename,
            name=module_name or filename,
            node_type="module",
            code=source,
            doc=ast.get_docstring(tree) or "",
            signature=f"module {module_name}",
            identifiers=self._extract_identifiers(tree),
            line=1,
        )

        def walk_body(body, parent_qualified, parent_nid):
            """
            Parcourt aussi les définitions situées dans if, try,
            with, for, while et autres blocs de contrôle.
            """

            for statement in body:

                if isinstance(statement, ast.ClassDef):
                    qualified = (
                        f"{parent_qualified}.{statement.name}"
                        if parent_qualified else statement.name
                    )

                    nid = self._node_id(module_name, qualified)

                    self._add_node(
                        nid=nid,
                        file=filename,
                        name=qualified,
                        node_type="class",
                        code=ast.get_source_segment(source, statement) or "",
                        doc=ast.get_docstring(statement) or "",
                        signature=self._extract_signature(statement),
                        identifiers=self._extract_identifiers(statement),
                        line=getattr(statement, "lineno", None),
                    )

                    self._add_edge(
                        parent_nid,
                        nid,
                        "CONTIENT",
                    )

                    walk_body(
                        statement.body,
                        qualified,
                        nid,
                    )

                    continue

                if isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    qualified = (
                        f"{parent_qualified}.{statement.name}"
                        if parent_qualified else statement.name
                    )

                    nid = self._node_id(module_name, qualified)

                    node_type = (
                        "method"
                        if parent_nid != module_nid
                        and self.nodes_data[parent_nid]["type"] == "class"
                        else "function"
                    )

                    self._add_node(
                        nid=nid,
                        file=filename,
                        name=qualified,
                        node_type=node_type,
                        code=ast.get_source_segment(source, statement) or "",
                        doc=ast.get_docstring(statement) or "",
                        signature=self._extract_signature(statement),
                        identifiers=self._extract_identifiers(statement),
                        line=getattr(statement, "lineno", None),
                    )

                    self._add_edge(
                        parent_nid,
                        nid,
                        "CONTIENT",
                    )

                    walk_body(
                        statement.body,
                        qualified,
                        nid,
                    )

                    continue

                # Recherche des définitions dans les blocs de contrôle.
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, list):
                        continue

                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        walk_body(
                            [child],
                            parent_qualified,
                            parent_nid,
                        )

                    elif isinstance(child, ast.AST):
                        for nested in ast.iter_child_nodes(child):
                            if isinstance(
                                nested,
                                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                            ):
                                walk_body(
                                    [nested],
                                    parent_qualified,
                                    parent_nid,
                                )

        walk_body(tree.body, None, module_nid)

    # ========================================================
    # RÉSOLUTION DES IMPORTS
    # ========================================================

    def _package_context(self, module_name, filename):
        """
        Un fichier __init__.py appartient au package lui-même.
        Un module ordinaire appartient au package parent.
        """

        if Path(filename).name == "__init__.py":
            return module_name

        return module_name.rpartition(".")[0]

    def _resolve_import_module(self, module_name, imported_module, level, filename):
        if level == 0:
            return imported_module

        package = self._package_context(module_name, filename)

        if not package:
            return None

        relative = "." * level + imported_module

        try:
            return importlib.util.resolve_name(relative, package)
        except (ImportError, ValueError):
            return None

    def _module_exists(self, module_name):
        if module_name in self.module_paths:
            return True

        # Le package racine peut être représenté par un __init__.py.
        return module_name in self.module_paths

    # ========================================================
    # INDEX DES SYMBOLES
    # ========================================================

    def _build_symbol_index(self):
        self.symbol_index.clear()

        for nid, data in self.nodes_data.items():
            if data["type"] not in (
                "function",
                "method",
                "class",
            ):
                continue

            short_name = data["name"].split(".")[-1]

            self.symbol_index[short_name].append(nid)
            self.symbol_index[data["name"]].append(nid)

    # ========================================================
    # RÉSOLUTION SELF / CLS
    # ========================================================

    def _resolve_self_or_cls_call(self, called_name, source_nid):
        if not called_name.startswith(("self.", "cls.")):
            return None

        data = self.nodes_data.get(source_nid)

        if not data:
            return None

        qualified = data["name"].split(".")

        if len(qualified) < 2:
            return None

        # Recherche de la classe ancêtre la plus proche.
        class_qualified = None

        for index in range(len(qualified) - 1, 0, -1):
            candidate_name = ".".join(qualified[:index])
            candidate_nid = self._node_id(
                source_nid.split("::", 1)[0],
                candidate_name,
            )

            candidate_data = self.nodes_data.get(candidate_nid)

            if candidate_data and candidate_data["type"] == "class":
                class_qualified = candidate_name
                break

        if class_qualified is None:
            return None

        module_name = source_nid.split("::", 1)[0]
        method_name = called_name.split(".", 1)[1]

        # Refuse les attributs complexes non résolus.
        if "." in method_name:
            return None

        target = self._node_id(
            module_name,
            f"{class_qualified}.{method_name}",
        )

        target_data = self.nodes_data.get(target)

        if target_data and target_data["type"] == "method":
            return target

        return None

    # ========================================================
    # RÉSOLUTION DES APPELS
    # ========================================================

    def _resolve_call(self, called_name, source_nid, module_name):
        if not called_name:
            return None

        if called_name.startswith(("self.", "cls.")):
            return self._resolve_self_or_cls_call(
                called_name,
                source_nid,
            )

        # super() n'est pas résolu sans analyse fiable de l'héritage.
        if called_name.startswith("super()."):
            return None

        imports = self.import_maps.get(
            module_name,
            {"symbols": {}, "modules": {}},
        )

        imported_symbols = imports["symbols"]
        imported_modules = imports["modules"]

        parts = called_name.split(".")

        # 1. Symbole importé.
        if parts[0] in imported_symbols:
            target = imported_symbols[parts[0]]

            if len(parts) == 1:
                return target

            target_data = self.nodes_data.get(target)

            if target_data and target_data["type"] == "module":
                module = target_data["name"]
                candidate = self._node_id(
                    module,
                    ".".join(parts[1:]),
                )

                if candidate in self.nodes_data:
                    return candidate

        # 2. Module importé.
        if parts[0] in imported_modules:
            imported_module = imported_modules[parts[0]]

            remaining = ".".join(parts[1:])

            # Chercher le module le plus spécifique disponible.
            candidate_modules = [
                imported_module,
            ]

            if remaining:
                candidate_modules.append(
                    f"{imported_module}.{remaining}"
                )

            for module in reversed(candidate_modules):
                if self._module_exists(module):
                    qualified = (
                        remaining
                        if module == imported_module
                        else None
                    )

                    candidate = self._node_id(
                        module,
                        qualified,
                    )

                    if candidate in self.nodes_data:
                        return candidate

            candidate = self._node_id(
                imported_module,
                remaining,
            )

            if candidate in self.nodes_data:
                return candidate

        # 3. Résolution locale.
        local_candidate = self._node_id(
            module_name,
            called_name,
        )

        if local_candidate in self.nodes_data:
            return local_candidate

        # 4. Recherche globale seulement si le nom est unique
        # et désigne une fonction ou classe de premier niveau.
        if len(parts) == 1:
            candidates = list(dict.fromkeys(
                self.symbol_index.get(called_name, [])
            ))

            if len(candidates) == 1:
                target = candidates[0]
                target_data = self.nodes_data[target]

                if "." not in target_data["name"]:
                    return target

        return None

    # ========================================================
    # CONSTRUCTION DES RELATIONS
    # ========================================================

    def _extract_relationships(self):
        self._build_symbol_index()

        for filename, info in self.file_asts.items():
            tree = info["tree"]
            module_name = info["module_name"]
            module_nid = self._node_id(module_name)

            path_parts = Path(filename).parts

            is_test = (
                Path(filename).name.startswith("test_")
                or "tests" in path_parts
            )

            visitor = ScopeVisitor(module_name)
            visitor.visit(tree)

            import_symbols = {}
            import_modules = {}

            # ------------------------------
            # IMPORTS
            # ------------------------------

            for item in visitor.imports:
                if item["kind"] == "import":
                    imported = item["module"]

                    if item["alias"]:
                        local_name = item["alias"]
                        import_modules[local_name] = imported
                    else:
                        local_name = imported.split(".")[0]
                        import_modules[local_name] = local_name

                    if imported in self.module_paths:
                        target_nid = self._node_id(imported)

                        self._add_edge(
                            module_nid,
                            target_nid,
                            "IMPORTE",
                            line=item["line"],
                            source_is_test=is_test,
                        )

                elif item["kind"] == "from":
                    imported_module = self._resolve_import_module(
                        module_name,
                        item["module"],
                        item.get("level", 0),
                        filename,
                    )

                    if imported_module is None:
                        continue

                    imported_name = item["name"]
                    local_name = item["alias"] or imported_name

                    if imported_name == "*":
                        continue

                    imported_symbol = self._node_id(
                        imported_module,
                        imported_name,
                    )

                    imported_submodule = (
                        f"{imported_module}.{imported_name}"
                        if imported_module
                        else imported_name
                    )

                    imported_module_node = self._node_id(
                        imported_module,
                    )

                    if imported_symbol in self.nodes_data:
                        import_symbols[local_name] = imported_symbol

                        self._add_edge(
                            module_nid,
                            imported_symbol,
                            "IMPORTE",
                            line=item["line"],
                            source_is_test=is_test,
                        )

                    elif imported_submodule in self.module_paths:
                        import_modules[local_name] = imported_submodule

                        target_nid = self._node_id(
                            imported_submodule,
                        )

                        self._add_edge(
                            module_nid,
                            target_nid,
                            "IMPORTE",
                            line=item["line"],
                            source_is_test=is_test,
                        )

                    elif imported_module in self.module_paths:
                        import_symbols[local_name] = imported_module_node

                        self._add_edge(
                            module_nid,
                            imported_module_node,
                            "IMPORTE",
                            line=item["line"],
                            source_is_test=is_test,
                        )

            self.import_maps[module_name] = {
                "symbols": import_symbols,
                "modules": import_modules,
            }

            # ------------------------------
            # APPELS
            # ------------------------------

            for call in visitor.calls:
                called_name = call["name"]

                if not called_name:
                    continue

                source_scope = call["source"]

                if source_scope == f"{module_name}::module":
                    source_nid = module_nid
                else:
                    source_nid = self._node_id(
                        module_name,
                        source_scope,
                    )

                if source_nid not in self.graph:
                    source_nid = module_nid

                target_nid = self._resolve_call(
                    called_name,
                    source_nid,
                    module_name,
                )

                if target_nid and target_nid != source_nid:
                    self._add_edge(
                        source_nid,
                        target_nid,
                        "APPELLE",
                        line=call["line"],
                        source_is_test=is_test,
                    )

    # ========================================================
    # INDEXATION
    # ========================================================

    def build_index(self, build_semantic=False):
        LOGGER.info("Découverte des fichiers...")

        py_files, md_files = self._discover_files()

        self.graph.clear()
        self.nodes_data.clear()
        self.file_asts.clear()
        self.module_paths.clear()
        self.import_maps.clear()
        self.parse_errors.clear()

        self.bm25 = None
        self.embeddings = None
        self.semantic_indexed = False
        self.indexed = False

        # ------------------------------
        # DOCUMENTATION MARKDOWN
        # ------------------------------

        for md_file in md_files:
            relative = md_file.relative_to(
                self.repo_path
            ).as_posix()

            try:
                content = md_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                LOGGER.warning(
                    "Lecture impossible : %s (%s)",
                    relative,
                    exc,
                )
                continue

            nid = f"{relative}::doc"

            self._add_node(
                nid=nid,
                file=relative,
                name=relative,
                node_type="documentation",
                code=content,
                doc="Documentation Markdown",
                signature=relative,
                identifiers=self._tokenize(content),
            )

        # ------------------------------
        # MODULES PYTHON
        # ------------------------------

        for py_file in py_files:
            relative = py_file.relative_to(
                self.repo_path
            ).as_posix()

            module_name = self._module_name(py_file)

            try:
                source = py_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                tree = ast.parse(
                    source,
                    filename=relative,
                )

            except (SyntaxError, UnicodeError, OSError) as exc:
                self.parse_errors.append({
                    "file": relative,
                    "error": str(exc),
                })

                LOGGER.warning(
                    "AST ignoré : %s (%s)",
                    relative,
                    exc,
                )
                continue

            self.module_paths[module_name] = relative

            self.file_asts[relative] = {
                "tree": tree,
                "source": source,
                "module_name": module_name,
            }

            self._extract_ast_nodes(
                tree,
                source,
                module_name,
                relative,
            )

        # ------------------------------
        # RELATIONS
        # ------------------------------

        self._extract_relationships()

        self.node_ids = list(self.nodes_data.keys())

        if not self.node_ids:
            LOGGER.warning("Aucun nœud indexable.")
            return self

        # ------------------------------
        # INDEX BM25
        # ------------------------------

        tokenized_corpus = [
            self._tokenize(
                self.nodes_data[nid]["bm25_text"]
            )
            for nid in self.node_ids
        ]

        # BM25 nécessite un corpus non vide.
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.indexed = True

        # ------------------------------
        # INDEX SÉMANTIQUE FACULTATIF
        # ------------------------------

        if build_semantic:
            self.build_semantic_index()

        LOGGER.info(
            "Indexation terminée : %d nœuds | %d relations | "
            "%d erreurs AST",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self.parse_errors),
        )

        return self

    def build_semantic_index(self):
        if not self.indexed:
            raise RuntimeError(
                "Exécuter build_index() avant l'index sémantique."
            )

        dense_corpus = [
            self.nodes_data[nid]["dense_text"]
            for nid in self.node_ids
        ]

        LOGGER.info(
            "Encodage sémantique de %d nœuds...",
            len(dense_corpus),
        )

        self.embeddings = self.encoder.encode(
            dense_corpus,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self.embeddings = np.asarray(
            self.embeddings,
            dtype=np.float32,
        )

        self.semantic_indexed = True

        LOGGER.info(
            "Index sémantique construit : %s",
            self.embeddings.shape,
        )

        return self

    # ========================================================
    # RECHERCHE
    # ========================================================

    @staticmethod
    def _stable_ranks(scores):
        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        order = np.argsort(
            -scores,
            kind="stable",
        )

        ranks = np.empty(
            len(scores),
            dtype=np.int64,
        )

        ranks[order] = np.arange(
            1,
            len(scores) + 1,
        )

        return ranks

    def hybrid_search(
        self,
        query,
        top_k=5,
        rrf_k=60,
        lexical_only=False,
    ):
        if not self.indexed:
            raise RuntimeError(
                "Exécuter build_index() avant la recherche."
            )

        if top_k < 1:
            raise ValueError("top_k doit être supérieur à zéro.")

        q_tokens = self._tokenize(query)

        bm25_scores = np.asarray(
            self.bm25.get_scores(q_tokens),
            dtype=np.float64,
        )

        # Si aucun terme ne correspond, ne pas attribuer
        # artificiellement un classement lexical à tous les nœuds.
        lexical_active = (
            len(q_tokens) > 0
            and np.any(bm25_scores > 0)
        )

        rrf_scores = defaultdict(float)

        if lexical_active:
            bm25_ranks = self._stable_ranks(
                bm25_scores
            )

            for index, nid in enumerate(self.node_ids):
                rrf_scores[nid] += (
                    1.0 / (rrf_k + bm25_ranks[index])
                )

        if self.semantic_indexed and not lexical_only:
            q_emb = self.encoder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]

            vec_scores = np.dot(
                self.embeddings,
                q_emb,
            )

            vec_ranks = self._stable_ranks(
                vec_scores
            )

            for index, nid in enumerate(self.node_ids):
                rrf_scores[nid] += (
                    1.0 / (rrf_k + vec_ranks[index])
                )

        # Repli lexical si l'index sémantique n'est pas disponible.
        if not rrf_scores:
            if not lexical_active:
                return []

        ranked = sorted(
            rrf_scores.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return ranked[:top_k]

    # ========================================================
    # CONTEXTE GRAPHIQUE
    # ========================================================

    def query_with_graph(
        self,
        query,
        top_k=5,
        max_neighbors=10,
        lexical_only=False,
    ):
        anchors = self.hybrid_search(
            query,
            top_k=top_k,
            lexical_only=lexical_only,
        )

        context_bundle = []
        visited = set()

        def add_context(nid, role, score=None):
            if (
                nid in visited
                or nid not in self.nodes_data
            ):
                return

            visited.add(nid)

            context_bundle.append({
                "id": nid,
                "role": role,
                "score": score,
            })

        for nid, score in anchors:
            add_context(
                nid,
                "ANCRAGE",
                score,
            )

            neighbors = []

            for _, target, key, data in self.graph.out_edges(
                nid,
                keys=True,
                data=True,
            ):
                relation = data.get("relation")

                if relation != "CONTIENT":
                    neighbors.append((
                        target,
                        f"DÉPENDANCE : {relation}",
                    ))

            for source, _, key, data in self.graph.in_edges(
                nid,
                keys=True,
                data=True,
            ):
                relation = data.get("relation")

                if relation != "CONTIENT":
                    neighbors.append((
                        source,
                        f"APPELANT : {relation}",
                    ))

            for target, role in neighbors[:max_neighbors]:
                add_context(target, role)

        return context_bundle

    # ========================================================
    # DÉDUPLICATION HIÉRARCHIQUE
    # ========================================================

    def deduplicate_hierarchy(self, context_bundle):
        selected_nids = [
            item["id"]
            for item in context_bundle
        ]

        pruned_bundle = []

        for item in context_bundle:
            nid = item["id"]
            data = self.nodes_data.get(nid)

            if not data:
                continue

            node_type = data["type"]

            if node_type == "module":
                module_prefix = nid.replace(
                    "::module",
                    "::",
                )

                has_child = any(
                    other != nid
                    and other.startswith(module_prefix)
                    for other in selected_nids
                )

                if has_child:
                    continue

            elif node_type == "class":
                class_prefix = f"{nid}."

                has_child = any(
                    other.startswith(class_prefix)
                    for other in selected_nids
                )

                if has_child:
                    continue

            pruned_bundle.append(item)

        return pruned_bundle

    # ========================================================
    # CONTEXTE LLM
    # ========================================================

    def format_prompt_context(
        self,
        query,
        top_k=5,
        max_neighbors=10,
        max_chars=12000,
        lexical_only=False,
    ):
        if max_chars is not None and max_chars < 1:
            raise ValueError(
                "max_chars doit être supérieur à zéro."
            )

        raw_bundle = self.query_with_graph(
            query,
            top_k=top_k,
            max_neighbors=max_neighbors,
            lexical_only=lexical_only,
        )

        pruned_bundle = self.deduplicate_hierarchy(
            raw_bundle
        )

        header = [
            "=== CONTEXTE DE RECHERCHE HYBRIDE ===",
            f"REQUÊTE : {query}",
            f"NŒUDS SÉLECTIONNÉS : {len(pruned_bundle)}",
            "",
        ]

        blocks = []
        total_chars = sum(
            len(block) + 1
            for block in header
        )

        for item in pruned_bundle:
            nid = item["id"]
            data = self.nodes_data[nid]

            block_header = (
                f"--- [{item['role']}] {nid} ---"
            )

            if item["score"] is not None:
                block_header += (
                    f" [RRF={item['score']:.6f}]"
                )

            content = (
                f"{block_header}\n"
                f"TYPE : {data['type']}\n"
                f"FICHIER : {data['file']}\n"
                f"LIGNE : {data['line']}\n"
                f"SIGNATURE : {data['signature']}\n"
                f"DOCSTRING : {data['doc']}\n"
                f"{data['code']}\n"
            )

            separator = "\n"
            required = len(separator) + len(content)

            if (
                max_chars is not None
                and total_chars + required > max_chars
            ):
                remaining = max_chars - total_chars

                if remaining > 100:
                    content = content[:remaining - 50]
                    blocks.append(
                        content
                        + "\n[CONTENU PARTIEL]"
                    )

                blocks.append(
                    "[CONTEXTE TRONQUÉ : limite atteinte]"
                )
                break

            blocks.append(content)
            total_chars += required

        return "\n".join(header + blocks)

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    def diagnostic(self):
        relation_counts = defaultdict(int)

        for _, _, data in self.graph.edges(data=True):
            relation_counts[data.get("relation", "INCONNUE")] += 1

        return {
            "repository": str(self.repo_path),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "python_files": len(self.file_asts),
            "modules": len(self.module_paths),
            "parse_errors": len(self.parse_errors),
            "bm25_ready": self.bm25 is not None,
            "semantic_ready": self.semantic_indexed,
            "relations": dict(relation_counts),
            "errors": list(self.parse_errors),
        }


# ============================================================
# TESTS UNITAIRES
# ============================================================

def run_unit_tests():
    import tempfile
    import unittest
    import shutil

    class TestHybridGraphRAG(unittest.TestCase):

        def setUp(self):
            self.temp_dir = tempfile.mkdtemp()
            self.repo = Path(self.temp_dir)

        def tearDown(self):
            shutil.rmtree(self.temp_dir)

        def make_rag(self):
            return RepoHybridGraphRAG(
                repo_path=self.repo,
                lazy_encoder=True,
            )

        def test_lazy_encoder(self):
            (self.repo / "sample.py").write_text(
                "def hello():\n    return 1\n",
                encoding="utf-8",
            )

            rag = self.make_rag()
            rag.build_index()

            self.assertIsNone(rag._encoder)
            self.assertTrue(rag.indexed)
            self.assertIsNotNone(rag.bm25)

        def test_self_resolution(self):
            (self.repo / "worker.py").write_text(
                "class Worker:\n"
                "    def step(self):\n"
                "        self.run()\n"
                "    def run(self):\n"
                "        pass\n",
                encoding="utf-8",
            )

            rag = self.make_rag()
            rag.build_index()

            source = "worker::Worker.step"
            target = "worker::Worker.run"

            self.assertIn(source, rag.graph)
            self.assertIn(target, rag.graph)

            relations = [
                data["relation"]
                for _, _, data in rag.graph.edges(
                    source,
                    data=True,
                )
                if _ == target
            ]

            self.assertIn("APPELLE", relations)

        def test_cls_resolution(self):
            (self.repo / "worker.py").write_text(
                "class Worker:\n"
                "    @classmethod\n"
                "    def create(cls):\n"
                "        cls.build()\n"
                "    @classmethod\n"
                "    def build(cls):\n"
                "        pass\n",
                encoding="utf-8",
            )

            rag = self.make_rag()
            rag.build_index()

            source = "worker::Worker.create"
            target = "worker::Worker.build"

            self.assertTrue(
                any(
                    data["relation"] == "APPELLE"
                    for _, destination, data in rag.graph.edges(
                        source,
                        data=True,
                    )
                    if destination == target
                )
            )

        def test_nested_control_flow_definition(self):
            (self.repo / "sample.py").write_text(
                "if True:\n"
                "    def conditional_function():\n"
                "        return 1\n",
                encoding="utf-8",
            )

            rag = self.make_rag()
            rag.build_index()

            self.assertIn(
                "sample::conditional_function",
                rag.graph,
            )

        def test_homonym_rejection(self):
            (self.repo / "a.py").write_text(
                "class EngineA:\n"
                "    def execute(self):\n"
                "        pass\n",
                encoding="utf-8",
            )

            (self.repo / "b.py").write_text(
                "class EngineB:\n"
                "    def execute(self):\n"
                "        pass\n",
                encoding="utf-8",
            )

            (self.repo / "caller.py").write_text(
                "def launch():\n"
                "    execute()\n",
                encoding="utf-8",
            )

            rag = self.make_rag()
            rag.build_index()

            caller = "caller::launch"

            self.assertFalse(
                any(
                    data["relation"] == "APPELLE"
                    and destination.endswith(".execute")
                    for _, destination, data in rag.graph.edges(
                        caller,
                        data=True,
                    )
                )
            )

        def test_multiple_relations_preserved(self):
            (self.repo / "sample.py").write_text(
                "def target():\n"
                "    pass\n"
                "\n"
                "def caller():\n"
                "    target()\n",
                encoding="utf-8",
            )

            rag = self.make_rag()
            rag.build_index()

            source = "sample::module"
            target = "sample::target"

            rag._add_edge(source, target, "RELATION_TEST_A")
            rag._add_edge(source, target, "RELATION_TEST_B")

            relations = {
                data["relation"]
                for _, _, data in rag.graph.edges(
                    source,
                    data=True,
                )
                if data.get("relation") in {
                    "RELATION_TEST_A",
                    "RELATION_TEST_B",
                }
            }

            self.assertEqual(
                relations,
                {"RELATION_TEST_A", "RELATION_TEST_B"},
            )

        def test_camel_case_tokenization(self):
            rag = self.make_rag()

            tokens = rag._tokenize(
                "SentenceTransformer MultiNodeRouter"
            )

            self.assertIn("sentence", tokens)
            self.assertIn("transformer", tokens)
            self.assertIn("multi", tokens)
            self.assertIn("node", tokens)
            self.assertIn("router", tokens)

        def test_hierarchical_deduplication(self):
            (self.repo / "pipe.py").write_text(
                "class Pipeline:\n"
                "    def process(self):\n"
                "        return 42\n",
                encoding="utf-8",
            )

            rag = self.make_rag()
            rag.build_index()

            bundle = [
                {
                    "id": "pipe::module",
                    "role": "ANCRAGE",
                    "score": 0.05,
                },
                {
                    "id": "pipe::Pipeline",
                    "role": "ANCRAGE",
                    "score": 0.04,
                },
                {
                    "id": "pipe::Pipeline.process",
                    "role": "ANCRAGE",
                    "score": 0.03,
                },
            ]

            result = rag.deduplicate_hierarchy(bundle)

            self.assertEqual(
                [item["id"] for item in result],
                ["pipe::Pipeline.process"],
            )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestHybridGraphRAG
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    return result.wasSuccessful()


# ============================================================
# INTERFACE CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quantum-Lab — moteur indépendant de recherche "
            "hybride et de cartographie du code."
        )
    )

    parser.add_argument(
        "--repo",
        default=".",
        help="Chemin du dépôt à analyser.",
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Question ou termes à rechercher.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de résultats principaux.",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Budget maximal du contexte en caractères.",
    )

    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Construire les index sans effectuer de recherche.",
    )

    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Utiliser uniquement BM25.",
    )

    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Construire l'index SentenceTransformer.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Exécuter les tests unitaires intégrés.",
    )

    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Afficher les statistiques du dépôt.",
    )

    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Modèle SentenceTransformer.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.test:
        success = run_unit_tests()
        raise SystemExit(0 if success else 1)

    rag = RepoHybridGraphRAG(
        repo_path=args.repo,
        embed_model=args.model,
        lazy_encoder=True,
    )

    rag.build_index(
        build_semantic=args.semantic,
    )

    if args.diagnostic:
        diagnostic = rag.diagnostic()

        print("\n=== DIAGNOSTIC QUANTUM-LAB ===")

        for key, value in diagnostic.items():
            print(f"{key}: {value}")

    if args.build_only:
        print("\nIndexation terminée.")
        return

    if args.query:
        context = rag.format_prompt_context(
            query=args.query,
            top_k=args.top_k,
            max_chars=args.max_chars,
            lexical_only=args.lexical_only,
        )

        print("\n" + context)

    elif not args.diagnostic:
        parser.print_help()


if __name__ == "__main__":
    main()
