#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REPO HYBRID GRAPH RAG (V2 Stabilisée)
=====================================

Moteur de recherche hybride robuste pour dépôts Python.

Améliorations V2 :
    - Séparation stricte : syntaxique (AST), sémantique (Dense), lexicale (BM25).
    - Résolution stricte de self/cls dans la classe englobante.
    - Élimination des liaisons arbitraires sur méthodes homonymes non qualifiées.
    - Déduplication hiérarchique du contexte LLM.
    - Tests de non-régression intégrés.

Dépendances :
    pip install numpy networkx rank-bm25 sentence-transformers
"""

import ast
import re
import logging
import unittest
import tempfile
import shutil

from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

LOGGER = logging.getLogger("HybridGraphRAG")

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
    ".vscode"
}


# ============================================================
# PARCOURS AST : PORTÉES, APPELS ET IMPORTS
# ============================================================

class ScopeVisitor(ast.NodeVisitor):
    """
    Parcourt le module AST en conservant la hiérarchie exacte des portées.
    """

    def __init__(self, module_name):
        self.module_name = module_name
        self.scope_stack = []
        self.calls = []
        self.imports = []

    @property
    def current_scope(self):
        if self.scope_stack:
            return self.scope_stack[-1]
        return f"{self.module_name}::module"

    def _enter_scope(self, node):
        name = node.name
        if self.scope_stack:
            qualified = f"{self.scope_stack[-1]}.{name}"
        else:
            qualified = name

        self.scope_stack.append(qualified)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        self._enter_scope(node)

    def visit_AsyncFunctionDef(self, node):
        self._enter_scope(node)

    def visit_ClassDef(self, node):
        self._enter_scope(node)

    def visit_Call(self, node):
        self.calls.append({
            "source": self.current_scope,
            "name": self._call_name(node.func),
            "line": getattr(node, "lineno", None)
        })
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "kind": "import",
                "module": alias.name,
                "name": None,
                "alias": alias.asname
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append({
                "kind": "from",
                "module": node.module or "",
                "name": alias.name,
                "alias": alias.asname,
                "level": node.level
            })
        self.generic_visit(node)

    @staticmethod
    def _call_name(func):
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

        return None


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
        lazy_encoder=False
    ):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Dépôt introuvable : {self.repo_path}")

        self.embed_model_name = embed_model
        self.batch_size = batch_size
        self.excluded_dirs = IGNORED_DIRS | set(excluded_dirs or [])

        self.graph = nx.DiGraph()
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

        self._encoder = None
        if not lazy_encoder:
            self._init_encoder()

    def _init_encoder(self):
        if self._encoder is None:
            LOGGER.info("Chargement du modèle d'embeddings : %s", self.embed_model_name)
            self._encoder = SentenceTransformer(self.embed_model_name)

    @property
    def encoder(self):
        self._init_encoder()
        return self._encoder

    # ========================================================
    # TOKENISATION
    # ========================================================

    def _tokenize(self, text):
        if not isinstance(text, str):
            return []

        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        expanded = []

        for token in tokens:
            expanded.append(token)
            if "_" in token:
                expanded.extend(token.split("_"))

            camel_parts = re.findall(
                r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+",
                token
            )
            expanded.extend(part.lower() for part in camel_parts)

        return [t for t in expanded if len(t) > 1]

    # ========================================================
    # DÉCOUVERTE ET IDENTIFICATION
    # ========================================================

    def _discover_files(self):
        py_files = []
        md_files = []

        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue

            relative = path.relative_to(self.repo_path)
            if any(part in self.excluded_dirs for part in relative.parts):
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
        if qualified_name is None or qualified_name == f"{module_name}::module":
            return f"{module_name}::module"
        return f"{module_name}::{qualified_name}"

    # ========================================================
    # SÉPARATION DES REPRÉSENTATIONS (SYNTAXIQUE / SÉMANTIQUE / LEXICALE)
    # ========================================================

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
        line=None
    ):
        if nid in self.nodes_data:
            return

        identifiers = identifiers or []
        ident_str = " ".join(identifiers)

        # 1. Représentation sémantique dense (compacte, évite la troncature)
        dense_text = (
            f"Nom: {name}\n"
            f"Type: {node_type}\n"
            f"Signature: {signature}\n"
            f"Docstring: {doc}\n"
            f"Identifiants: {ident_str}"
        ).strip()

        # 2. Représentation lexicale BM25 (précision des correspondances)
        bm25_text = (
            f"{file} {name} {node_type} {signature} {doc} {ident_str} {code}"
        )

        self.graph.add_node(
            nid,
            type=node_type,
            file=file,
            name=name
        )

        self.nodes_data[nid] = {
            "id": nid,
            "file": file,
            "name": name,
            "type": node_type,
            "code": code,              # 3. Représentation syntaxique brute
            "doc": doc,
            "signature": signature,
            "line": line,
            "dense_text": dense_text,
            "bm25_text": bm25_text
        }

    # ========================================================
    # EXTRACTION SYNTAXIQUE AST
    # ========================================================

    def _extract_identifiers(self, node):
        idents = set()
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name):
                idents.add(subnode.id)
            elif isinstance(subnode, ast.Attribute):
                idents.add(subnode.attr)
            elif isinstance(subnode, ast.arg):
                idents.add(subnode.arg)
        return sorted(idents)

    def _extract_signature(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            return f"def {node.name}({', '.join(args)})"
        if isinstance(node, ast.ClassDef):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            return f"class {node.name}({', '.join(bases)})"
        return ""

    def _extract_ast_nodes(self, tree, source, module_name, filename):
        module_nid = self._node_id(module_name)
        module_doc = ast.get_docstring(tree) or ""

        # Enregistrement du module
        self._add_node(
            nid=module_nid,
            file=filename,
            name=module_name,
            node_type="module",
            code=source,
            doc=module_doc,
            signature=f"module {module_name}",
            identifiers=self._extract_identifiers(tree),
            line=1
        )

        def visit_scope(node, parent_qualified=None):
            if isinstance(node, ast.ClassDef):
                node_type = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node_type = "function"
            else:
                return

            qualified = f"{parent_qualified}.{node.name}" if parent_qualified else node.name
            nid = self._node_id(module_name, qualified)

            segment = ast.get_source_segment(source, node) or ""
            doc = ast.get_docstring(node) or ""
            sig = self._extract_signature(node)
            idents = self._extract_identifiers(node)

            self._add_node(
                nid=nid,
                file=filename,
                name=qualified,
                node_type=node_type,
                code=segment,
                doc=doc,
                signature=sig,
                identifiers=idents,
                line=getattr(node, "lineno", None)
            )

            parent_nid = (
                self._node_id(module_name, parent_qualified)
                if parent_qualified
                else module_nid
            )

            self.graph.add_edge(parent_nid, nid, relation="CONTIENT")

            for child in node.body:
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit_scope(child, qualified)

        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_scope(node)

    # ========================================================
    # RÉSOLUTION STRICTE DES RELATIONS
    # ========================================================

    def _resolve_import_module(self, module_name, imported_module, level):
        if level == 0:
            return imported_module

        package = module_name.rpartition(".")[0]
        relative = "." * level + imported_module

        try:
            import importlib.util
            return importlib.util.resolve_name(relative, package)
        except (ImportError, ValueError):
            return None

    def _build_symbol_index(self):
        self.symbol_index.clear()
        for nid, data in self.nodes_data.items():
            if data["type"] not in ("function", "class"):
                continue

            short_name = data["name"].split(".")[-1]
            self.symbol_index[short_name].append(nid)
            self.symbol_index[data["name"]].append(nid)

    def _resolve_self_or_cls_call(self, called_name, source_nid):
        """
        Résout strictement les appels self.<method> et cls.<method>
        au sein de la classe englobante.
        """
        if not called_name.startswith(("self.", "cls.")):
            return None

        if source_nid not in self.nodes_data:
            return None

        parts = source_nid.split("::", 1)
        if len(parts) != 2:
            return None

        module_name, qualified_name = parts
        scope = qualified_name.split(".")

        # Doit au moins être sous la forme : Classe.methode
        if len(scope) < 2:
            return None

        # Recherche de la classe parente dans la pile
        class_name = ".".join(scope[:-1])
        method_name = called_name.split(".", 1)[1]

        candidate = f"{module_name}::{class_name}.{method_name}"

        data = self.nodes_data.get(candidate)
        if data is not None and data["type"] == "function":
            return candidate

        return None

    def _resolve_call(self, called_name, source_nid, module_name):
        if not called_name:
            return None

        # 1. Résolution de self.method() et cls.method()
        self_call = self._resolve_self_or_cls_call(called_name, source_nid)
        if self_call:
            return self_call

        # Rejeter explicitement les appels self / cls résiduels non résolus
        if called_name.startswith(("self.", "cls.")):
            return None

        parts = called_name.split(".")
        imports = self.import_maps.get(module_name, {"symbols": {}, "modules": {}})
        imported_symbols = imports["symbols"]
        imported_modules = imports["modules"]

        # 2. Appel via symbole importé
        if parts[0] in imported_symbols:
            target = imported_symbols[parts[0]]
            if len(parts) == 1:
                return target

            target_data = self.nodes_data.get(target)
            if target_data and target_data["type"] == "module":
                module = target_data["name"]
                qualified = ".".join(parts[1:])
                candidate = self._node_id(module, qualified)
                if candidate in self.nodes_data:
                    return candidate

        # 3. Appel via module importé
        if parts[0] in imported_modules:
            imported_module = imported_modules[parts[0]]
            remaining = ".".join(parts[1:])
            candidate = self._node_id(imported_module, remaining)
            if candidate in self.nodes_data:
                return candidate

        # 4. Appel local dans le même module
        local_candidate = self._node_id(module_name, called_name)
        if local_candidate in self.nodes_data:
            return local_candidate

        # 5. Recherche globale prudente par nom court :
        # UNIQUEMENT pour les fonctions ou classes directes (pas de résolutions d'attributs aléatoires)
        if len(parts) == 1:
            candidates = self.symbol_index.get(called_name, [])
            unique_candidates = list(dict.fromkeys(candidates))

            # Exclusion stricte : si le candidat unique est une méthode rattachée
            # à une classe sans appel qualifié, on rejette l'association arbitraire
            if len(unique_candidates) == 1:
                target_nid = unique_candidates[0]
                target_data = self.nodes_data.get(target_nid)
                if target_data and "." not in target_data["name"]:
                    return target_nid

        return None

    def _extract_relationships(self):
        self._build_symbol_index()

        for fname, info in self.file_asts.items():
            tree = info["tree"]
            module_name = info["module_name"]
            module_nid = self._node_id(module_name)

            is_test = Path(fname).name.startswith("test_") or "tests" in Path(fname).parts
            import_relation = "TESTE" if is_test else "IMPORTE"
            call_relation = "TESTE" if is_test else "APPELLE"

            visitor = ScopeVisitor(module_name)
            visitor.visit(tree)

            import_map = {}
            imported_modules = {}

            # Imports
            for item in visitor.imports:
                if item["kind"] == "import":
                    imported = item["module"]
                    local_name = item["alias"] or imported.split(".")[0]
                    imported_modules[local_name] = imported

                    if imported in self.module_paths:
                        target_nid = self._node_id(imported)
                        self.graph.add_edge(module_nid, target_nid, relation=import_relation)

                elif item["kind"] == "from":
                    imported_module = self._resolve_import_module(
                        module_name,
                        item["module"],
                        item.get("level", 0)
                    )
                    if not imported_module:
                        continue

                    imported_name = item["name"]
                    local_name = item["alias"] or imported_name

                    imported_symbol = self._node_id(imported_module, imported_name)
                    imported_module_node = self._node_id(imported_module)

                    if imported_symbol in self.nodes_data:
                        import_map[local_name] = imported_symbol
                        self.graph.add_edge(module_nid, imported_symbol, relation=import_relation)
                    elif imported_module in self.module_paths:
                        import_map[local_name] = imported_module_node
                        self.graph.add_edge(module_nid, imported_module_node, relation=import_relation)

            self.import_maps[module_name] = {
                "symbols": import_map,
                "modules": imported_modules
            }

            # Appels
            for call in visitor.calls:
                called_name = call["name"]
                if not called_name:
                    continue

                source_nid = self._node_id(
                    module_name,
                    call["source"] if call["source"] != f"{module_name}::module" else None
                )

                if source_nid not in self.graph:
                    source_nid = module_nid

                target_nid = self._resolve_call(called_name, source_nid, module_name)

                if target_nid and target_nid != source_nid:
                    self.graph.add_edge(
                        source_nid,
                        target_nid,
                        relation=call_relation,
                        line=call["line"]
                    )

    # ========================================================
    # INDEXATION
    # ========================================================

    def build_index(self):
        LOGGER.info("Découverte des fichiers...")
        py_files, md_files = self._discover_files()

        self.graph.clear()
        self.nodes_data.clear()
        self.file_asts.clear()
        self.module_paths.clear()
        self.import_maps.clear()
        self.parse_errors.clear()

        # Markdown
        for md_file in md_files:
            relative = md_file.relative_to(self.repo_path).as_posix()
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                LOGGER.warning("Lecture impossible : %s (%s)", relative, exc)
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
                identifiers=self._tokenize(content)
            )

        # Fichiers Python
        for py_file in py_files:
            relative = py_file.relative_to(self.repo_path).as_posix()
            module_name = self._module_name(py_file)

            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=relative)
            except (SyntaxError, UnicodeError, OSError) as exc:
                self.parse_errors.append({"file": relative, "error": str(exc)})
                LOGGER.warning("AST ignoré : %s (%s)", relative, exc)
                continue

            self.module_paths[module_name] = relative
            self.file_asts[relative] = {
                "tree": tree,
                "source": source,
                "module_name": module_name
            }
            self._extract_ast_nodes(tree, source, module_name, relative)

        self._extract_relationships()

        self.node_ids = list(self.nodes_data.keys())
        if not self.node_ids:
            self.bm25 = None
            self.embeddings = None
            self.indexed = False
            return

        # Construction index BM25
        tokenized_corpus = [
            self._tokenize(self.nodes_data[nid]["bm25_text"])
            for nid in self.node_ids
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Construction index vectoriel dense (via représentation sémantique compacte)
        dense_corpus = [
            self.nodes_data[nid]["dense_text"]
            for nid in self.node_ids
        ]

        LOGGER.info("Encodage vectoriel compact (%d nœuds)...", len(dense_corpus))
        self.embeddings = self.encoder.encode(
            dense_corpus,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

        self.indexed = True
        LOGGER.info(
            "Indexation V2 terminée : %d nœuds | %d relations",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges()
        )

    # ========================================================
    # RECHERCHE HYBRIDE & DÉDUPLICATION
    # ========================================================

    @staticmethod
    def _stable_ranks(scores):
        scores = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    def hybrid_search(self, query, top_k=5, rrf_k=60):
        if not self.indexed:
            raise RuntimeError("Exécuter build_index() avant la recherche.")

        q_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranks = self._stable_ranks(bm25_scores)

        q_emb = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        vec_scores = np.dot(self.embeddings, q_emb)
        vec_ranks = self._stable_ranks(vec_scores)

        rrf_scores = {}
        for idx, nid in enumerate(self.node_ids):
            score = (1.0 / (rrf_k + bm25_ranks[idx])) + (1.0 / (rrf_k + vec_ranks[idx]))
            rrf_scores[nid] = float(score)

        ranked = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def query_with_graph(self, query, top_k=5, max_neighbors=10):
        anchors = self.hybrid_search(query, top_k=top_k)
        context_bundle = []
        visited = set()

        def add_context(nid, role, score=None):
            if nid in visited or nid not in self.nodes_data:
                return
            visited.add(nid)
            context_bundle.append({"id": nid, "role": role, "score": score})

        for nid, score in anchors:
            add_context(nid, "ANCRAGE", score)
            neighbors = []

            for _, target, data in self.graph.out_edges(nid, data=True):
                if data.get("relation") != "CONTIENT":
                    neighbors.append((target, f"DÉPENDANCE : {data.get('relation')}"))

            for source, _, data in self.graph.in_edges(nid, data=True):
                if data.get("relation") != "CONTIENT":
                    neighbors.append((source, f"APPELANT : {data.get('relation')}"))

            for target, role in neighbors[:max_neighbors]:
                add_context(target, role)

        return context_bundle

    # ========================================================
    # DÉDUPLICATION HIÉRARCHIQUE & CONTEXTE LLM
    # ========================================================

    def deduplicate_hierarchy(self, context_bundle):
        """
        Supprime les nœuds parents (modules, classes) si un enfant
        plus spécifique est déjà retenu dans le même contexte.
        """
        selected_nids = [item["id"] for item in context_bundle]
        pruned_bundle = []

        for item in context_bundle:
            nid = item["id"]
            node_type = self.nodes_data[nid]["type"]

            if node_type == "module":
                module_prefix = nid.replace("::module", "::")
                has_child = any(other != nid and other.startswith(module_prefix) for other in selected_nids)
                if has_child:
                    continue

            elif node_type == "class":
                class_prefix = f"{nid}."
                has_child = any(other.startswith(class_prefix) for other in selected_nids)
                if has_child:
                    continue

            pruned_bundle.append(item)

        return pruned_bundle

    def format_prompt_context(self, query, top_k=5, max_neighbors=10, max_chars=None):
        raw_bundle = self.query_with_graph(query, top_k=top_k, max_neighbors=max_neighbors)
        pruned_bundle = self.deduplicate_hierarchy(raw_bundle)

        blocks = [
            "=== CONTEXTE DE RECHERCHE HYBRIDE (V2) ===",
            f"REQUÊTE : {query}",
            f"NŒUDS SÉLECTIONNÉS : {len(pruned_bundle)}",
            ""
        ]

        total_chars = sum(len(b) for b in blocks)

        for item in pruned_bundle:
            nid = item["id"]
            data = self.nodes_data[nid]

            header = f"--- [{item['role']}] {nid} ---"
            if item["score"] is not None:
                header += f" [RRF={item['score']:.6f}]"

            content = (
                f"{header}\n"
                f"TYPE : {data['type']}\n"
                f"FICHIER : {data['file']}\n"
                f"LIGNE : {data['line']}\n"
                f"SIGNATURE : {data['signature']}\n"
                f"{data['code']}\n"
            )

            if max_chars is not None and total_chars + len(content) > max_chars:
                blocks.append("\n[CONTEXTE TRONQUÉ : limite de caractères atteinte]")
                break

            blocks.append(content)
            total_chars += len(content)

        return "\n".join(blocks)


# ============================================================
# TESTS DE NON-RÉGRESSION UNITAIRES
# ============================================================

class TestRepoHybridGraphRAGV2(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_self_and_cls_resolution(self):
        code = (
            "class Worker:\n"
            "    def step(self):\n"
            "        self.run()\n"
            "    def run(self):\n"
            "        pass\n"
        )
        (self.repo_path / "worker.py").write_text(code, encoding="utf-8")

        rag = RepoHybridGraphRAG(repo_path=self.repo_path, lazy_encoder=True)
        rag.build_index()

        step_nid = "worker::Worker.step"
        run_nid = "worker::Worker.run"

        self.assertIn(step_nid, rag.graph)
        self.assertIn(run_nid, rag.graph)
        self.assertTrue(rag.graph.has_edge(step_nid, run_nid))
        self.assertEqual(rag.graph[step_nid][run_nid]["relation"], "APPELLE")

    def test_homonym_collision_rejection(self):
        code_a = (
            "class EngineA:\n"
            "    def execute(self):\n"
            "        pass\n"
        )
        code_b = (
            "class EngineB:\n"
            "    def execute(self):\n"
            "        pass\n"
        )
        code_caller = (
            "def launch():\n"
            "    execute()\n"
        )
        (self.repo_path / "a.py").write_text(code_a, encoding="utf-8")
        (self.repo_path / "b.py").write_text(code_b, encoding="utf-8")
        (self.repo_path / "caller.py").write_text(code_caller, encoding="utf-8")

        rag = RepoHybridGraphRAG(repo_path=self.repo_path, lazy_encoder=True)
        rag.build_index()

        caller_nid = "caller::launch"
        # Il ne doit pas créer de relation arbitraire vers l'une des méthodes execute()
        out_edges = list(rag.graph.out_edges(caller_nid))
        self.assertEqual(len(out_edges), 0)

    def test_hierarchical_deduplication(self):
        code = (
            "class Pipeline:\n"
            "    def process(self):\n"
            "        return 42\n"
        )
        (self.repo_path / "pipe.py").write_text(code, encoding="utf-8")

        rag = RepoHybridGraphRAG(repo_path=self.repo_path, lazy_encoder=True)
        rag.build_index()

        bundle = [
            {"id": "pipe::module", "role": "ANCRAGE", "score": 0.05},
            {"id": "pipe::Pipeline", "role": "ANCRAGE", "score": 0.04},
            {"id": "pipe::Pipeline.process", "role": "ANCRAGE", "score": 0.03}
        ]

        pruned = rag.deduplicate_hierarchy(bundle)
        pruned_ids = [p["id"] for p in pruned]

        # Le module et la classe doivent être retirés car la méthode process est présente
        self.assertEqual(pruned_ids, ["pipe::Pipeline.process"])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRepoHybridGraphRAGV2)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
