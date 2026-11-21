
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
REPO HYBRID GRAPH RAG
=====================

Moteur de recherche hybride pour dépôts Python.

Architecture :
    1. Extraction AST récursive.
    2. Graphe orienté de dépendances.
    3. Identification des modules, classes, fonctions et méthodes.
    4. Extraction des imports et appels.
    5. Indexation lexicale BM25.
    6. Indexation vectorielle Sentence Transformers.
    7. Fusion Reciprocal Rank Fusion (RRF).
    8. Expansion topologique du contexte.
    9. Construction d'un contexte fermé pour LLM.

Dépendances :
    pip install numpy networkx rank-bm25 sentence-transformers

Modèle par défaut :
    all-MiniLM-L6-v2

Limites :
    - Résolution statique des appels Python.
    - Les appels dynamiques et réflexifs ne sont pas garantis.
    - Les imports conditionnels peuvent être détectés sans
      être nécessairement exécutés.
"""

import ast
import re
import logging

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
# OUTIL AST : EXTRACTION DES PORTÉES
# ============================================================

class ScopeVisitor(ast.NodeVisitor):
    """
    Parcourt un module et conserve la portée courante.

    Chaque appel est associé à la fonction, méthode ou classe
    dans laquelle il apparaît.
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
            "line": node.lineno
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
        batch_size=32
    ):

        self.repo_path = Path(repo_path).resolve()

        if not self.repo_path.is_dir():
            raise NotADirectoryError(
                f"Dépôt introuvable : {self.repo_path}"
            )

        self.embed_model = embed_model
        self.batch_size = batch_size

        self.excluded_dirs = (
            IGNORED_DIRS | set(excluded_dirs or [])
        )

        # Graphe
        self.graph = nx.DiGraph()

        # Données des nœuds
        self.nodes_data = {}

        # Ordre stable pour les index
        self.node_ids = []

        # Index lexicaux et vectoriels
        self.bm25 = None
        self.embeddings = None

        # Métadonnées
        self.file_asts = {}
        self.module_paths = {}
        self.symbol_index = defaultdict(list)
        self.import_maps = {}

        self.parse_errors = []

        self.indexed = False

        LOGGER.info("Chargement du modèle : %s", embed_model)

        self.encoder = SentenceTransformer(embed_model)

    # ========================================================
    # TOKENISATION
    # ========================================================

    def _tokenize(self, text):

        if not isinstance(text, str):
            return []

        tokens = re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower()
        )

        expanded = []

        for token in tokens:

            expanded.append(token)

            # snake_case
            if "_" in token:
                expanded.extend(token.split("_"))

            # CamelCase
            camel_parts = re.findall(
                r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+",
                token
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

        return (
            sorted(py_files),
            sorted(md_files)
        )

    # ========================================================
    # NOMMAGE DES MODULES
    # ========================================================

    def _module_name(self, path):

        relative = path.relative_to(self.repo_path)

        parts = list(relative.with_suffix("").parts)

        if parts[-1] == "__init__":
            parts.pop()

        return ".".join(parts)

    # ========================================================
    # IDENTIFIANTS DES NŒUDS
    # ========================================================

    @staticmethod
    def _node_id(module_name, qualified_name=None):

        if qualified_name is None:
            return f"{module_name}::module"

        return f"{module_name}::{qualified_name}"

    # ========================================================
    # AJOUT D'UN NŒUD
    # ========================================================

    def _add_node(
        self,
        nid,
        file,
        name,
        node_type,
        code,
        doc="",
        line=None
    ):

        if nid in self.nodes_data:
            return

        text_for_search = (
            f"{file}\n"
            f"{name}\n"
            f"{node_type}\n"
            f"{doc}\n"
            f"{code}"
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
            "code": code,
            "doc": doc,
            "line": line,
            "text_for_search": text_for_search
        }

    # ========================================================
    # EXTRACTION DES NŒUDS AST
    # ========================================================

    def _extract_ast_nodes(self, tree, source, module_name, filename):

        module_nid = self._node_id(module_name)

        self._add_node(
            nid=module_nid,
            file=filename,
            name=module_name,
            node_type="module",
            code=source,
            doc=ast.get_docstring(tree) or ""
        )

        def visit_scope(node, parent_qualified=None):

            if isinstance(node, ast.ClassDef):
                node_type = "class"

            elif isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                node_type = "function"

            else:
                return

            if parent_qualified:
                qualified = f"{parent_qualified}.{node.name}"
            else:
                qualified = node.name

            nid = self._node_id(module_name, qualified)

            segment = (
                ast.get_source_segment(source, node) or ""
            )

            doc = ast.get_docstring(node) or ""

            self._add_node(
                nid=nid,
                file=filename,
                name=qualified,
                node_type=node_type,
                code=segment,
                doc=doc,
                line=getattr(node, "lineno", None)
            )

            parent_nid = (
                self._node_id(module_name, parent_qualified)
                if parent_qualified
                else module_nid
            )

            self.graph.add_edge(
                parent_nid,
                nid,
                relation="CONTIENT"
            )

            # Les classes peuvent contenir des méthodes.
            # Les fonctions peuvent contenir des fonctions imbriquées.
            for child in node.body:
                if isinstance(
                    child,
                    (
                        ast.ClassDef,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef
                    )
                ):
                    visit_scope(child, qualified)

        for node in tree.body:

            if isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef
                )
            ):
                visit_scope(node)

    # ========================================================
    # RÉSOLUTION DES IMPORTS RELATIFS
    # ========================================================

    def _resolve_import_module(self, module_name, imported_module, level):

        if level == 0:
            return imported_module

        package = module_name.rpartition(".")[0]

        if module_name.endswith(".__init__"):
            package = module_name.rpartition(".")[0]

        relative = "." * level + imported_module

        try:
            import importlib.util

            return importlib.util.resolve_name(
                relative,
                package
            )

        except (ImportError, ValueError):
            return None

    # ========================================================
    # CONSTRUCTION DE L'INDEX DES SYMBOLES
    # ========================================================

    def _build_symbol_index(self):

        self.symbol_index.clear()

        for nid, data in self.nodes_data.items():

            if data["type"] not in ("function", "class"):
                continue

            short_name = data["name"].split(".")[-1]

            self.symbol_index[short_name].append(nid)

            self.symbol_index[data["name"]].append(nid)

    # ========================================================
    # EXTRACTION DES IMPORTS ET DES APPELS
    # ========================================================

    def _extract_relationships(self):

        self._build_symbol_index()

        for fname, info in self.file_asts.items():

            tree = info["tree"]
            module_name = info["module_name"]

            module_nid = self._node_id(module_name)

            is_test = (
                Path(fname).name.startswith("test_")
                or "tests" in Path(fname).parts
            )

            import_relation = (
                "TESTE" if is_test else "IMPORTE"
            )

            call_relation = (
                "TESTE" if is_test else "APPELLE"
            )

            visitor = ScopeVisitor(module_name)
            visitor.visit(tree)

            import_map = {}
            imported_modules = {}

            # ---------------- IMPORTS ----------------

            for item in visitor.imports:

                if item["kind"] == "import":

                    imported = item["module"]

                    local_name = (
                        item["alias"]
                        or imported.split(".")[0]
                    )

                    imported_modules[local_name] = imported

                    target = self.module_paths.get(imported)

                    if target:
                        target_nid = self._node_id(imported)

                        self.graph.add_edge(
                            module_nid,
                            target_nid,
                            relation=import_relation
                        )

                elif item["kind"] == "from":

                    imported_module = self._resolve_import_module(
                        module_name,
                        item["module"],
                        item.get("level", 0)
                    )

                    if not imported_module:
                        continue

                    imported_name = item["name"]

                    local_name = (
                        item["alias"] or imported_name
                    )

                    # Import de module ou de symbole.
                    imported_symbol = self._node_id(
                        imported_module,
                        imported_name
                    )

                    imported_module_node = self._node_id(
                        imported_module
                    )

                    if imported_symbol in self.nodes_data:

                        import_map[local_name] = imported_symbol

                        self.graph.add_edge(
                            module_nid,
                            imported_symbol,
                            relation=import_relation
                        )

                    elif imported_module in self.module_paths:

                        import_map[local_name] = imported_module_node

                        self.graph.add_edge(
                            module_nid,
                            imported_module_node,
                            relation=import_relation
                        )

            self.import_maps[module_name] = {
                "symbols": import_map,
                "modules": imported_modules
            }

            # ---------------- APPELS ----------------

            for call in visitor.calls:

                called_name = call["name"]

                if not called_name:
                    continue

                source_nid = self._node_id(
                    module_name,
                    call["source"]
                    if call["source"] != f"{module_name}::module"
                    else None
                )

                if source_nid not in self.graph:
                    source_nid = module_nid

                target_nid = self._resolve_call(
                    called_name,
                    module_name
                )

                if target_nid and target_nid != source_nid:

                    self.graph.add_edge(
                        source_nid,
                        target_nid,
                        relation=call_relation,
                        line=call["line"]
                    )

    # ========================================================
    # RÉSOLUTION PRUDENTE DES APPELS
    # ========================================================

    def _resolve_call(self, called_name, module_name):

        if not called_name:
            return None

        parts = called_name.split(".")

        imports = self.import_maps.get(
            module_name,
            {"symbols": {}, "modules": {}}
        )

        imported_symbols = imports["symbols"]
        imported_modules = imports["modules"]

        # 1. Appel via un symbole importé.
        if parts[0] in imported_symbols:

            target = imported_symbols[parts[0]]

            if len(parts) == 1:
                return target

            target_data = self.nodes_data.get(target)

            if target_data and target_data["type"] == "module":

                module = target_data["name"]

                qualified = ".".join(parts[1:])

                candidate = self._node_id(
                    module,
                    qualified
                )

                if candidate in self.nodes_data:
                    return candidate

        # 2. Appel via un module importé.
        if parts[0] in imported_modules:

            imported_module = imported_modules[parts[0]]

            remaining = ".".join(parts[1:])

            candidate = self._node_id(
                imported_module,
                remaining
            )

            if candidate in self.nodes_data:
                return candidate

        # 3. Symbole du même module.
        local_candidate = self._node_id(
            module_name,
            called_name
        )

        if local_candidate in self.nodes_data:
            return local_candidate

        # 4. Recherche par nom court.
        # Résolution uniquement si la correspondance est unique.
        short_name = parts[-1]

        candidates = self.symbol_index.get(
            short_name,
            []
        )

        unique = list(dict.fromkeys(candidates))

        if len(unique) == 1:
            return unique[0]

        # Ambiguïté : aucune arête arbitraire.
        return None

    # ========================================================
    # INDEXATION PRINCIPALE
    # ========================================================

    def build_index(self):

        LOGGER.info("Découverte des fichiers...")

        py_files, md_files = self._discover_files()

        LOGGER.info(
            "%d fichiers Python | %d fichiers Markdown",
            len(py_files),
            len(md_files)
        )

        self.graph.clear()
        self.nodes_data.clear()
        self.file_asts.clear()
        self.module_paths.clear()
        self.import_maps.clear()
        self.parse_errors.clear()

        # ---------------- MARKDOWN ----------------

        for md_file in md_files:

            relative = md_file.relative_to(
                self.repo_path
            ).as_posix()

            try:
                content = md_file.read_text(
                    encoding="utf-8",
                    errors="replace"
                )
            except OSError as exc:
                LOGGER.warning(
                    "Lecture impossible : %s (%s)",
                    relative,
                    exc
                )
                continue

            nid = f"{relative}::doc"

            self._add_node(
                nid=nid,
                file=relative,
                name=relative,
                node_type="documentation",
                code=content
            )

        # ---------------- AST ----------------

        for py_file in py_files:

            relative = py_file.relative_to(
                self.repo_path
            ).as_posix()

            module_name = self._module_name(py_file)

            try:
                source = py_file.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                tree = ast.parse(
                    source,
                    filename=relative
                )

            except (SyntaxError, UnicodeError, OSError) as exc:

                self.parse_errors.append({
                    "file": relative,
                    "error": str(exc)
                })

                LOGGER.warning(
                    "AST ignoré : %s (%s)",
                    relative,
                    exc
                )
                continue

            self.module_paths[module_name] = relative

            self.file_asts[relative] = {
                "tree": tree,
                "source": source,
                "module_name": module_name
            }

            self._extract_ast_nodes(
                tree,
                source,
                module_name,
                relative
            )

        # ---------------- RELATIONS ----------------

        self._extract_relationships()

        # ---------------- INDEX BM25 ----------------

        self.node_ids = list(self.nodes_data.keys())

        corpus_texts = [
            self.nodes_data[nid]["text_for_search"]
            for nid in self.node_ids
        ]

        if not corpus_texts:
            self.bm25 = None
            self.embeddings = None
            self.indexed = False

            LOGGER.warning("Aucun document indexable.")
            return

        tokenized_corpus = [
            self._tokenize(text)
            for text in corpus_texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

        # ---------------- INDEX VECTORIEL ----------------

        LOGGER.info("Calcul des embeddings...")

        self.embeddings = self.encoder.encode(
            corpus_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        self.embeddings = np.asarray(
            self.embeddings,
            dtype=np.float32
        )

        self.indexed = True

        LOGGER.info(
            "INDEXATION TERMINÉE : %d nœuds | %d relations | %d erreurs AST",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self.parse_errors)
        )

    # ========================================================
    # CLASSEMENT STABLE
    # ========================================================

    @staticmethod
    def _stable_ranks(scores):

        scores = np.asarray(
            scores,
            dtype=np.float64
        )

        order = np.argsort(
            -scores,
            kind="stable"
        )

        ranks = np.empty(
            len(scores),
            dtype=np.int64
        )

        ranks[order] = np.arange(
            1,
            len(scores) + 1
        )

        return ranks

    # ========================================================
    # RECHERCHE HYBRIDE
    # ========================================================

    def hybrid_search(
        self,
        query,
        top_k=5,
        rrf_k=60
    ):

        if not self.indexed:
            raise RuntimeError(
                "Exécuter build_index() avant la recherche."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "La requête doit être une chaîne non vide."
            )

        if top_k < 1:
            raise ValueError("top_k doit être positif.")

        if rrf_k < 1:
            raise ValueError("rrf_k doit être positif.")

        # BM25
        q_tokens = self._tokenize(query)

        bm25_scores = self.bm25.get_scores(
            q_tokens
        )

        bm25_ranks = self._stable_ranks(
            bm25_scores
        )

        # Vectoriel
        q_emb = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        vec_scores = np.dot(
            self.embeddings,
            q_emb
        )

        vec_ranks = self._stable_ranks(
            vec_scores
        )

        # RRF
        rrf_scores = {}

        for idx, nid in enumerate(self.node_ids):

            score = (
                1.0 / (rrf_k + bm25_ranks[idx])
                +
                1.0 / (rrf_k + vec_ranks[idx])
            )

            rrf_scores[nid] = float(score)

        ranked = sorted(
            rrf_scores.items(),
            key=lambda item: (
                -item[1],
                item[0]
            )
        )

        return ranked[:top_k]

    # ========================================================
    # EXPANSION DU GRAPHE
    # ========================================================

    def query_with_graph(
        self,
        query,
        top_k=5,
        max_neighbors=10
    ):

        anchors = self.hybrid_search(
            query,
            top_k=top_k
        )

        context_bundle = []
        visited = set()

        def add_context(nid, role, score=None):

            if nid in visited:
                return

            if nid not in self.nodes_data:
                return

            visited.add(nid)

            context_bundle.append({
                "id": nid,
                "role": role,
                "score": score
            })

        for nid, score in anchors:

            add_context(
                nid,
                "ANCRAGE",
                score
            )

            neighbors = []

            # Dépendances sortantes.
            for _, target, data in self.graph.out_edges(
                nid,
                data=True
            ):

                if data.get("relation") != "CONTIENT":

                    neighbors.append((
                        target,
                        f"DÉPENDANCE : {data.get('relation')}"
                    ))

            # Appelants et validations entrantes.
            for source, _, data in self.graph.in_edges(
                nid,
                data=True
            ):

                if data.get("relation") != "CONTIENT":

                    neighbors.append((
                        source,
                        f"APPELANT / VALIDATION : {data.get('relation')}"
                    ))

            for target, role in neighbors[:max_neighbors]:

                add_context(
                    target,
                    role
                )

        return context_bundle

    # ========================================================
    # CONSTRUCTION DU CONTEXTE LLM
    # ========================================================

    def format_prompt_context(
        self,
        query,
        top_k=5,
        max_neighbors=10,
        max_chars=None
    ):

        results = self.query_with_graph(
            query,
            top_k=top_k,
            max_neighbors=max_neighbors
        )

        blocks = [
            "=== CONTEXTE DE RECHERCHE HYBRIDE ===",
            f"REQUÊTE : {query}",
            f"NŒUDS RETENUS : {len(results)}",
            ""
        ]

        total_chars = sum(
            len(block) for block in blocks
        )

        for item in results:

            nid = item["id"]
            data = self.nodes_data[nid]

            header = (
                f"--- [{item['role']}] {nid} ---"
            )

            if item["score"] is not None:
                header += (
                    f" [RRF={item['score']:.6f}]"
                )

            content = (
                f"{header}\n"
                f"TYPE : {data['type']}\n"
                f"FICHIER : {data['file']}\n"
                f"LIGNE : {data['line']}\n"
                f"{data['code']}\n"
            )

            if (
                max_chars is not None
                and total_chars + len(content) > max_chars
            ):
                blocks.append(
                    "\n[CONTEXTE TRONQUÉ : limite atteinte]"
                )
                break

            blocks.append(content)
            total_chars += len(content)

        return "\n".join(blocks)

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    def diagnostics(self):

        return {
            "repository": str(self.repo_path),
            "indexed": self.indexed,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "modules": len(self.file_asts),
            "symbols": sum(
                1
                for data in self.nodes_data.values()
                if data["type"] in ("class", "function")
            ),
            "documentation": sum(
                1
                for data in self.nodes_data.values()
                if data["type"] == "documentation"
            ),
            "parse_errors": self.parse_errors,
            "embedding_shape": (
                tuple(self.embeddings.shape)
                if self.embeddings is not None
                else None
            )
        }


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":

    rag = RepoHybridGraphRAG(
        repo_path=".",
        embed_model="all-MiniLM-L6-v2"
    )

    rag.build_index()

    print("\n=== DIAGNOSTIC ===")

    for key, value in rag.diagnostics().items():
        print(f"{key}: {value}")

    if rag.indexed:

        requete = (
            "Comment slh_solver et wavepackets "
            "utilisent-ils la calibration ?"
        )

        print("\n=== RECHERCHE HYBRIDE ===")

        resultats = rag.hybrid_search(
            requete,
            top_k=5
        )

        for nid, score in resultats:
            print(f"{score:.6f} | {nid}")

        print("\n=== CONTEXTE GRAPHIQUE ===")

        contexte = rag.format_prompt_context(
            requete,
            top_k=5,
            max_neighbors=10
        )

        print(contexte)
