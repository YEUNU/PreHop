#!/usr/bin/env python3
"""JSON-line bridge into pinned BrowseNet and PropRAG checkouts.

This file intentionally contains no copied upstream implementation.  It loads
the official checkout selected by the parent process and calls its public
index/retrieval functions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any

RESULT_PREFIX = "__PREHOP_OFFICIAL_RESULT__="

# Executing this file directly places the repository's ``scripts`` directory
# first on sys.path. Its local ``datasets`` package would otherwise shadow the
# Hugging Face dependency imported by upstream model libraries.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _SCRIPT_DIR]


def emit(payload: dict[str, Any]) -> None:
    print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def load_corpus(output_dir: Path) -> list[dict[str, Any]]:
    rows = json.loads((output_dir / "input" / "corpus.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Staged corpus is empty or malformed")
    return rows


def _browsenet_dataset_key(corpus_tag: str) -> str:
    # BrowseNet chooses its official few-shot decomposition prompt by a
    # substring in the dataset name. MultiHop-RAG is closest to its HotpotQA
    # branch; MuSiQue uses its native branch.
    return f"musique_{corpus_tag}" if "musique" in corpus_tag.casefold() else f"hotpotqa_{corpus_tag}"


def _patch_browsenet_storage(official_root: Path, output_dir: Path) -> None:
    root_text = str(official_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from src import BrowseNet as browse_module
    from src.indexer import NER, colbertv2_knn, kg_construct
    from src.retriever import subquerygeneration

    for module in (browse_module, NER, colbertv2_knn, kg_construct, subquerygeneration):
        module.ROOT_DIR = output_dir
    colbert_runtime = output_dir / "colbert_runtime"
    (colbert_runtime / "colbert").mkdir(parents=True, exist_ok=True)
    (colbert_runtime / "exp").mkdir(parents=True, exist_ok=True)
    runtime_checkpoint = colbert_runtime / "exp" / "colbertv2.0"
    official_checkpoint = official_root / "src" / "indexer" / "exp" / "colbertv2.0"
    if not runtime_checkpoint.exists():
        runtime_checkpoint.symlink_to(official_checkpoint, target_is_directory=True)
    colbertv2_knn.FILE_DIR = colbert_runtime
    colbertv2_knn.FILE_DIR_PARENT = output_dir


def browsenet_index(official_root: Path, output_dir: Path, corpus_tag: str) -> dict[str, Any]:
    import torch

    rows = load_corpus(output_dir)
    dataset = _browsenet_dataset_key(corpus_tag)
    dataset_dir = output_dir / "datasets" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts" / dataset).mkdir(parents=True, exist_ok=True)
    (output_dir / "results" / dataset).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "corpus.json").write_text(
        json.dumps([{"title": row["title"], "text": row["text"]} for row in rows], ensure_ascii=False),
        encoding="utf-8",
    )
    _patch_browsenet_storage(official_root, output_dir)
    from src.BrowseNet import BrowseNet

    engine = BrowseNet(
        dataset=dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ner_model=os.environ.get("RAG_BROWSENET_NER_MODEL", "gliner"),
        sem_model=os.environ.get("RAG_BROWSENET_SEM_MODEL", "nvembedv2"),
        subquery_model=os.environ.get(
            "RAG_BROWSENET_SUBQUERY_MODEL", os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
        ),
        colbert_threshold=float(os.environ.get("RAG_BROWSENET_COLBERT_THRESHOLD", "0.9")),
        n_subgraphs=int(os.environ.get("RAG_BROWSENET_N_SUBGRAPHS", "5")),
        alpha=float(os.environ.get("RAG_BROWSENET_ALPHA", "0.0")),
    )
    engine.index()
    graph = engine.KG
    return {"documents": len(rows), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}


class BrowseNetService:
    def __init__(self, official_root: Path, output_dir: Path, corpus_tag: str):
        import torch

        self.output_dir = output_dir
        self.corpus_tag = corpus_tag
        self.rows = load_corpus(output_dir)
        self.dataset = _browsenet_dataset_key(corpus_tag)
        _patch_browsenet_storage(official_root, output_dir)
        from src.BrowseNet import BrowseNet

        self.engine = BrowseNet(
            dataset=self.dataset,
            device="cuda" if torch.cuda.is_available() else "cpu",
            ner_model=os.environ.get("RAG_BROWSENET_NER_MODEL", "gliner"),
            sem_model=os.environ.get("RAG_BROWSENET_SEM_MODEL", "nvembedv2"),
            subquery_model=os.environ.get(
                "RAG_BROWSENET_SUBQUERY_MODEL", os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model")
            ),
            colbert_threshold=float(os.environ.get("RAG_BROWSENET_COLBERT_THRESHOLD", "0.9")),
            n_subgraphs=int(os.environ.get("RAG_BROWSENET_N_SUBGRAPHS", "5")),
            alpha=float(os.environ.get("RAG_BROWSENET_ALPHA", "0.0")),
        )
        # Calling index is the official loading path. Every construction stage
        # is cache-aware and therefore only validates/loads completed artifacts.
        self.engine.index()

    def query(self, question: str) -> list[dict[str, Any]]:
        from openai import OpenAI
        from src.prompts.sq_gen_prompts import hotpot_few_shot_demo, init_prompt, musique_few_shot_demo
        from src.retriever.retrievers import browsenet_retriever
        from src.retriever.subquerygeneration import query_openai_model

        prompt = f"{init_prompt}\n{musique_few_shot_demo if 'musique' in self.dataset else hotpot_few_shot_demo}"
        model = self.engine.subquery_model
        client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            base_url=os.environ.get("VLLM_URL") or os.environ.get("OPENAI_BASE_URL"),
        )
        response = query_openai_model(prompt, question, client, model)
        if response is None or not response.choices or not response.choices[0].message.content:
            raise RuntimeError("BrowseNet subquery generation returned no content")
        split_queries = {question: response.choices[0].message.content}
        ids = browsenet_retriever(
            [{"question": question}],
            split_queries,
            self.engine.KG,
            self.engine.chunk_embs,
            self.engine.encoder,
            n_subgraphs=self.engine.n_subgraphs,
            corp_text=self.engine.corp_text,
            hybrid_alpha=self.engine.alpha,
        )[0]
        return [
            {
                "source_id": self.rows[int(idx)]["source_id"],
                "title": self.rows[int(idx)]["title"],
                "text": self.rows[int(idx)]["text"],
            }
            for idx in ids
        ]


def _patch_proprag_config(official_root: Path) -> tuple[Any, Any]:
    # Upstream imports both optional offline engines at package-import time.
    # This adapter uses its online OpenAI-compatible path, so keep those
    # imports inert instead of installing two additional model runtimes.
    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        vllm.SamplingParams = type("SamplingParams", (), {})
        vllm.LLM = type("LLM", (), {})
        sys.modules["vllm"] = vllm
    if "gritlm" not in sys.modules:
        gritlm = types.ModuleType("gritlm")
        gritlm.GritLM = type("GritLM", (), {})
        sys.modules["gritlm"] = gritlm
    source = official_root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from proprag.PropRAG import PropRAG
    from proprag.utils.config_utils import BaseConfig

    def _post_init(config):
        if config.save_dir is None:
            config.save_dir = "outputs" if config.dataset is None else os.path.join("outputs", config.dataset)

    BaseConfig.__post_init__ = _post_init
    return PropRAG, BaseConfig


def _proprag_config(BaseConfig: Any, output_dir: Path, corpus_len: int):
    return BaseConfig(
        api_key=os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
        llm_base_url=os.environ.get("VLLM_URL") or os.environ.get("OPENAI_BASE_URL"),
        llm_name=os.environ.get("VLLM_SERVED_MODEL_NAME", "generation-model"),
        dataset=None,
        save_dir=str(output_dir / "outputs"),
        embedding_model_name=os.environ.get("RAG_PROPRAG_EMBEDDING_MODEL", "nvidia/NV-Embed-v2"),
        force_index_from_scratch=False,
        force_openie_from_scratch=False,
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=4,
        beam_width=4,
        max_path_length=3,
        second_stage_filter_k=40,
        corpus_len=corpus_len,
        openie_mode="online",
        use_propositions=True,
    )


def proprag_index(official_root: Path, output_dir: Path) -> dict[str, Any]:
    rows = load_corpus(output_dir)
    PropRAG, BaseConfig = _patch_proprag_config(official_root)
    engine = PropRAG(global_config=_proprag_config(BaseConfig, output_dir, len(rows)))
    engine.index([f"{row['title']}\n{row['text']}" for row in rows])
    return {"documents": len(rows), "nodes": engine.graph.vcount(), "edges": engine.graph.ecount()}


class PropRAGService:
    def __init__(self, official_root: Path, output_dir: Path):
        self.rows = load_corpus(output_dir)
        self.by_text: dict[str, list[dict[str, Any]]] = {}
        for row in self.rows:
            self.by_text.setdefault(f"{row['title']}\n{row['text']}", []).append(row)
        PropRAG, BaseConfig = _patch_proprag_config(official_root)
        self.engine = PropRAG(global_config=_proprag_config(BaseConfig, output_dir, len(self.rows)))
        # The official example indexes and queries on the same object. Several
        # proposition maps therefore live only in memory even though the graph,
        # OpenIE rows, and embeddings are persisted. Replaying the official
        # cache-aware index entry point restores those maps without new model
        # calls and preserves the published retrieval path across processes.
        self.engine.index([f"{row['title']}\n{row['text']}" for row in self.rows])

    def query(self, question: str) -> list[dict[str, Any]]:
        solution = self.engine.retrieve([question], num_to_retrieve=200)[0]
        scores = solution.doc_scores.tolist() if hasattr(solution.doc_scores, "tolist") else list(solution.doc_scores)
        documents = []
        for rank, text in enumerate(solution.docs):
            matches = self.by_text.get(text, [])
            if not matches:
                raise RuntimeError("PropRAG returned a passage outside the staged corpus")
            row = matches[0]
            documents.append(
                {
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "text": row["text"],
                    "score": float(scores[rank]) if rank < len(scores) else None,
                }
            )
        return documents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("browsenet", "proprag"), required=True)
    parser.add_argument("--mode", choices=("index", "serve"), required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-tag", required=True)
    args = parser.parse_args()
    try:
        if args.mode == "index":
            request = json.loads(sys.stdin.readline())
            if request.get("operation") != "index":
                raise ValueError("Index worker expected operation=index")
            stats = (
                browsenet_index(args.official_root, args.output_dir, args.corpus_tag)
                if args.strategy == "browsenet"
                else proprag_index(args.official_root, args.output_dir)
            )
            emit({"ok": True, "stats": stats})
            return 0

        service = (
            BrowseNetService(args.official_root, args.output_dir, args.corpus_tag)
            if args.strategy == "browsenet"
            else PropRAGService(args.official_root, args.output_dir)
        )
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request.get("operation")
                if operation == "ready":
                    emit({"ok": True, "ready": True})
                elif operation == "query":
                    emit({"ok": True, "documents": service.query(str(request["query"]))})
                elif operation == "shutdown":
                    emit({"ok": True})
                    return 0
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            except Exception as exc:  # noqa: BLE001 - query failures must not kill the persistent worker
                traceback.print_exc(file=sys.stderr)
                emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 0
    except Exception as exc:  # noqa: BLE001 - serialize startup/index failures across the process boundary
        traceback.print_exc(file=sys.stderr)
        emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
