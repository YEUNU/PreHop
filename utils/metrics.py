import logging
import re
import string
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from core.config import RAGConfig
from utils.formatters import format_context_from_nodes
from utils.prompts import MULTIHOPRAG_JUDGE_PROMPT

logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_LABEL_RE = re.compile(r"(?is)(?:final\s+answer|@@ANSWER|answer)\s*:?\s*(.+?)(?:\n\n|\Z)")


def normalize_answer(s):
    """Normalize answer text for comparison."""
    if not s:
        return ""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_final_answer(answer_text: str) -> str:
    """Extract the answer span used by deterministic answer metrics.

    Benchmark responses may contain a rationale before a ``Final Answer:`` or
    ``\\boxed{...}`` marker. The deterministic metrics must score the same
    final span that the shared answer-label logic inspects, rather than the
    entire response body.
    """
    if not answer_text:
        return ""
    boxed = _BOXED_RE.findall(answer_text)
    if boxed:
        return boxed[-1].strip()
    matches = _FINAL_LABEL_RE.findall(answer_text)
    if matches:
        return matches[-1].strip()[:400]
    return answer_text[-300:].strip()


def _answer_overlap_metrics(prediction: str, reference: str) -> tuple[float, float, float, float]:
    """Return EM, token-F1, precision, and recall after normalization."""
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    if not pred_norm or not ref_norm:
        return (1.0, 1.0, 1.0, 1.0) if pred_norm == ref_norm else (0.0, 0.0, 0.0, 0.0)
    if pred_norm == ref_norm:
        return 1.0, 1.0, 1.0, 1.0

    pred_tokens = pred_norm.split()
    ref_tokens = ref_norm.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0, 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return 0.0, f1, precision, recall


def calculate_answer_metrics(
    response: str,
    ground_truth: str,
    answer_aliases: list[str] | None = None,
    question_type: str = "",
) -> dict[str, Any]:
    """Calculate deterministic answer metrics with optional aliases.

    Null queries do not have a substantive answer target for EM/F1. Their
    answer metrics are therefore ``UNJUDGED_SCORE`` and their honest-refusal
    outcome is reported separately as ``null_refusal``.
    """
    final_answer = extract_final_answer(response)
    is_null = str(question_type or "").strip().lower() == "null_query"
    if is_null:
        return {
            "final_answer_extracted": final_answer[:400],
            "answer_em": UNJUDGED_SCORE,
            "answer_f1": UNJUDGED_SCORE,
            "answer_precision": UNJUDGED_SCORE,
            "answer_recall": UNJUDGED_SCORE,
            "null_refusal": 1.0 if _is_insufficient_text(final_answer) else 0.0,
        }

    references = [str(ground_truth or "").strip()]
    references.extend(str(alias).strip() for alias in (answer_aliases or []) if str(alias).strip())
    references = list(dict.fromkeys(reference for reference in references if reference))
    if not references:
        return {
            "final_answer_extracted": final_answer[:400],
            "answer_em": UNJUDGED_SCORE,
            "answer_f1": UNJUDGED_SCORE,
            "answer_precision": UNJUDGED_SCORE,
            "answer_recall": UNJUDGED_SCORE,
            "null_refusal": UNJUDGED_SCORE,
        }

    scores = [_answer_overlap_metrics(final_answer, reference) for reference in references]
    best = max(scores, key=lambda values: (values[1], values[0], values[2], values[3]))
    return {
        "final_answer_extracted": final_answer[:400],
        "answer_em": best[0],
        "answer_f1": best[1],
        "answer_precision": best[2],
        "answer_recall": best[3],
        "null_refusal": UNJUDGED_SCORE,
    }


def calculate_evidence_match(retrieved_sources: list[Any], expected_doc: str, expected_page: int | None = None) -> dict:
    """
    증거 매칭 - 문서/페이지 레벨. `calculate_multihop_doc_recall`이
    (page 없이) 재사용한다.
    Supports both string filenames and structured [title, page, ...] lists.

    Args:
        retrieved_sources: List of strings or lists [title, page, sent_id]
        expected_doc: 예상 문서명 (e.g., "3M_2018_10K")
        expected_page: 예상 페이지 번호 (optional)

    Returns:
        dict with 'doc_match', 'page_match'
    """
    if not retrieved_sources or not expected_doc:
        return {"doc_match": 0.0, "page_match": 0.0}

    doc_match = 0.0
    page_match = 0.0

    def normalize_doc_id(value: str) -> str:
        if not value:
            return ""
        lowered = str(value).lower().strip()
        lowered = re.sub(r"\.(pdf|txt|md|json)$", "", lowered)
        lowered = lowered.replace("10-k", "10k").replace("10-q", "10q")
        lowered = re.sub(r"[^a-z0-9]+", "", lowered)
        return lowered

    def tokenize_doc_id(value: str) -> set[str]:
        if not value:
            return set()
        lowered = str(value).lower()
        lowered = lowered.replace("10-k", "10k").replace("10-q", "10q")
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return {tok for tok in lowered.split() if tok}

    expected_doc_norm = normalize_doc_id(expected_doc)
    expected_doc_tokens = tokenize_doc_id(expected_doc)

    for source in retrieved_sources:
        src_title = ""
        src_page = None

        # Dict Source: {"doc": ..., "page": ..., "text": ...}
        if isinstance(source, dict):
            src_title = str(source.get("doc", "")).lower()
            src_page = source.get("page")

        # Structured Source: [title, page, sent_id]
        elif isinstance(source, (list, tuple)) and len(source) >= 2:
            src_title = str(source[0]).lower()
            src_page = source[1]

        # String Source: "Title" or "Title_page_5"
        elif isinstance(source, str):
            src_title = source

        src_doc_norm = normalize_doc_id(src_title)
        src_doc_tokens = tokenize_doc_id(src_title)

        is_doc_match = False
        if expected_doc_norm and src_doc_norm:
            if expected_doc_norm in src_doc_norm or src_doc_norm in expected_doc_norm:
                is_doc_match = True
            else:
                sim = SequenceMatcher(None, expected_doc_norm, src_doc_norm).ratio()
                if sim >= 0.92:
                    is_doc_match = True

        if not is_doc_match and expected_doc_tokens and src_doc_tokens:
            overlap = len(expected_doc_tokens.intersection(src_doc_tokens))
            min_required = max(1, int(len(expected_doc_tokens) * 0.6))
            if overlap >= min_required:
                is_doc_match = True

        if is_doc_match:
            doc_match = 1.0
            if expected_page is not None:
                if isinstance(src_page, (int, float)) and int(src_page) == expected_page:
                    page_match = 1.0
                    break
                if isinstance(source, str):
                    source_lower = source.lower()
                    page_pattern = (
                        f"page_{expected_page:03d}" if isinstance(expected_page, int) else f"page_{expected_page}"
                    )
                    if page_pattern in source_lower or f"_page_{expected_page}" in source_lower:
                        page_match = 1.0
                        break

    return {"doc_match": doc_match, "page_match": page_match}


def _parse_unit_score(raw: Any) -> float | None:
    try:
        if raw is None:
            return None
        value = float(raw)
        return max(0.0, min(1.0, value))
    except (TypeError, ValueError, OverflowError):
        return None


def _is_insufficient_text(text: Any) -> bool:
    # Matches the 3-way taxonomy: any recognized abstain phrase (Hypo's
    # "insufficient evidence", HopRAG's "I do not know", or natural-language
    # refusals) → not a hallucination.
    from utils.abstain import is_abstain

    return is_abstain(text)


# Sentinel for a row the LLM judge never scored (judge failed, or — in batch
# mode — not yet resolved). Kept out of [0,1] so aggregation can exclude it
# instead of treating an unjudged row as a wrong (0.0) answer.
UNJUDGED_SCORE = -1.0


async def _run_combined_judge(
    judge_prompt: str,
    response: str,
    vllm_client,
) -> dict:
    """Run the single-call judge with independent answer/context axes.

    The judge returns answer correctness (`score`), context support
    (`groundedness`), and context-based hallucination. A wrong answer can be
    grounded, and a correct answer can be unsupported; the resolver preserves
    that distinction.
    """
    judge_model, judge_payload = await _call_judge_llm(judge_prompt, vllm_client)
    return _resolve_judge_fields(judge_payload, response, judge_model)


async def _call_judge_llm(judge_prompt: str, vllm_client) -> tuple[str, dict | None]:
    """The synchronous LLM judge call using the configured evaluation model.

    Returns (judge_model, judge_payload). Factored out so the batch path can
    reuse `_resolve_judge_fields` on payloads it fetched elsewhere. A failed or
    malformed judge stays unjudged; it is never silently scored by a different
    model, which would mix evaluator policies within one benchmark run.
    """
    judge_model = RAGConfig.EVAL_MODEL
    judge_payload: dict | None = None
    if vllm_client:
        try:
            judge_payload = await vllm_client.generate_json(
                [{"role": "user", "content": judge_prompt}],
                model=RAGConfig.EVAL_MODEL,
            )
        except Exception as e:  # noqa: BLE001 - evaluator providers use heterogeneous exceptions
            logger.error(f"LLM Judge failed: {e}")
            judge_payload = None
    return judge_model, judge_payload


def _resolve_judge_fields(
    judge_payload: dict | None,
    response: str,
    judge_model: str,
) -> dict:
    """Resolve answer correctness, groundedness, and hallucination fields.

    Pure / no I/O so judge responses are resolved deterministically.
    When the payload carries no usable score (judge failed, or batch not yet
    resolved) the row is marked UNJUDGED_SCORE (-1) — never silently 0 — so
    aggregation excludes it rather than counting it as a wrong answer.
    """
    parsed_score = _parse_unit_score((judge_payload or {}).get("score"))
    parsed_groundedness = _parse_unit_score((judge_payload or {}).get("groundedness"))
    parsed_hallu = _parse_unit_score((judge_payload or {}).get("hallucination"))

    if parsed_score is not None:
        judge_score = parsed_score
        judge_reason = str((judge_payload or {}).get("reason", "")) or "combined_judge"
    else:
        judge_score = UNJUDGED_SCORE
        judge_reason = "unjudged_no_score"

    # Abstentions have no substantive claim to ground, so groundedness is not
    # applicable. Inspect only the extracted final answer: rationale may
    # mention an abstention phrase before concluding with a real answer.
    final_answer = extract_final_answer(response)
    is_non_answer = _is_insufficient_text(final_answer) or not final_answer.strip()
    if is_non_answer:
        groundedness = UNJUDGED_SCORE
        groundedness_source = "rule_non_answer"
    elif parsed_groundedness is not None:
        groundedness = 1.0 if parsed_groundedness >= 0.5 else 0.0
        groundedness_source = "combined_judge"
    else:
        groundedness = UNJUDGED_SCORE
        groundedness_source = "unjudged"

    # Honest-abstain is deterministic. For substantive answers hallucination
    # requires the explicit context-directed field; it is never derived from
    # answer correctness alone.
    if _is_insufficient_text(final_answer):
        hallucination = 0.0
        hallucination_reason = "non_answer_insufficient"
        hallucination_source = "rule_non_answer"
    elif not final_answer.strip():
        hallucination = 0.0
        hallucination_reason = "non_answer_empty"
        hallucination_source = "rule_non_answer"
    elif parsed_hallu is not None:
        hallucination = 1.0 if parsed_hallu >= 0.5 else 0.0
        hallucination_reason = str((judge_payload or {}).get("reason", "")) or "combined_judge"
        hallucination_source = "combined_judge"
    else:
        hallucination = UNJUDGED_SCORE
        hallucination_reason = "unjudged_no_hallucination_field"
        hallucination_source = "unjudged"

    return {
        "llm_judge_score": judge_score,
        "llm_judge_reason": judge_reason,
        "groundedness": groundedness,
        "groundedness_source": groundedness_source,
        "hallucination": hallucination,
        "hallucination_reason": hallucination_reason,
        "hallucination_source": hallucination_source,
        "hallucination_model": judge_model,
    }


async def _judge_or_defer(
    judge_prompt: str,
    response: str,
    vllm_client,
    batch_collector=None,
    custom_id: str | None = None,
) -> dict:
    """Register the default Batch request or run the explicit sync debug path."""
    if batch_collector is not None:
        if custom_id is None:
            raise ValueError("Batch judge requires a stable custom_id")
        batch_collector.register(custom_id, judge_prompt)
        judge = _resolve_judge_fields(None, response, RAGConfig.EVAL_MODEL)
        judge["_deferred_judge"] = True
        judge["judge_custom_id"] = custom_id
        return judge
    return await _run_combined_judge(judge_prompt, response, vllm_client)


def _format_judge_context(retrieved_sources: list[Any] | None) -> str:
    """Normalize retrieved source records into the judge's evidence view."""
    nodes: list[dict[str, Any]] = []
    for source in retrieved_sources or []:
        if isinstance(source, dict):
            node = dict(source)
            node["title"] = node.get("title") or node.get("doc") or node.get("source") or "Unknown"
            node["text"] = str(node.get("text") or "")
            nodes.append(node)
        elif isinstance(source, (list, tuple)) and source:
            nodes.append(
                {
                    "title": str(source[0] or "Unknown"),
                    "page": source[1] if len(source) > 1 else 0,
                    "sent_id": source[2] if len(source) > 2 else 0,
                    "text": str(source[3] if len(source) > 3 else ""),
                }
            )
    return format_context_from_nodes(nodes) if nodes else "(empty retrieved context)"


# --- Multi-hop dataset metrics (MultiHop-RAG, MuSiQue) ---


def _fact_matches_chunk(fact_norm: str, chunk_norm: str) -> bool:
    """True if a gold evidence fact is contained in / overlaps a retrieved
    chunk. `fact_norm`/`chunk_norm` are already `normalize_answer`-ed.

    A MultiHop-RAG `fact` is a sentence pulled from a source article, so the
    retrieved chunk that supports it should contain that sentence (or share
    most of its tokens after table/whitespace reflow). Substring first, then
    a 0.6 token-coverage fallback for reflowed chunks.
    """
    if not fact_norm or not chunk_norm:
        return False
    if fact_norm in chunk_norm or chunk_norm in fact_norm:
        return True
    fact_tokens = set(fact_norm.split())
    if not fact_tokens:
        return False
    overlap = len(fact_tokens & set(chunk_norm.split()))
    return (overlap / len(fact_tokens)) >= 0.6


def _source_chunk_text(source: Any) -> str:
    """Pull the chunk body text out of a retrieved source of any shape
    (dict {"text"}, list [title, page, text], or raw string)."""
    if isinstance(source, dict):
        return str(source.get("text", "") or "")
    if isinstance(source, (list, tuple)) and len(source) >= 3:
        return str(source[2] or "")
    if isinstance(source, str):
        return source
    return ""


def calculate_retrieval_ranking_metrics(
    retrieved_sources: list[Any],
    gold_facts: list[str],
    ks: tuple[int, ...] = (4, 10),
) -> dict:
    """Fact-level ranking metrics for sentence-aligned evidence.

    The current implementation is used for MultiHop-RAG;
    paragraph-level MuSiQue evidence is intentionally excluded by the caller.
    The returned ``hits@k`` values are fact-recall fractions, not binary
    query-level hit rates. Relevance is `_fact_matches_chunk` against the
    ranked retrieved chunk texts.

    - hits@k : recall of distinct gold facts within the top-k chunks.
    - mrr@10 : reciprocal rank of the first chunk covering ANY gold fact.
    - map@10 : average precision over the ranked chunks (a chunk is
               "relevant" when it covers a not-yet-covered gold fact),
               normalized by the gold-fact count.
    """
    gold_norm = [g for g in (normalize_answer(f) for f in (gold_facts or []) if f) if g]
    empty_value = UNJUDGED_SCORE if not gold_norm else 0.0
    result: dict[str, float] = {f"hits@{k}": empty_value for k in ks}
    result["mrr@10"] = empty_value
    result["map@10"] = empty_value
    if not gold_norm or not retrieved_sources:
        return result

    chunk_norms = [normalize_answer(_source_chunk_text(s)) for s in retrieved_sources]
    total_gold = len(gold_norm)

    # MRR / MAP over the ranked list (count a gold fact once, at first cover).
    covered: set[int] = set()
    first_hit_rank: int | None = None
    relevant_count = 0
    ap_sum = 0.0
    for rank, cn in enumerate(chunk_norms, start=1):
        newly = {gi for gi, g in enumerate(gold_norm) if gi not in covered and _fact_matches_chunk(g, cn)}
        if not newly:
            continue
        covered |= newly
        relevant_count += 1
        if first_hit_rank is None:
            first_hit_rank = rank
        if rank <= 10:
            ap_sum += relevant_count / rank

    result["mrr@10"] = (1.0 / first_hit_rank) if (first_hit_rank and first_hit_rank <= 10) else 0.0
    result["map@10"] = ap_sum / total_gold

    # Hits@k: distinct gold facts recalled within the top-k chunks.
    for k in ks:
        top_k = chunk_norms[:k]
        hit = sum(1 for g in gold_norm if any(_fact_matches_chunk(g, cn) for cn in top_k))
        result[f"hits@{k}"] = hit / total_gold

    return result


def calculate_multihop_doc_recall(retrieved_sources: list[Any], gold_docs: list[str]) -> float:
    """Coarse doc-level recall: fraction of gold articles (by title) that
    appear among the retrieved sources. Complements the fact-level ranking
    metrics with a title-match view robust to chunk reflow."""
    gold = [d for d in (gold_docs or []) if d and str(d).strip()]
    if not gold:
        return UNJUDGED_SCORE
    hit = 0
    for doc in gold:
        m = calculate_evidence_match(retrieved_sources, doc, expected_page=None)
        if m["doc_match"] >= 1.0:
            hit += 1
    return hit / len(gold)


def _source_doc_title(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("doc", source.get("title", source.get("source", ""))) or "")
    if isinstance(source, (list, tuple)) and source:
        return str(source[0] or "")
    if isinstance(source, str):
        return source
    return ""


def _normalize_doc_key(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\.(pdf|txt|md|json)$", "", text)
    text = text.replace("10-k", "10k").replace("10-q", "10q")
    return re.sub(r"[^a-z0-9]+", "", text)


def calculate_evidence_doc_metrics(
    retrieved_sources: list[Any],
    gold_docs: list[str],
) -> dict[str, float]:
    """Compute title-level evidence precision/recall/F1 over unique sources.

    This is the common evidence unit for the active datasets. It is deliberately
    separate from sentence/fact ranking metrics because MuSiQue stores full
    supporting paragraphs while the retriever returns chunks.
    """
    gold_keys = {_normalize_doc_key(doc) for doc in (gold_docs or []) if _normalize_doc_key(doc)}
    retrieved_keys = {
        _normalize_doc_key(_source_doc_title(source))
        for source in (retrieved_sources or [])
        if _normalize_doc_key(_source_doc_title(source))
    }
    if not gold_keys:
        return {
            "evidence_doc_precision": UNJUDGED_SCORE,
            "evidence_doc_recall": UNJUDGED_SCORE,
            "evidence_doc_f1": UNJUDGED_SCORE,
        }
    if not retrieved_keys:
        return {
            "evidence_doc_precision": 0.0,
            "evidence_doc_recall": 0.0,
            "evidence_doc_f1": 0.0,
        }

    true_positive = len(gold_keys & retrieved_keys)
    precision = true_positive / len(retrieved_keys)
    recall = true_positive / len(gold_keys)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evidence_doc_precision": precision,
        "evidence_doc_recall": recall,
        "evidence_doc_f1": f1,
    }


async def evaluate_multihoprag_response(
    query: str,
    response: str,
    ground_truth: str,
    retrieved_sources: list[Any],
    evidence_facts: list[str] | None = None,
    evidence_docs: list[str] | None = None,
    question_type: str = "",
    dataset: str = "",
    answer_aliases: list[str] | None = None,
    vllm_client=None,
    batch_collector=None,
    custom_id: str | None = None,
) -> dict:
    """Evaluate answer quality and dataset-appropriate evidence quality.

    Deterministic normalized EM/F1 is the primary answer signal. The LLM judge
    keeps semantic correctness and context groundedness as separate
    diagnostic axes. Sentence/fact ranking metrics are skipped for MuSiQue
    because its gold evidence is paragraph-level, while title-level evidence
    precision/recall/F1 is reported for every dataset.
    """
    judge_prompt = MULTIHOPRAG_JUDGE_PROMPT.format(
        question_type=question_type or "unknown",
        query=query,
        ground_truth=ground_truth,
        response=response,
        retrieved_context=_format_judge_context(retrieved_sources),
    )
    judge = await _judge_or_defer(judge_prompt, response, vllm_client, batch_collector, custom_id)

    answer_metrics = calculate_answer_metrics(
        response,
        ground_truth,
        answer_aliases=answer_aliases,
        question_type=question_type,
    )
    dataset_marker = str(dataset or "").strip().lower()
    if dataset_marker == "musique":
        ranking = {
            "hits@4": UNJUDGED_SCORE,
            "hits@10": UNJUDGED_SCORE,
            "mrr@10": UNJUDGED_SCORE,
            "map@10": UNJUDGED_SCORE,
        }
    else:
        ranking = calculate_retrieval_ranking_metrics(retrieved_sources, evidence_facts or [])
    evidence_docs_metrics = calculate_evidence_doc_metrics(retrieved_sources, evidence_docs or [])

    return {
        **judge,
        **answer_metrics,
        **ranking,
        # Coarse view of doc_recall (1.0 if any gold article surfaced).
        **evidence_docs_metrics,
        "doc_match": (
            1.0
            if evidence_docs_metrics["evidence_doc_recall"] > 0.0
            else 0.0
            if evidence_docs_metrics["evidence_doc_recall"] >= 0.0
            else UNJUDGED_SCORE
        ),
    }
