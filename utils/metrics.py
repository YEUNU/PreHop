import json
import logging
import re
import string
from collections import Counter
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
            "official_answer_em": UNJUDGED_SCORE,
            "official_answer_f1": UNJUDGED_SCORE,
            # Unlike the custom answer EM/F1 policy, MultiHop-RAG's official
            # QA evaluator scores null questions against its supplied
            # "Insufficient information." gold answer.
            "official_qa_accuracy": _official_multihoprag_qa_accuracy(final_answer, ground_truth),
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
            "official_answer_em": UNJUDGED_SCORE,
            "official_answer_f1": UNJUDGED_SCORE,
            "official_qa_accuracy": UNJUDGED_SCORE,
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
        # MuSiQue's AnswerMetric uses this normalized EM/F1 and takes the
        # maximum over aliases.  Keep the official names beside the shared
        # primary fields so an artifact can state exactly what it reports.
        "official_answer_em": best[0],
        "official_answer_f1": best[1],
        # MultiHop-RAG's published QA script treats any shared whitespace
        # token as correct.  This deliberately permissive score is retained
        # only for protocol comparison; normalized EM/F1 remains primary.
        "official_qa_accuracy": _official_multihoprag_qa_accuracy(final_answer, ground_truth),
    }


def _official_multihoprag_qa_accuracy(prediction: str, reference: str) -> float:
    """Reproduce MultiHop-RAG QA's any-token-intersection decision rule.

    The benchmark harness scores the extracted final answer rather than a
    model rationale.  Apart from that shared output-boundary policy, this
    intentionally uses only lowercase whitespace tokenization: it is not the
    repository's normalized EM/F1 metric.
    """
    pred_tokens = {token for token in str(prediction or "").lower().split() if token}
    gold_tokens = {token for token in str(reference or "").lower().split() if token}
    return 1.0 if pred_tokens & gold_tokens else 0.0


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
    """Run the single-call supplemental judge with answer/context axes.

    The judge returns answer correctness (`score`) and context support
    (`groundedness`). Hallucination is derived from groundedness in the local
    resolver, preventing two redundant LLM labels from disagreeing.
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

    # Honest abstention is not a substantive hallucination claim.  For a
    # substantive answer, hallucination is exactly the complement of the
    # independently judged context-groundedness field.  Keep the legacy field
    # name for artifact compatibility, but never trust a model-supplied value.
    if _is_insufficient_text(final_answer):
        hallucination = 0.0
        hallucination_reason = "non_answer_insufficient"
        hallucination_source = "rule_non_answer"
    elif not final_answer.strip():
        hallucination = 0.0
        hallucination_reason = "non_answer_empty"
        hallucination_source = "rule_non_answer"
    elif groundedness >= 0:
        hallucination = 1.0 - groundedness
        hallucination_reason = "derived_from_groundedness"
        hallucination_source = "derived_from_groundedness"
    else:
        hallucination = UNJUDGED_SCORE
        hallucination_reason = "unjudged_no_groundedness_field"
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


def _official_multihoprag_fact_match(fact: str, chunk: str) -> bool:
    """Official MultiHop-RAG retrieval matching: whitespace/newline substring.

    Do not normalize punctuation, case, or token overlap here.  Those looser
    rules belong to the repository's diagnostic fact-coverage metric.
    """
    fact_compact = str(fact or "").replace(" ", "").replace("\n", "")
    chunk_compact = str(chunk or "").replace(" ", "").replace("\n", "")
    return bool(fact_compact and fact_compact in chunk_compact)


def calculate_retrieval_ranking_metrics(
    retrieved_sources: list[Any],
    gold_facts: list[str],
    ks: tuple[int, ...] = (4, 10),
) -> dict:
    """Emit official MultiHop-RAG retrieval metrics and separate fact recall.

    ``official_hits@k`` is query-level any-hit, while
    ``evidence_fact_recall@k`` is this project's more informative fraction of
    gold facts recovered.  They must never share a field name.
    """
    gold_raw = [str(f) for f in (gold_facts or []) if str(f).strip()]
    gold_norm = [normalize_answer(f) for f in gold_raw]
    empty_value = UNJUDGED_SCORE if not gold_raw else 0.0
    result: dict[str, float] = {
        **{f"official_hits@{k}": empty_value for k in ks},
        **{f"evidence_fact_recall@{k}": empty_value for k in ks},
        "official_mrr@10": empty_value,
        "official_map@10": empty_value,
    }
    if not gold_raw or not retrieved_sources:
        return result

    chunks = [_source_chunk_text(source) for source in retrieved_sources]
    chunk_norms = [normalize_answer(chunk) for chunk in chunks]
    total_gold = len(gold_raw)

    # Exact translation of the official ranking evaluator: a rank contributes
    # each *new* matched fact divided by its 1-based rank.  Multiple newly
    # found facts in one chunk all contribute at that rank.
    covered: set[int] = set()
    first_hit_rank: int | None = None
    ap_sum = 0.0
    for rank, chunk in enumerate(chunks[:10], start=1):
        newly = {idx for idx, fact in enumerate(gold_raw) if idx not in covered and _official_multihoprag_fact_match(fact, chunk)}
        if newly:
            if first_hit_rank is None:
                first_hit_rank = rank
            covered |= newly
            ap_sum += len(newly) / rank
    result["official_mrr@10"] = 1.0 / first_hit_rank if first_hit_rank else 0.0
    result["official_map@10"] = ap_sum / total_gold

    for k in ks:
        top_raw = chunks[:k]
        result[f"official_hits@{k}"] = float(
            any(_official_multihoprag_fact_match(fact, chunk) for fact in gold_raw for chunk in top_raw)
        )
        top_norm = chunk_norms[:k]
        recalled = sum(1 for fact in gold_norm if any(_fact_matches_chunk(fact, chunk) for chunk in top_norm))
        result[f"evidence_fact_recall@{k}"] = recalled / total_gold
    return result


def _source_doc_title(source: Any) -> str:
    if isinstance(source, dict):
        return str(source.get("doc", source.get("title", source.get("source", ""))) or "")
    if isinstance(source, (list, tuple)) and source:
        return str(source[0] or "")
    if isinstance(source, str):
        return source
    return ""


_MUSIQUE_PARAGRAPH_ID_RE = re.compile(
    r"(?:paragraph[-_ ]?id\s*:\s*|musique_)(musique:[a-f0-9]{12,64}|[a-f0-9]{12,64})",
    re.IGNORECASE,
)


def _source_paragraph_identity(source: Any) -> str:
    """Find the stable MuSiQue paragraph identity exposed by every adapter.

    Corpus filenames carry the hash identity, and the corpus body repeats it
    as a metadata header.  The two routes cover adapters that expose only a
    source filename or only retrieved text.
    """
    candidates: list[str] = []
    if isinstance(source, dict):
        candidates.extend(str(source.get(key) or "") for key in ("paragraph_id", "source", "doc", "text"))
    elif isinstance(source, (list, tuple)):
        candidates.extend(str(value or "") for value in source)
    else:
        candidates.append(str(source or ""))
    for candidate in candidates:
        direct = candidate.strip()
        if direct.startswith("musique:"):
            return direct.lower()
        match = _MUSIQUE_PARAGRAPH_ID_RE.search(candidate)
        if match:
            value = match.group(1).lower()
            return value if value.startswith("musique:") else f"musique:{value}"
    return ""


def calculate_musique_support_metrics(retrieved_sources: list[Any], gold_paragraph_ids: list[str]) -> dict[str, float]:
    """SupportMetric formula over global paragraph identities, not titles.

    MuSiQue's official evaluator receives query-local paragraph ``idx``
    predictions.  This global RAG corpus instead retrieves source documents,
    so these are deliberately named ``paragraph_support_*`` rather than
    ``official_support_*`` despite using the same set P/R/F1 formula.
    """
    gold = {str(value).strip().lower() for value in (gold_paragraph_ids or []) if str(value).strip()}
    retrieved = {_source_paragraph_identity(source) for source in (retrieved_sources or [])}
    retrieved.discard("")
    if not gold:
        value = UNJUDGED_SCORE
        return {
            "paragraph_support_precision": value,
            "paragraph_support_recall": value,
            "paragraph_support_f1": value,
        }
    true_positive = len(gold & retrieved)
    precision = true_positive / len(retrieved) if retrieved else 0.0
    recall = true_positive / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "paragraph_support_precision": precision,
        "paragraph_support_recall": recall,
        "paragraph_support_f1": f1,
    }


def _normalize_doc_key(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\.(pdf|txt|md|json)$", "", text)
    text = text.replace("10-k", "10k").replace("10-q", "10q")
    return re.sub(r"[^a-z0-9]+", "", text)


def calculate_evidence_doc_metrics(
    retrieved_sources: list[Any],
    gold_docs: list[str],
) -> dict[str, float]:
    """Compute title-level evidence P/R/F1 as a diagnostic-only view.

    It remains useful for MultiHop-RAG article inspection, but it is not
    MuSiQue's official support metric: a title can name multiple paragraphs.
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
    evidence_paragraph_ids: list[str] | None = None,
    question_type: str = "",
    dataset: str = "",
    answer_aliases: list[str] | None = None,
    vllm_client=None,
    batch_collector=None,
    custom_id: str | None = None,
    judge_enabled: bool = False,
) -> dict:
    """Evaluate answer quality and dataset-appropriate evidence quality.

    Deterministic normalized EM/F1 is the primary answer signal. The LLM judge
    keeps semantic correctness and context groundedness as separate
    diagnostic axes. Sentence/fact ranking metrics are skipped for MuSiQue
    because its gold evidence is paragraph-level, while title-level evidence
    precision/recall/F1 is reported for every dataset.
    """
    if judge_enabled:
        aliases = [str(alias).strip() for alias in (answer_aliases or []) if str(alias).strip()]
        judge_prompt = MULTIHOPRAG_JUDGE_PROMPT.format(
            question_type=question_type or "unknown",
            query=query,
            ground_truth=ground_truth,
            answer_aliases=json.dumps(aliases, ensure_ascii=False) if aliases else "(none)",
            response=response,
            retrieved_context=_format_judge_context(retrieved_sources),
        )
        judge = await _judge_or_defer(judge_prompt, response, vllm_client, batch_collector, custom_id)
    else:
        judge = {
            "llm_judge_score": UNJUDGED_SCORE,
            "llm_judge_reason": "judge_disabled",
            "groundedness": UNJUDGED_SCORE,
            "groundedness_source": "judge_disabled",
            "hallucination": UNJUDGED_SCORE,
            "hallucination_reason": "judge_disabled",
            "hallucination_source": "judge_disabled",
            "hallucination_model": "",
        }

    answer_metrics = calculate_answer_metrics(
        response,
        ground_truth,
        answer_aliases=answer_aliases,
        question_type=question_type,
    )
    dataset_marker = str(dataset or "").strip().lower()
    if dataset_marker == "musique":
        # MuSiQue officially supports alias-aware EM/F1.  It does not use
        # MultiHop-RAG's permissive token-intersection QA decision.
        answer_metrics["official_qa_accuracy"] = UNJUDGED_SCORE
        ranking = {
            "official_hits@4": UNJUDGED_SCORE,
            "official_hits@10": UNJUDGED_SCORE,
            "official_mrr@10": UNJUDGED_SCORE,
            "official_map@10": UNJUDGED_SCORE,
            "evidence_fact_recall@4": UNJUDGED_SCORE,
            "evidence_fact_recall@10": UNJUDGED_SCORE,
        }
        support_metrics = calculate_musique_support_metrics(retrieved_sources, evidence_paragraph_ids or [])
    else:
        # MultiHop-RAG officially supports its any-token QA accuracy, not
        # MuSiQue's alias-aware answer evaluator.
        answer_metrics["official_answer_em"] = UNJUDGED_SCORE
        answer_metrics["official_answer_f1"] = UNJUDGED_SCORE
        ranking = calculate_retrieval_ranking_metrics(retrieved_sources, evidence_facts or [])
        support_metrics = {
            "paragraph_support_precision": UNJUDGED_SCORE,
            "paragraph_support_recall": UNJUDGED_SCORE,
            "paragraph_support_f1": UNJUDGED_SCORE,
        }
    evidence_docs_metrics = calculate_evidence_doc_metrics(retrieved_sources, evidence_docs or [])

    return {
        **judge,
        **answer_metrics,
        **ranking,
        **support_metrics,
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
