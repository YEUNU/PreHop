"""Present existing passage connections to an evidence selector."""

GRAPH_EVIDENCE_INSTRUCTIONS = (
    "Use the stored graph connections below to check whether paragraphs supply "
    "complementary facts needed for the question. HOP links propose cross-document "
    "dependencies; NEXT links mark adjacent passages. The links were constructed "
    "before the question was asked. They are search hints, not proof that either "
    "paragraph is relevant or that a factual relation is true. Check the passage "
    "text. When a useful paragraph depends on a fact in a linked paragraph, consider "
    "both together and keep the necessary connecting evidence in the selected set. "
    "Rank complementary evidence ahead of redundant restatements, while keeping "
    "the most useful evidence early. A linked paragraph that adds no necessary "
    "information need not be selected. Do not reserve slots for graph results or "
    "select a paragraph solely because it has links. Use the same requested number "
    "of candidate IDs and the same JSON output format."
)


def add_graph_evidence(prompt: str, candidates: dict[str, dict]) -> str:
    """Add only recorded HOP/NEXT edges whose endpoints are supplied candidates."""
    ids = {node["id"]: candidate_id for candidate_id, node in candidates.items()}
    links = set()
    for candidate_id, node in candidates.items():
        for path in node.get("retrieval_paths") or []:
            kind, source = path.get("kind"), path.get("source_chunk_id")
            if kind in {"hop", "next"} and source in ids and source != node["id"]:
                links.add((kind.upper(), ids[source], candidate_id))
    text = "\n".join(f"{kind}: {source} -> {target}" for kind, source, target in sorted(links))
    return (GRAPH_EVIDENCE_INSTRUCTIONS + "\n\n" + prompt +
            "\n\nStored graph connections:\n" + (text or "None within this candidate set."))
