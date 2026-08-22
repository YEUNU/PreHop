def answer_role() -> str:
    """Return the one dataset-neutral role shared by all current benchmarks."""
    return "a multi-hop research assistant"


def build_answer_prompt(context: str, query: str) -> str:
    """Shared synthesis prompt for in-repo and adapted retrieval methods.

    Keeping this in one function ensures retrieval, rather than prompt wording,
    is the variable being compared across Prehop, Naive, and HopRAG.
    """
    return (
        f"You are {answer_role()}. Answer the question using only the provided context.\n"
        "If the context is insufficient, say you do not know.\n"
        "\n"
        f"Context:\n{context}\n"
        "\n"
        f"Question: {query}\n"
        "\n"
        "Answer:"
    )
