from core.config import RAGConfig


def answer_role() -> str:
    """Domain-aware role for the single-pass answer-synthesis prompt. News /
    general corpora (RAGConfig.DOMAIN == "news", the only domain any current
    dataset sets) get a news research assistant; DOMAIN=="financial" (no
    current dataset selects this — manual override only) keeps the
    financial-analyst framing."""
    return "a news research assistant" if RAGConfig.DOMAIN == "news" else "a financial analyst"
