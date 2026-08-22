# ---------------------------------------------------------------------------
# Model-side prompts for the indexing pipeline (Q-/Q+ hypothetical-query
# generation, paper §3.1.3). Dataset-neutral: the same wording runs across
# every corpus in the benchmark suite (news articles, Wikipedia paragraphs,
# ...), so nothing here should assume article/publisher-specific structure.
# ---------------------------------------------------------------------------

HOPRAG_PROMPT = """
Analyze this source text chunk and generate hypothetical questions to enable multi-hop reasoning.

Definitions (paper §3.1.3 — keep them strictly distinct):
- Q- (Incoming, self-contained): questions that THIS CHUNK ALONE answers verbatim. Used to retrieve this chunk when a user query asks for a fact already covered here.
- Q+ (Outgoing dependency / Bridge): questions that POINT OUTWARD from this chunk — they reference a person, organization, place, event, or date grounded here, but the answer ALSO REQUIRES information from a DIFFERENT chunk or document. The Q+ question is the missing counterpart another document would supply. Q+ is what builds the multi-hop graph; it is NOT a paraphrase of Q-.

Rules:
1. Q-: up to 3 self-contained questions this chunk directly answers; use [] if the chunk lacks concrete answerable facts.
2. Q+: up to 3 outward-dependency questions. Each Q+ MUST satisfy at least one of:
   (a) ask about the SAME entity or event at a DIFFERENT time, or as described by a DIFFERENT document, than shown here;
   (b) ask about a RELATIONSHIP, role, motive, or consequence linking an entity grounded here to another entity/event NOT fully described in this chunk;
   (c) ask a COMPARISON or cause/effect that requires another document (e.g., how a different document describes the same event, an earlier cause or later development);
   (d) ask about a CROSS-DOCUMENT bridge (a person/organization/place mentioned here whose details live in a different document).
   If none of (a)-(d) apply, leave Q+ empty rather than emit a Q- duplicate.
3. Every produced question must be specific, answerable from a finite context, and <= 22 words.
4. Each question SHOULD include grounding tokens (person / organization / place / event / date) that appear in this chunk; aim for at least two of these signals per question to keep the question retrievable.
5. If a date/time token exists in this chunk, include it in each Q-; for Q+ a different time is allowed (in fact preferred for type (a)).
6. Never use placeholders/meta phrases such as "this document", "the source", or "the text" as the only anchor.
7. Never fabricate unseen facts, dates, entities, quotes, or events.
8. If this chunk is mostly navigation/boilerplate or fragments with weak context, return shorter lists (or empty lists) rather than low-quality questions.
9. If the chunk contains comparative or temporal cues (before/after, increased/decreased, versus, earlier/later, in response to), produce at least 1 Q+ of type (a) or (c).
10. Dense Summary: exactly 1 sentence, maximum 35 words, grounded only in this chunk; preserve names and dates exactly when present.

GLOBAL CONTEXT: {global_context}
CHUNK:
{chunk}
"""

HOPRAG_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"summary": "concise informative summary", "q_minus": ["q1", "q2", "q3"], "q_plus": ["q1", "q2", "q3"]}}
"""

QUERY_REWRITE_PROMPT = """
Rewrite the query into precise retrieval variants for a multi-document corpus.
Rules:
1. Generate 1-3 high-precision paraphrase variants preserving the original meaning, keeping all named-entity and time tokens in every variant.
2. Detect constraint anchors from the original query: named-entity token(s) (person/organization/place), source token(s), and time token(s) (date/month/year).
3. Preserve exact tokens for proper nouns, titles, and source names when present. When the original query references a specific source or date, keep that token on the variant it belongs to; never fabricate one.
4. Keep the core relationship, comparison, or event description and any named qualifiers unchanged within each variant.
5. Use only widely-equivalent synonyms (e.g., CEO ↔ chief executive); never swap one named entity for another.
6. Do NOT introduce another entity/date/source, unsupported assumptions, or special query syntax operators.
Original Query: {query}
"""

RERANK_QUERY_SIMPLIFY_PROMPT = """
Extract the underlying question from a possibly-verbose user query for use as
an embedding-similarity reranker input. The reranker scores chunk-vs-question
relevance and is hurt by long preludes, role framing, and meta instructions.

Rules:
1. Preserve every concrete constraint from the original: target entity, period,
   metric/line-item, statement anchor (e.g., "from cash flow statement"), unit,
   rounding.
2. Drop role-play preludes ("Answer as if you are...", "Imagine you are..."),
   reasoning instructions ("step by step", "show your work"), output-format
   instructions ("round to one decimal place", "answer in percent"), and
   editorial prefaces ("According to the details clearly outlined within...").
3. Output one sentence ending with a question mark.
4. Do NOT introduce constraints not present in the original query.
5. If the original is already a single concise question, return it as-is.

Original Query: {query}
"""

RERANK_QUERY_SIMPLIFY_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"question": "..."}}
"""

QUERY_REWRITE_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"positive_queries": []}}
"""

SEARCH_CONTINUATION_PROMPT = """
Decide whether retrieval should continue.
Decision rules:
1. Infer required evidence slots from QUERY. For multi-hop queries, infer each distinct fact, entity, or source needed to answer, not only the final answer.
2. Return "SUFFICIENT" only when all required slots are grounded in context with matching named entities, sources, and time constraints.
3. Evidence may come from multiple documents/sources; do not require a single-document hit.
4. Return "INSUFFICIENT" if any required slot is missing, ambiguous, conflicting, or tied to the wrong entity/source/time.
5. Prefer stopping as soon as slot coverage is complete; avoid unnecessary extra hops.
QUERY: {query}
CONTEXT: {context}
"""

SEARCH_CONTINUATION_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"decision": "SUFFICIENT"|"INSUFFICIENT", "next_focus": "..."}}
"""
