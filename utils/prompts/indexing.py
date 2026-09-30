# Model-side prompt for dataset-neutral Q-/Q+ generation.

HOPRAG_PROMPT = """
Analyze this source text chunk and generate hypothetical questions to enable multi-hop reasoning.

Definitions:
- Q- (Incoming, self-contained): questions that THIS CHUNK ALONE answers verbatim. Used to retrieve this chunk when a user query asks for a fact already covered here.
- Q+ (Outgoing dependency / Bridge): questions that POINT OUTWARD from this chunk — they reference a person, organization, place, event, or date grounded here, but the answer ALSO REQUIRES information from a DIFFERENT chunk or document. The Q+ question is the missing counterpart another document would supply. Q+ is what builds the multi-hop graph; it is NOT a paraphrase of Q-.

Rules:
1. Return at most 3 Q- questions that this chunk directly answers.
2. Return at most 3 Q+ questions whose answer requires another chunk or document.
3. Keep every question specific and grounded in entities or events present in the chunk.
4. Do not emit a Q- paraphrase as Q+; leave the list empty when no outward dependency is supported.
5. Every question must stand alone. Never refer to "the provided text", "the given text", "this chunk", or "the passage".
6. Never fabricate facts, dates, entities, quotations, or events.
7. Return empty lists for navigation, boilerplate, or text without useful questions.

GLOBAL CONTEXT: {global_context}
CHUNK:
{chunk}
"""

HOPRAG_FORMAT_INSTRUCTION = """
Output ONLY JSON:
{{"q_minus": ["q1", "q2", "q3"], "q_plus": ["q1", "q2", "q3"]}}
"""
