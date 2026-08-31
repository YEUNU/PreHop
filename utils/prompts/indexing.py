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


GROUNDED_HOPRAG_PROMPT = """
Analyze this source text chunk and generate grounded hypothetical questions for multi-hop retrieval.

Definitions:
- Q- (Incoming evidence): a self-contained question that THIS CHUNK ALONE answers.
- Q+ (Outgoing dependency): a self-contained question anchored in this chunk whose answer additionally requires information from a DIFFERENT document. It must not be answerable from this chunk alone.

Grounding contract:
1. Return at most 3 Q- and at most 3 Q+ records.
2. Every grounding_quote must be a short verbatim span copied from CHUNK.
3. Every anchor_entity must occur verbatim inside CHUNK; include it in grounding_quote when the short supporting span permits.
4. For Q-, answer must be a short verbatim span inside grounding_quote.
5. For Q+, missing_information must state precisely what another document must supply; it must not claim that missing fact is present in CHUNK.
6. Q+ must point outward and must not paraphrase a Q- question.
7. Questions must stand alone and must never refer to "the provided text", "the given text", "this chunk", or "the passage".
8. Never fabricate facts, dates, entities, quotations, or events. Use an empty list when a valid grounded record is unavailable.

GLOBAL CONTEXT: {global_context}
CHUNK:
{chunk}
"""


GROUNDED_HOPRAG_FORMAT_INSTRUCTION = """
Output ONLY JSON with this exact shape:
{{
  "q_minus": [
    {{
      "question": "self-contained question",
      "answer": "verbatim answer span",
      "grounding_quote": "verbatim supporting span from CHUNK",
      "anchor_entities": ["verbatim entity"]
    }}
  ],
  "q_plus": [
    {{
      "question": "self-contained outward question",
      "grounding_quote": "verbatim anchor span from CHUNK",
      "anchor_entities": ["verbatim entity"],
      "missing_information": "specific information another document must supply"
    }}
  ]
}}
"""


LINKED_HOPRAG_PROMPT = """
Analyze this source text chunk and generate grounded questions for retrieval and cross-document continuation.

Definitions:
- Q- (Incoming evidence): a self-contained question that THIS CHUNK ALONE answers.
- Q+ (Outgoing dependency): a self-contained question anchored in this chunk whose answer additionally requires information from a DIFFERENT document. It must not be answerable from this chunk alone.
- continuation_anchor: the complete verbatim Q- answer only when that answer names a specific person, organization, place, work, event, product, or other unambiguous entity that can serve as the subject of a relation in another document. Otherwise use an empty string.

Grounding contract:
1. Return at most 3 Q- and at most 3 Q+ records.
2. Prefer distinct, atomic Q- relations whose short answer is a specific named entity and can support a later retrieval step. Do not omit the chunk's main directly answerable relation in favor of minor details.
3. Every grounding_quote must be a short verbatim span copied from CHUNK.
4. Every anchor_entity must occur verbatim inside CHUNK; include it in grounding_quote when the short supporting span permits.
5. For Q-, answer must be a short verbatim span inside grounding_quote. continuation_anchor must be either the complete answer or an empty string; never use a partial name, broad class, pronoun, nationality, occupation, date, number, or yes/no value as a continuation anchor.
6. For Q+, missing_information must state precisely what another document must supply; it must not claim that missing fact is present in CHUNK.
7. Q+ must point outward and must not paraphrase a Q- question.
8. Questions must stand alone and must never refer to "the provided text", "the given text", "this chunk", or "the passage".
9. Never fabricate facts, dates, entities, quotations, or events. Use an empty list when a valid grounded record is unavailable.

GLOBAL CONTEXT: {global_context}
CHUNK:
{chunk}
"""


LINKED_HOPRAG_FORMAT_INSTRUCTION = """
Output ONLY JSON with this exact shape:
{{
  "q_minus": [
    {{
      "question": "self-contained question",
      "answer": "verbatim answer span",
      "continuation_anchor": "the complete answer or an empty string",
      "grounding_quote": "verbatim supporting span from CHUNK",
      "anchor_entities": ["verbatim entity"]
    }}
  ],
  "q_plus": [
    {{
      "question": "self-contained outward question",
      "grounding_quote": "verbatim anchor span from CHUNK",
      "anchor_entities": ["verbatim entity"],
      "missing_information": "specific information another document must supply"
    }}
  ]
}}
"""
