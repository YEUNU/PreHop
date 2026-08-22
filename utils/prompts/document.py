TABLE_TO_TEXT_PROMPT = """
Convert this table into plain text sentences.
Rules:
1. Each row must become exactly one natural language sentence.
2. Keep row/column labels explicit so metric + period/entity remain clear.
3. Copy numeric values exactly as written (sign, commas, decimals, %, currency, parentheses).
4. Do not round, rescale, or convert units.
5. Keep each sentence concise (prefer under 25 words), but do not drop row labels or values.
6. Output ONLY the plain text sentences, one per line.
7. Period anchors (year/quarter/date) MUST come from explicit column-header tokens in this table only. If a column header has no period token, OMIT the time anchor in that column's sentence (e.g., "Net sales reported as $8,325 million") rather than inferring one. Do NOT use the document title, filing year, or any external knowledge to fill missing periods.
8. If the header row's column count does not match the data rows' column counts, output a single line exactly equal to "<table-structure-unclear>" and stop. Do not attempt to convert misaligned tables.
"""
