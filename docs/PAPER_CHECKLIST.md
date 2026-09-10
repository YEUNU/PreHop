# Manuscript Readiness Checklist

This checklist applies to [the manuscript](prehop_paper.md). It separates
correct reporting in the current draft from completion of the research and
submission requirements. A checked item is limited to the evidence stated;
it does not certify acceptance or universal reader comprehension.

## NAACL 2027 long-paper target

Checked against the [main-conference call](https://2027.naacl.org/calls/main_conference_papers/)
and [ARR submission requirements](https://aclrollingreview.org/cfp) on September 10, 2026.
The ARR deadline is October 12, 2026; NAACL commitment is December 23, 2026
(both 23:59 UTC−12). ARR review and venue commitment are separate steps.

- [x] Official anonymous ACL review style, eight-page main text, required
  Limitations after the conclusion, and two-column appendices after references.
- [x] No author-identifying contact details, affiliations, private workspace
  paths, or project-repository links in the PDF. Public third-party citations
  are retained. Supplemental code/data still require their own anonymity check.
- [x] Superseded preprint citations for RAG, LightRAG, GFM-RAG, MultiHop-RAG,
  and LinearRAG updated to verified conference publications in Markdown and BibTeX.
- [ ] Complete the Responsible NLP Checklist with section references and truthful
  explanations; confirm data/model licenses and all author-specific answers.
- [ ] Disclose AI writing/coding assistance in that checklist. Known assistance
  includes literature lookup, manuscript editing, analysis code, and typesetting.
  Authors must review its scope. The anonymous review PDF omits acknowledgements;
  an optional ethics section is not required merely for formatting compliance.
- [ ] Finalize authors, OpenReview profiles, reviewer registration, preprint
  status, and any resubmission/overlap declarations. No submission has been made.

## Content-reduction priorities

The paper must remain understandable and assessable without its appendices.
Use the following order when new results make the main text exceed eight pages:

1. **Remove repetition.** Explain each operation where it is defined; later
   sections interpret it or refer back. Cut revision history and repeated status
   prose before cutting scientific information.
2. **Shorten prose without changing scope.** Preserve the actor, operation,
   comparison, measured quantity, and qualification needed to interpret a claim.
   Compress repeated examples and captions, while retaining units and definitions.
3. **Move reproducibility details, not central evidence.** Full prompts,
   ancillary settings, extra examples, and supplementary reader/length diagnostics
   may stay in appendices. The main text retains the method's decisions and the
   conditions required to assess comparison fairness.
4. **Consolidate overlapping displays.** Combine tables only when their systems,
   populations, units, and interpretation remain clear. Do not delete a model,
   metric, or empty result slot solely to save space.
5. **Narrow the research scope only deliberately.** Removing a benchmark or an
   ablation changes the claims and requires a consistent revision throughout the
   paper. It is not an automatic typesetting decision.

Protect the index-time Q+–Q− matching rule, HOP/NEXT activation and expansion,
query-time selection, official metric definitions and populations, measured
ranking/latency results, native-system comparability conditions, A/B/C contrasts,
link-usefulness analysis, and matched precomputed-versus-online timing design.
Retain the distinction between full-system performance and causal component
attribution. Neither limitations nor inconclusive outcomes are expendable.

Do not shrink margins, body fonts, or captions, modify the official style, hide
claims in Limitations, or move essential ablations out of the main text to meet
the budget. After a cut, verify table data, cross-references, claim support, and
rendered pagination. If the draft already fits, stop cutting.

The current revision condenses setup and repeated ablation prose, expands the
interpretation of measured ranking/coverage/latency, and reserves explicit link
usefulness, timing, and co-evidence connectivity result tables. Method equations, original
system-row values (apart from consistent display rounding), comparison
conditions, and planned analyses are preserved.

## Current-draft checks

- [x] **Stable presentation.** The seven system tables use Prehop, HopRAG,
  MS GraphRAG, LightRAG, GFM-RAG, LinearRAG, Naive RAG in that order. Names,
  metric scales, decimal precision, and missing-value markers are consistent.
- [x] **Result insertion coverage.** Tables 4 and 6–11 reserve unmeasured
  benchmark, representation, link-usefulness, timing, and co-evidence connectivity outcomes.
  The [result register](RESULTS.md#publication-conventions-and-result-insertion)
  maps incoming results to affected tables and prose. No partial score is filled.
- [x] **Insertion/layout check.** Compared original system rows by name after
  reordering; only documented timing rounding changed displayed values. Tested
  representative numeric values and confidence intervals in a separate temporary
  render. The released PDF retains blank outcomes, 15 tables, and an eight-page
  main text (15 pages including limitations, references, and appendices).

- [x] **Research claim.** The abstract, introduction, discussion, and conclusion
  describe reusable index-time passage connections and competitive full-system
  retrieval as a research question. Observed full-system ranking differences are reported; no causal speedup from
  precomputation is claimed.
- [x] **Method definition.** Passage questions, source exclusion, destination
  matching, link activation, candidate scoring, and final selection are defined
  in Section 3; stored connections are distinguished from complete answer paths.
- [x] **Experiment separation.** Section 5.4 and Appendix A distinguish the
  completed Prehop configuration from new A/B/C controls. All conditions retain
  connectivity; A must be rerun under the common policy.
- [x] **System coverage.** All seven primary strategies in
  [the registry](../core/strategy_registry.py) occur exactly once in each of the
  seven system tables (Tables 1–4 and 12–14). HopRAG has unmeasured result slots.
- [x] **Model reporting.** Section 4.2 identifies the generation model and the
  embedding/retrieval models of completed systems without claiming identical
  retrieval backbones. Appendix E records the configured generation settings.
- [x] **Evaluation populations.** The six completed MultiHop-RAG runs share
  2,556 query IDs (2,255 evidence-bearing queries for retrieval), with zero
  terminal query failures. The active HotpotQA release has 1,000 rows, 944 unique original IDs and 9,221 passages; model runs remain unmeasured.
- [x] **Numerical evidence.** Source-file hashes and per-query metric means were
  previously checked for the six retained MultiHop-RAG runs. Main retrieval values,
  latency rounding, and supplementary coverage/group/length values were
  compared with the [recorded evidence](RESULTS.md) and its linked artifacts.
- [x] **Unmeasured outcomes.** Missing benchmark rows and A/B/C result cells
  retain `—`; these markers are not zero scores or projected results.
- [x] **Metric interpretation.** Official benchmark metrics, added literal fact
  coverage, and HotpotQA sentence-level support scores are distinguished. Query latency
  and batch throughput are defined separately. Returned word counts are
  supplementary, not model input-token or consumed-context measurements.
- [x] **Scope and uncertainty.** Limitations cover native system differences,
  single runs, component attribution, literal matching, generated-link errors,
  and pending HotpotQA (HippoRAG corpus) evaluation.
- [x] **Document consistency.** Table numbering (1–15), figure numbering (1–3),
  local links, English prose, and the distinction between completed and planned
  experiments were checked. This is not a fresh external bibliographic audit.
- [x] **Rendered output.** Built the manuscript with the unmodified official ACL
  review template. Checked the two-column layout, native equations, vector
  figures, result cells, captions, citation resolution, embedded fonts, and
  page boundaries. The presentation retains its separate slide layout.


## Academic-detail checks

These checks distinguish methodological information from implementation detail.
Keep details that define a comparison or change the evaluated method; place
necessary low-level choices in one supplementary location.

- [x] **Code-level branches.** Keep the main search flow in Section 3; document supplementary-candidate and score-inheritance rules once in E.1.
- [x] **Operational settings.** Exclude queue, retry, cache, trace, and deployment procedures from the method narrative.
- [x] **Unmotivated constants.** Retain settings that define retrieval, generation, or experiment controls; do not catalogue unrelated runtime defaults.
- [x] **Duplicated reproduction information.** Section 5.4 defines A/B/C; A.2 adds only execution controls and the index-cost qualification.
- [x] **Artifact-management language.** Describe completed results and unmeasured outcomes instead of file-handling workflow.
- [x] **Repeated caveats.** State interpretation limits in Limitations; keep table-specific units and missing-value definitions with tables.
- [x] **Secondary statistics.** Keep returned-word counts in Appendix B.3, with their meaning and measurement limits.
- [x] **Algorithm and prompt fidelity.** Preserve the scientific scoring equation, actual prompt text, model identities, experiment controls, and result values.

## Open completion requirements

- [ ] **Representation experiments.** Execute A/B/C, verify full-query results,
  and fill Tables 6–7. Preserve the agreed controls and report measured outcomes.
- [ ] **Remaining benchmark comparisons.** Complete missing system–dataset
  pairs in Tables 2 and 4 and the corresponding cost/supplementary entries.
  Alternatively, explicitly finalize a narrower submission scope and revise
  the comparison claims and tables consistently.
- [x] **ACL typesetting.** Applied the official anonymous review template,
  including A4, two columns, Times body text, line numbers, and ACL bibliography
  formatting. The main text and its tables/figures fit eight pages; limitations,
  references, and appendices follow. Recheck pagination after new results.
- [x] **Track-specific requirements.** Checked NAACL 2027 main-conference long-paper
  rules and their ARR requirements. Author-side completion remains listed above.
- [ ] **Final submission readiness.** Complete the author-side requirements,
  remaining empirical work or scope revision, and the final reference/package
  audit before submission. Do not equate a compliant layout with completed research.

## Recheck after updates

When a new run completes, recheck its query population, source identity, metrics,
and result slots before updating narrative claims. When tables move, recheck
all references and the PDF. Reopen any completed check affected by a change.

## Presentation checks

The completed checks below describe the previous rendered presentation. Recheck
its dataset counts and supporting-fact coverage against the current manuscript
before distributing a new export.

- [x] Apply the eight academic-detail checks to slide bodies and speaker notes.
- [x] Keep model details and returned-text lengths in the appendix; retain
  retrieval rules and experiment controls needed to explain the method.
- [x] Remove repeated closing summaries and duplicated A/B/C control text.
- [x] Match the manuscript's seven-system scope and include unmeasured table
  rows as `—`; the coverage chart displays completed runs only.
- [x] Replaced the retired benchmark slides with pending HotpotQA (HippoRAG corpus)
  results, added link-usefulness and connection-timing designs on slides 21–22,
  and regenerated the 27-slide presentation and its PDF. Checked changed slide
  layouts, official metric columns, empty results, and speaker notes.
- [x] Preserve completed scores and identify hypothetical examples and
  unexecuted ablations. Check rendered slide boundaries and table placement.

## Figures and export stability

The editable figure sources are SVG. PNG and PDF exports are derived from the
same sources; the presentation embeds PNG to preserve layout independently of
PowerPoint font substitution. Bitmap display stability does not guarantee
legibility at every projection size.

| Figure | Editable source | Manuscript location | Presentation location |
|---|---|---|---|
| Overview | [SVG](../fig/prehop_paper_overview.svg) | Section 3, Figure 1 | Slide 4 |
| Passage-link example | [SVG](../fig/prehop_paper_links.svg) | Appendix D, Figure 3 | Slide 5 |
| Query retrieval | [SVG](../fig/prehop_paper_retrieval.svg) | Section 3.3, Figure 2 | Slide 8 |

- [x] Check arrows against generation, matching, stored-link reuse, and evidence
  pooling. Both passages generate questions in the link example.
- [x] Show original-query three-channel search for every input length, with
  no query-rewrite branch or evidence-conditioned re-search; retain directly retrieved evidence in the final candidate pool.
- [x] Distinguish the single-pass pipeline, completed MultiHop-RAG and pending HotpotQA Prehop results, and unexecuted A/B/C controls.
- [x] Check labels, box fit, aspect ratios, and figure placement in the local
  PNG/PDF exports and the LibreOffice-rendered presentation.
- [x] Keep SVG, PNG, and PDF figure exports synchronized with manuscript and
  presentation captions.
- [ ] Test the PPTX in the actual presentation computer's PowerPoint and check
  readability at the intended projection size. This has not been performed.

## Reading and export files

- [Manuscript](prehop_paper.md) and [ACL review PDF](../prehop_paper.pdf)
- [Presentation](../prehop_presentation.pptx) and [presentation PDF](../prehop_presentation.pdf)
- [Result evidence](RESULTS.md) and [ablation specification](PAPER_ABLATION_DESIGN.md)

The manuscript PDF uses the official ACL review template. The
[build instructions](acl/README.md) describe regeneration from Markdown. The
presentation contains 27 slides; its main experiment section
includes completed MultiHop-RAG and planned HotpotQA (HippoRAG corpus). Recheck figure slide numbers after reordering slides.

- [ ] Complete the link-usefulness and automatic co-evidence analyses specified
  in [the ablation design](PAPER_ABLATION_DESIGN.md#actual-query-link-usefulness).

- [x] Implement the separate connection-timing ablation, including common
  destination resolution, Neo4j destination storage and paired cluster timing analysis.
- [ ] Execute the separate connection-timing ablation in
  [the design](PAPER_ABLATION_DESIGN.md#stored-versus-online-connection-timing).
  Verify destination agreement, matched conditions, preparation cost, and paired
  timing before making a measured precomputation speedup claim.

- [x] Prepare the pinned HippoRAG HotpotQA release, preserving duplicate occurrences, original sentence identities and official scoring rules.
- [ ] Report original-question cluster intervals and automatic co-evidence connectivity for the completed ablations.
- [ ] Complete all seven HotpotQA model runs before filling result cells.

- [x] Completed and verified single-pass Prehop on MultiHop-RAG; replaced its
  quality, timing, and supplementary statistics with the new run. HotpotQA
  and ablation results remain unmeasured.

- [x] Removed earlier Prehop scores, costs, returned-size statistics, confidence
  intervals, chart marks, and comparative conclusions from current paper and
  presentation reporting. Current Prehop values come only from the new
  single-pass run; pending dataset and ablation cells remain unmeasured.

- [x] Rechecked all three figure exports after layout repair: kept each figure
  with its caption in the manuscript PDF, reserved at least 0.29 inches between
  slide images and captions, preserved image aspect ratios, matched embedded
  PNG bytes to current sources, and inspected all three rendered placements.

- [x] Compared vertical, branching, and candidate-set diagram layouts. Selected
  the candidate-set view, aligned all five query-flow arrows as straight segments,
  and matched the manuscript, caption, and slides to direct retrieval plus stored
  HOP/NEXT traversal, duplicate removal, evidence selection, and answer generation.

- [x] Audited the complete current schematic set (Figures 1–3, slides 4/5/8)
  and inspected all 27 rendered slides, including the fictional retrieval
  example and ablation controls. Checked label widths, canvas bounds, aspect
  ratios, embedded source identity, arrows, and caption placement. Clarified
  HOP versus NEXT construction and activation. Superseded, unreferenced SVG and
  drawio schematics are preserved under `fig/archive/`, outside the current set.

- [x] Updated Prehop from the completed original-query single-pass MultiHop-RAG
  run: verified 2,556 unique query IDs, 2,255 retrieval-eligible questions, zero
  terminal failures, corpus/index identity, per-query averages, exact-fact
  coverage, evidence-count groups, and source-index cost attribution. Updated
  manuscript, slides, and exports without reusing old Prehop query measurements.

### Current-result uncertainty and ablation interpretation

- [x] Align current Prehop and five completed comparators by query ID, original
  question, and gold facts; verify source hashes and 2,255 eligible questions.
- [x] Recompute paired MAP@10 and AllFacts@10 intervals from current outputs;
  report LinearRAG intervals in the main paper and all five in RESULTS.md.
- [x] Expand A/B/C result-row labels and state the B–C link-construction contrast.
- [x] Rebuild the ACL PDF and inspect the changed ablation tables.
- [x] Reconstruct complete direct/HOP/NEXT candidate sets from current saved
  pre-selection traces; verify payload hashes and final passage identities.
  Report primary-configuration utility separately in Appendix B.4, including
  low annotated relevance and NEXT overlap; preserve blank A/B/C outcomes.
- [ ] Run A/B/C to measure representation and link-construction effects.
- [ ] Run matched precomputed/online connection resolution to test causal latency
  savings, checking destination agreement and reporting preparation cost.
- [ ] Complete automatic co-evidence connectivity and degree-matched null comparisons; do not interpret these as semantic link correctness.
