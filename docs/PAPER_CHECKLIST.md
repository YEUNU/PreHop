# Manuscript Readiness Checklist

This checklist applies to [the manuscript](prehop_paper.md). It separates
correct reporting in the current draft from completion of the research and
submission requirements. A checked item is limited to the evidence stated;
it does not certify acceptance or universal reader comprehension.

## Current-draft checks

- [x] **Research claim.** The abstract, introduction, discussion, and conclusion
  describe reusable index-time passage connections and competitive full-system
  retrieval as a research question. No measured Prehop advantage or speedup is claimed.
- [x] **Method definition.** Passage questions, source exclusion, destination
  matching, link activation, candidate scoring, and final selection are defined
  in Section 3; stored connections are distinguished from complete answer paths.
- [x] **Experiment separation.** Section 5.4 and Appendix A distinguish the
  pending Prehop configuration from new A/B/C controls. All conditions retain
  connectivity; A must be rerun under the common policy.
- [x] **System coverage.** All seven primary strategies in
  [the registry](../core/strategy_registry.py) occur exactly once in each of the
  seven system tables (Tables 1–4 and 8–10). HopRAG has unmeasured result slots.
- [x] **Model reporting.** Section 4.2 identifies the generation model and the
  embedding/retrieval models of completed systems without claiming identical
  retrieval backbones. Appendix E records the configured generation settings.
- [x] **Evaluation populations.** The five completed MultiHop-RAG comparator runs share
  2,556 query IDs (2,255 evidence-bearing queries for retrieval), with zero
  terminal query failures. The prepared HotpotQA population is verified at
  7,405 dev queries and 5,233,329 corpus paragraphs; model runs remain unmeasured.
- [x] **Numerical evidence.** Source-file hashes and per-query metric means were
  previously checked for the five retained MultiHop-RAG comparator runs. Main retrieval values,
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
  and pending HotpotQA fullwiki evaluation.
- [x] **Document consistency.** Table numbering (1–11), figure numbering (1–3),
  local links, English prose, and the distinction between completed and planned
  experiments were checked. This is not a fresh external bibliographic audit.
- [ ] **Rendered output.** Regenerate the [PDF](../prehop_paper.pdf) after the
  prepared HotpotQA counts and annotation-coverage text change. Earlier layout
  checks do not establish synchronization with the current Markdown. This
  readable export is not the final conference template.

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
- [ ] **Conference typesetting.** Apply the target venue's submission template,
  page limits, anonymization rules, and final figure/table layout. The current
  single-column PDF does not satisfy this check by itself.
- [ ] **Submission policy and references.** Check the target track's current
  requirements, required declarations and reproducibility materials, and verify
  final bibliographic metadata before submission. No removed declaration is
  restored merely to mark this item complete.

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
- [x] Replaced the retired benchmark slides with pending HotpotQA fullwiki
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
- [x] Distinguish the single-pass pipeline, pending Prehop results, and unexecuted A/B/C controls.
- [x] Check labels, box fit, aspect ratios, and figure placement in the local
  PNG/PDF exports and the LibreOffice-rendered presentation.
- [x] Keep SVG, PNG, and PDF figure exports synchronized with manuscript and
  presentation captions.
- [ ] Test the PPTX in the actual presentation computer's PowerPoint and check
  readability at the intended projection size. This has not been performed.

## Reading and export files

- [Manuscript](prehop_paper.md) and [readable PDF](../prehop_paper.pdf)
- [Presentation](../prehop_presentation.pptx) and [presentation PDF](../prehop_presentation.pdf)
- [Result evidence](RESULTS.md) and [ablation specification](PAPER_ABLATION_DESIGN.md)

The manuscript PDF is a readable single-column export, not final conference
typesetting. The presentation contains 27 slides; its main experiment section
includes completed MultiHop-RAG and planned HotpotQA fullwiki. Recheck figure slide numbers after reordering slides.

- [ ] Complete the link-usefulness analysis and sampled-link annotation specified
  in [the ablation design](PAPER_ABLATION_DESIGN.md#link-usefulness-analysis-accompanying-abc).

- [x] Implement the separate connection-timing ablation, including common
  destination resolution, frozen-store identity and paired timing analysis.
- [ ] Execute the separate connection-timing ablation in
  [the design](PAPER_ABLATION_DESIGN.md#connection-timing-ablation-stored-lookup-versus-online-matching).
  Verify destination agreement, matched conditions, preparation cost, and paired
  timing before making a measured precomputation speedup claim.

- [x] Verify HotpotQA fullwiki corpus integrity, evaluation split, sentence
  identities and official scorer integration. Preserve the one unavailable
  upstream sentence label and report its coverage separately.
- [ ] Complete all seven HotpotQA model runs before filling result cells.

- [ ] Complete revised single-pass Prehop benchmarks before replacing the
  earlier configuration’s scores, timing, and associated conclusions.

- [x] Removed earlier Prehop scores, costs, returned-size statistics, confidence
  intervals, chart marks, and comparative conclusions from current paper and
  presentation reporting. Kept all Prehop result slots unmeasured.

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
