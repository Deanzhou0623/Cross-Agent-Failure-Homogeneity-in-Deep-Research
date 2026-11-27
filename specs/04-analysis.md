# 04 · Analysis

Status: Draft · Depends on: 03

## Goal

Compare *why* the failed runs failed: across models, across agents, and across phases. Done when a failure summary and example cases are generated from the saved traces, with every label checked by hand.

## Requirements

### Evidence exposure (`src/analyze.py`)

1. For each run, using the qrels (never shown to agents), compute for each evidence and gold document whether it was:
   - **retrieved**: appeared in any search result;
   - **read**: opened with `get_document`;
   - **cited**: its docid appears in the final explanation.
2. Assign each run an exposure level:
   - **E0**: no evidence document ever retrieved;
   - **E1**: retrieved, but none read;
   - **E2**: at least one read.

### Verdict check (before labeling)

3. The hosted Qwen3-32B judge is fp8 and rejects some answers that differ only in format (spec 01: it scored the published runs 3.5–5.5 points below the leaderboard). So every run judged **incorrect** gets a human verdict check first. The reviewer sees the extracted answer next to the correct answer and either confirms the verdict or overrides it to correct, with a short reason (for example `format only`).
   - Overrides go to `results/review/verdict_overrides.csv` (run ID, judge verdict, final verdict, reason, date). `judgments.jsonl` is never edited.
   - The analysis uses the final verdicts and reports how many verdicts were overridden, by model and phase. Only runs still incorrect after this check get failure labels.

### Failure labels

4. Labels (a failed run can have several, with one marked primary):

| Label | Meaning |
| --- | --- |
| Query formulation | Queries miss or distort key constraints, so the needed documents couldn't be found |
| Retrieval | Reasonable queries, but the retriever didn't return the needed documents |
| Evidence selection | Needed documents appeared in results but weren't opened, or weaker sources were preferred |
| Evidence utilization | A needed document was opened, but the relevant fact was missed or misread |
| Reasoning or synthesis | The right facts were found but combined or inferred wrongly |
| Verification | The candidate answer wasn't checked against the question's constraints, or was checked wrongly |
| Premature commitment | Fixed on an answer early, then only searched to confirm it |
| Stopping | Stopped while evidence was clearly missing, or used up the budget going in circles |
| Other | Free-text note required |

   `configs/labeling_guide.md` gives one short example per label and tie-break rules, for example evidence utilization versus reasoning.

5. **LLM pre-labeling:** for each failed run, a labeling model reads a condensed trace (queries, docids, snippets, documents read, plan or ledger updates, final answer) plus the question, the correct answer, and the exposure data. It proposes labels as JSON, each citing the trace event numbers (`seq`) that support it. Citations that point to non-existent events are rejected.
6. **Human review:** a simple Markdown checklist per run (`results/review/<run_id>.md`) shows the proposed labels with the cited events inline. The reviewer accepts, edits, or replaces them. Only reviewed labels are used in the analysis.
   - The review files quote the question, the answer, and documents, so they stay local (gitignored). The reviewed labels are exported to `results/review/labels.csv` (run ID, labels, primary label, cited `seq` numbers, short note) and committed. Notes must not quote the question or answer.
7. Report agreement between the LLM and the reviewer (Cohen's κ per label).

### Comparisons

8. On the same task, the average Jaccard similarity of failure label sets for pairs of runs that share:
   - the agent, with different models;
   - the model, with different agents;
   - neither.

   Each is compared against a shuffled baseline (labels permuted across tasks, 10,000 times, fixed seed). Similarity above the baseline suggests shared mechanisms beyond common labels.
9. The same comparison within each exposure level (E0–E2), to check whether shared failures only reflect a shared lack of evidence.
10. **Phase 1 against Phase 2** on the 5 shared tasks: for each task and agent, does the Phase 1 primary label persist, disappear (Phase 2 succeeds), or change?
11. Also report: accuracy by agent, model, phase, and difficulty stratum; label frequencies; accuracy by exposure level.

### Outputs

12. `results/analysis/`: CSV tables, a few simple charts (PNG), and `failure_summary.md`. The summary holds the main tables, short findings, and 3–5 example cases with trace excerpts: the most shared and the most divergent failures.
13. `python -m src.analyze` regenerates everything from the traces and reviewed labels. No manual steps after review.

### Optional

14. **Controlled-evidence check:** if budget remains, for up to 10 failed runs, put the gold document into the context and rerun once. This separates finding the evidence from using it.
15. **Second-opinion judge:** re-judge a random sample of about 50 final answers (fixed seed) with the Gemini Pro-tier model, and report how often it agrees with Qwen3-32B.

## Design notes

- One file, `src/analyze.py`, using pandas and matplotlib. Labeling calls go through `src/models.py`.
- **Labeling model:** a Gemini Pro-tier model, billed to Google AI Pro developer credits (about $9 estimated; whatever credit is left after the Gemini runs, about $9 after Phase 2). It is not one of the six study models. If the credits run short, the rest of the labeling uses GPT-6.1 Sol from the main budget.
- **Answer judge** stays the official Qwen3-32B (spec 01), for comparability with the leaderboard and long-term reproducibility.

## Done when

- [ ] Every "incorrect" verdict has been checked by hand; overrides are logged with reasons and counted in the summary.
- [ ] Every run still failed after the verdict check has human-reviewed labels, each citing real trace events.
- [ ] `failure_summary.md` and the tables are generated by one command, and regenerating gives identical files.
- [ ] Similarity results include the shuffled baseline, the exposure breakdown, and the Phase 1 against Phase 2 table.
- [ ] 3–5 example cases with trace excerpts are included.
- [ ] Tests on a small synthetic set of labeled runs give the expected similarity, exposure, and κ values.
- [ ] Labeling spend is at most $5, or $0 with credits.

## Open questions

1. Is a second human reviewer available for about 20% of runs, to check label reliability?
2. Run the optional evidence check (item 14)?
