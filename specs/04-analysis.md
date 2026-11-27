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

5. **LLM pre-labeling:** for each failed run, Claude (`claude-opus-5-5`, in a Claude Code session, no API spend) reads a condensed trace (queries, docids, snippets, documents read, plan or ledger updates, final answer) plus the question, the correct answer, and the exposure data. It proposes labels as JSON, each citing the trace event numbers (`seq`) that support it. Citations that point to non-existent events are rejected. `python -m src.analyze prelabel-inputs` writes the condensed traces (local, gitignored); the proposals are saved as `results/review/prelabels.jsonl` with the labeler, model ID and date.
6. **Human review:** a simple Markdown checklist per run (`results/review/<run_id>.md`) shows the proposed labels with the cited events inline. The reviewer accepts, edits, or replaces them. Only reviewed labels are used in the analysis.
   - The review files quote the question, the answer, and documents, so they stay local (gitignored). The reviewed labels are exported to `results/review/labels.csv` (run ID, labels, primary label, cited `seq` numbers, short note) and committed. Notes must not quote the question or answer.
7. **Second rater:** for a random 20% of failed runs (fixed seed), a fresh Claude subagent labels the run blind: no context from this session and no access to the first labels. Report Cohen's κ per label for first labels against second rater, and for first labels against the reviewed labels. The second rater is the same model family, so its κ is an upper bound on reliability; the README says so.

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

### Controlled-evidence check

14. For up to 10 failed runs, put the gold document into the context and rerun once with the same agent and model. This separates finding the evidence from using it.
    - Pick E0 and E1 failures first (evidence never read), spread over models and agents, fixed seed.
    - Reruns go through `src/runner.py` as phase `evidence`, judged by the same judge. Cap: $2 (Gemini runs bill to the credits).
    - Report how many of them now succeed, by model and exposure level.

## Design notes

- One file, `src/analyze.py`, using pandas and matplotlib. Labeling calls go through `src/models.py`.
- **Labeler:** Claude in a Claude Code session (decided 2026-11-27), so labeling costs nothing and no Gemini credits are spent. A second-opinion judge is dropped: the hand verdict check (item 3) covers it.
- **Answer judge** stays the official Qwen3-32B (spec 01), for comparability with the leaderboard and long-term reproducibility.

## Done when

- [ ] Every "incorrect" verdict has been checked by hand; overrides are logged with reasons and counted in the summary.
- [ ] Every run still failed after the verdict check has human-reviewed labels, each citing real trace events.
- [ ] `failure_summary.md` and the tables are generated by one command, and regenerating gives identical files.
- [ ] Similarity results include the shuffled baseline, the exposure breakdown, and the Phase 1 against Phase 2 table.
- [ ] 3–5 example cases with trace excerpts are included.
- [ ] Tests on a small synthetic set of labeled runs give the expected similarity, exposure, and κ values.
- [ ] Labeling spend is $0; the controlled-evidence check is at most $2.

## Decisions

1. Second rater: a blind, fresh-context Claude subagent on 20% of failed runs (item 7), since no second human reviewer is available. (2026-11-27)
2. The controlled-evidence check is in scope (item 14). (2026-11-27)
