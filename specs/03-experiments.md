# 03 · Experiments

Status: Draft · Depends on: 01, 02

## Goal

Run Phase 1 (90 runs) and Phase 2 (45 runs) within the budget, and save results that can be reproduced. Done when both phases' results are saved, judged, and reproducible from the committed configs.

## Requirements

### Runner (`src/runner.py`)

1. `python -m src.runner --phase 1|2 [--dry-run] [--only AGENT/MODEL/TASK]`. It reads `configs/tasks.yaml`, `configs/models.yaml`, and `configs/budget.yaml`, and runs every (task, agent, model) combination for the phase, one run each.
2. `--dry-run` lists the runs and estimates the cost from the smoke-test token counts (item 6), making no API calls.
3. **Budget caps:** a per-run cap (from spec 02), a per-phase cap (Phase 1 $6, Phase 2 $13), and a total cap of $30 across all spending. The runner checks spending before each run and stops cleanly when a cap is reached.
4. **Resume:** the run ID is `phase-task-agent-model`. A run with a completed trace is skipped, so re-running a phase only fills in what's missing.
5. **Errors:** an API error or crash is retried up to 3 times with the same settings. If it still fails, it is recorded as `missing`, not as a study failure.
6. Paid runs need `--confirm` and a typed confirmation showing the estimated cost. Nothing paid starts automatically.

### Order of work

7. **Smoke test** (about $2–3): about 3 runs per model, on tasks outside the 10, spread across the three agents (18 runs). Report the real tokens and cost per run, and update the Phase 1 and Phase 2 estimates. Stop and get approval if the estimate goes over a cap.
8. **Freeze:** commit the configs and prompts, then tag `phase1-start`. Any later change is written in the README's change log with its date and reason.
9. **Phase 1:** 90 runs.
10. **Phase 2:** 45 runs, on the 5 preselected tasks.

### After each phase

11. Judge every final answer with the spec 01 judge. Store the verdicts in `results/<phase>/judgments.jsonl`.
12. Write `results/<phase>/summary.csv`, one row per run: correct or not, status, tool calls, documents read, tokens, cost, wall time, plus a cost report (estimated against actual, by model).
13. Encrypt the traces with the BrowseComp-Plus canary scheme before committing them (AGENTS.md). A script decrypts them for analysis.

## Design notes

- One file, `src/runner.py`, of a few hundred lines. Runs go one at a time, with optional light concurrency per provider (`--workers 3`). No queue or database.
- Every trace records the git commit and config hashes, so any result can be traced back to the exact code and settings.
- **Gemini runs are billed to Google AI Pro developer credits** (about $15 available; estimated Gemini spend $6–8 with caching), through a project on the paid API tier. The free tier is not used: its daily limits are too low, and free-tier data may be used by Google. The runner tracks Gemini spend against its own $15 cap, separately from the $30 cap, which covers Anthropic, OpenAI, and judging.

## Done when

- [ ] A dry run lists 90 Phase 1 and 45 Phase 2 runs with a cost estimate.
- [ ] Mock-model tests pass: resume skips completed runs; caps stop the runner; errors retry and then mark the run `missing`.
- [ ] The smoke test is done and its report is in the PR. The updated estimate fits within the caps, or the author has approved a change.
- [ ] Phase 1 and Phase 2 traces, judgments, and summaries are saved in `results/`, with no unexplained `missing` runs.
- [ ] Reproducible: re-running the judge and summary steps on the saved traces gives identical `judgments.jsonl` and `summary.csv`, and the `--replay` retrieval cache serves every search the runs made.
- [ ] Total spending is within $30, or the author has approved a change.

## Open questions

1. Run Phase 2 before 2027-01-01? Gemini 3.8 Flash doubles in price on that date, which adds about $2 to Phase 2.
