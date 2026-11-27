# AGENTS.md

How to work in this repository. It is a small research pilot: keep the code compact and the experiment valid.

## Workflow

Work through the specs in order: `01-setup` → `02-agents` → `03-experiments` → `04-analysis`. For each spec:

1. Read it and propose a short plan.
2. Implement it, with tests.
3. Check its "Done when" list and note the evidence for each item.
4. Open one pull request for that spec, then stop until it is reviewed.

Each spec has: Goal, Requirements, Design notes, Done when, Open questions.

## Keep it small

- Plain Python 3.11+ with `uv`, `pytest`, and the official provider SDKs.
- Five modules in `src/`: `benchmark.py`, `models.py`, `agents.py`, `runner.py`, `analyze.py`.
- No frontend, database server, task queue, or deployment. Traces are JSONL files; outputs are CSV, JSON, or Markdown.
- Settings (model IDs, thinking levels, budgets, task IDs) live in `configs/`, not in code.

## Paid API calls

- Tests never call paid APIs; they use the mock model.
- Every command that can spend money supports `--dry-run` with a cost estimate, and stops at the budget caps in `configs/`.
- Paid runs need explicit approval from the project author each time. Never start one on your own.

## Keeping the experiment valid

- Agents never see answers, evidence labels, or gold documents. A test checks this.
- Retrieval always searches the full corpus.
- All three agents get the same tools, corpus, prompt format, and budget.
- Debug on tasks outside the 10 study tasks.
- Don't change prompts, budgets, tools, or the task list after Phase 1 starts. If a change is unavoidable, record it in the README with the date and reason.
- API errors and crashes are not study failures: log them and rerun.
- Never edit raw traces in `results/`; analysis regenerates everything from them.

## Trajectory logging is not optional

Every run logs every search query and its results, every document opened, every action and tool response, the final answer, and token usage and cost. Log reasoning only as the provider returns it, and never call a summary the model's chain of thought.

## Benchmark data

- Decrypted questions, answers, and document text live only in `data/`, which is gitignored.
- Traces contain question and document text. Commit them only encrypted, using the same canary scheme BrowseComp-Plus uses for its own run files.
- API keys come from environment variables and never appear in logs.
