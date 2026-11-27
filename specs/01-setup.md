# 01 · Setup

Status: Draft

## Goal

Load the benchmark, choose the 10 tasks, connect a full-corpus retriever and the judge, and configure the six models. Done when one sample query works end to end.

## Requirements

### Benchmark data (`src/benchmark.py`)

1. Download and decrypt the BrowseComp-Plus queries and answers into `data/` (gitignored), using the official decryption scheme. Record the dataset revision in `data/manifest.json`.
2. Load the evidence and gold qrels from the official repo's `topics-qrels/`.
3. Agents get only `{query_id, question}`. Answers and qrels stay in scoring and analysis code.

### Retrieval (`src/benchmark.py`)

4. Two tools, matching the official setup:
   - `search(query)`: top 5 hits from the **full corpus**, each `{rank, docid, score, snippet}`, with the snippet cut to 512 tokens using the Qwen3-0.6B tokenizer;
   - `get_document(docid)`: the document text, cut to 4,096 tokens, with a `truncated` flag.
5. Backend: Qwen3-Embedding-8B, the retriever of the stronger published runs. The official precomputed corpus index (1.6 GB) and the corpus (1.8 GB) are downloaded to `data/` and searched locally by inner product; each query is embedded through OpenRouter's hosted Qwen3-Embedding-8B, pinned to one provider, with the official query prefix. If this fails the check in item 6, use local BM25 instead (official prebuilt index, 2.2 GB, needs Java 21).
   *Changed 2026-11-26: the plan was NetMind's hosted server, but `mcp.netmind.ai` no longer resolves.*
6. Quick check: run 100 raw questions as plain retrieval queries. Evidence Recall@5 should land near the published value (Qwen3-8B 0.145, BM25 0.012).
7. Cache every response on disk, keyed by backend + query (or docid). A `--replay` mode serves from the cache only.

### Judge (`src/benchmark.py`)

8. The official grader prompt, `Qwen/Qwen3-32B`, temperature 0.7, top_p 0.8, through one hosted provider (DeepInfra or OpenRouter, pinned). Store the raw judge output.

### Task selection (`src/benchmark.py`, output in `configs/tasks.yaml`)

9. Download the four published runs (GPT-5 and o3, each with BM25 and Qwen3-8B; about 1.3 GB) and judge all 830 answers in each, for about $0.60. The four accuracies should land within about 2 points of the leaderboard (57.6, 50.5, 71.7, 65.9); this also checks the judge setup.
   *Result 2026-11-26 (OpenRouter → DeepInfra): 53.1, 47.0, 66.1, 60.8, i.e. 3.5–5.5 points low in every run, so the check does not pass. Settings and answer extraction match the official evaluator; every hosted Qwen3-32B is fp8, and it rejects some answers over capitalization or punctuation only. Decision: keep the official judge and add human checks. `show` prints the published runs' verdicts during screening, and in analysis every "incorrect" verdict on our runs is checked by hand before labeling, with any override logged.*
10. Strata: **hard** = solved by exactly 1 of the 4 runs; **moderately hard** = exactly 2. Keep only tasks with at least one evidence and one gold document.
11. Shuffle each stratum with seed `20261125`. Write the ordered candidates to `configs/task_screening.yaml` for a manual ambiguity check (accept or reject, with a reason).
12. Take the first 7 accepted hard and 3 accepted moderately hard tasks. Then draw the Phase 2 subset (3 hard + 2 moderately hard) with the same seed. Write `configs/tasks.yaml` with the IDs, each task's stratum, its per-run results, and a one-line rationale. Rerunning gives the same file.

### Models (`src/models.py`, `configs/models.yaml`)

13. One small wrapper per provider (official SDKs), plus a mock for tests. A common call: messages + tool definitions in, then text, tool calls, reasoning summary (if any), and usage out.
14. Reasoning settings from config:
    - Anthropic: `output_config.effort`;
    - OpenAI: `reasoning.effort`;
    - Gemini: `thinking_level`.

    Phase 1 is `high`; Phase 2 is `medium`.
15. Turn on prompt caching for each provider. Record input, cached-input, output, and reasoning tokens where reported, and compute cost from a dated price table in `configs/models.yaml`.
16. One real call per model (about $0.10 in total) confirms the ID, the reasoning setting, tool calling, and the usage fields.

## Design notes

- One file, `src/benchmark.py`, with small functions: `load_tasks`, `search`, `get_document`, `judge`, `select_tasks`. One file, `src/models.py`: `call_model(config, messages, tools) -> Response`.
- Decrypted data and the retrieval cache live under `data/`. Only task IDs, docids, and metrics are committed.
- A test scans tracked files for answer strings whenever `data/` is present, and fails if any appear.

## Done when

- [ ] `configs/tasks.yaml` has 7 hard + 3 moderately hard tasks and the 5 Phase 2 tasks, and is reproducible.
- [ ] The retrieval check (item 6) and the judge check (item 9) pass, and their numbers are recorded in the PR.
- [ ] Each of the six models answers one call with the correct reasoning setting and usage fields.
- [ ] **One sample query works:** a non-study question goes through `search` → `get_document` → one model call → `judge`, and the full exchange is printed.
- [ ] Tests pass offline with the mock model and no API keys.
- [ ] Total spend at most $2.

## Open questions

1. ~~NetMind's terms and cost.~~ Moot: the server is gone (see item 5).
2. Which judge provider, DeepInfra or OpenRouter? Pick the one whose check (item 9) passes at lower cost.
3. Downloading and decrypting the data needs the project author to run the command, or to allow it in Claude Code's permission settings.
