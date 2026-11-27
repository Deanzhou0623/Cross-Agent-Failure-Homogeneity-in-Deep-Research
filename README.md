# Cross-Agent Failure Homogeneity in Deep Research

A small research pilot. Do different search agents and LLM backbones fail **for the same reasons** on hard information-seeking tasks, once task difficulty and evidence exposure are accounted for?

The object of study is the *process* behind each failure (searches, documents seen and read, actions, final answer), not just accuracy.

## Design

| Phase | Tasks | Agents | Models | Thinking | Trajectories |
| --- | ---: | ---: | --- | --- | ---: |
| 1 · Failure discovery | 10 (7 hard, 3 moderately hard) | 3 | 3 cheap | High | 90 |
| 2 · Validation | 5 of the 10, chosen in advance | 3 | 3 mid-tier | Medium | 45 |
| **Total** | | | | | **135** |

One run per configuration. Debugging uses tasks outside the 10.

### Models

| Provider | Phase 1 · High | Phase 2 · Medium |
| --- | --- | --- |
| Anthropic | `claude-haiku-5-5` | `claude-sonnet-5-5` |
| OpenAI | `gpt-6-luna` | `gpt-6.1-terra` |
| Google | `gemini-3.5-flash-lite` | `gemini-3.8-flash` |

Models are matched by lineup position (each provider's smallest, then middle model), not by price. IDs and reasoning settings are checked with one real call each before any experiment. Gemini Flash-Lite defaults to `minimal` thinking, so it is set to `high` explicitly. Some providers silently route retired IDs to newer models (`gemini-3.5-flash` → 3.6, `gemini-3.7-flash` → 3.8), so every call logs the model that actually served it, and a mismatch stops the run. If 3.8 Flash is unavailable when Phase 2 starts, `gemini-3.6-flash` (same price) replaces it for the whole phase; the switch is never made mid-phase.

### Agents

All three use the same tools, corpus, prompt format, and budget.

| Agent | Strategy |
| --- | --- |
| ReAct | Think, search or read, observe, repeat |
| Plan-and-Execute | Write a plan, execute its steps, replan when stuck |
| Evidence-Driven | Keep a list of found evidence and open gaps; search for the biggest gap |

### Benchmark and retrieval

- **BrowseComp-Plus**: 830 questions over a fixed corpus of about 100K documents. Retrieval always searches the full corpus.
- **Tools**, as in the official setup: `search` returns the top 5 documents with 512-token snippets; `get_document` returns one document's text.
- **Retriever**: Qwen3-Embedding-8B, as in the stronger published runs: the official corpus index is searched locally, and only each query is embedded through a hosted API (about $0.01 per 1M tokens). Local BM25 is the fallback. Every search response is cached so runs can be replayed.
- **Budget per run**: at most 30 tool calls (10 of them document reads) and 15 minutes. When the budget runs out, the agent is asked once for its final answer.

### Choosing the 10 tasks

BrowseComp-Plus has no difficulty labels. It does publish four complete runs: GPT-5 and o3, each with BM25 and Qwen3-Embedding-8B. We score those runs with the official judge, then:

- **hard**: solved by exactly 1 of the 4 runs;
- **moderately hard**: solved by exactly 2 of the 4.

Each task must have annotated evidence and gold documents. Candidates are shuffled with seed `20261125` and checked by hand for ambiguous answers. The first 7 hard and 3 moderately hard that pass are kept. The Phase 2 subset (3 hard + 2 moderately hard) is drawn with the same seed before any of our runs. Our own models' results never affect the choice.

### Scoring and analysis

- **Correctness**: the official BrowseComp-Plus judge (Qwen3-32B with the official prompt), through a hosted API.
- **Failure labels**: an LLM proposes labels for each failed run, citing specific trace events, and every label is checked by hand. Labels: query formulation, retrieval, evidence selection, evidence utilization, reasoning or synthesis, verification, premature commitment, stopping. A run can have several.
- **Evidence exposure**: for each annotated evidence document, whether it was retrieved, read, or cited.
- **Comparison**: on the same task, how similar the failure labels are across models, across agents, and across phases (Jaccard similarity against a shuffled baseline), plus example cases.

## Logging

Every run writes a JSONL trace with:

- run, task, agent, exact model ID, and reasoning settings;
- every search query, with the ranked docids, scores, and snippets returned;
- every document opened;
- every model action and tool response, plus plan and evidence-list updates;
- the final answer and its judged correctness;
- tokens (input, cached, output, reasoning where reported), latency, and estimated cost.

Reasoning text is logged only as each provider exposes it (for example, summaries). Raw chain of thought is never claimed.

## Budget

A hard cap of **$30** for all API calls, with prompt caching on. The estimate is about $25–35 with caching. It drops to about $20–30 if the Gemini runs use Google AI Pro developer credits. A smoke test on a few non-study tasks measures real costs before the 135 runs, and nothing paid runs without explicit approval.

## Quick start

```bash
uv sync                                    # install
cp .env.example .env                       # then fill in the API keys
uv run pytest                              # offline tests, no keys needed

uv run python -m src.benchmark download    # queries, answers, qrels, published runs -> data/
uv run python -m src.benchmark download-index   # search index + corpus, ~3.4 GB -> data/
uv run python -m src.models check          # one short call per model (~$0.05)
uv run python -m src.benchmark check-retrieval
uv run python -m src.benchmark score-runs --dry-run   # then without --dry-run (~$0.60)
uv run python -m src.benchmark candidates  # then screen configs/task_screening.yaml by hand
uv run python -m src.benchmark finalize    # writes configs/tasks.yaml
uv run python -m src.benchmark sample --model haiku   # one query end to end
uv run python -m src.agents try --agent react --model haiku   # one agent run on a non-study task
```

## Project structure

```text
.
├── README.md
├── AGENTS.md              # how to work in this repo
├── specs/
│   ├── 01-setup.md        # tasks, models, retrieval
│   ├── 02-agents.md       # three agents and trajectory logging
│   ├── 03-experiments.md  # running the two phases with cost limits
│   └── 04-analysis.md     # comparing failed trajectories
├── src/
│   ├── benchmark.py       # data, retrieval, judge, task selection
│   ├── models.py          # one wrapper per provider, plus a mock
│   ├── agents.py          # the three agents and the trace logger
│   ├── runner.py          # runs a phase, enforces budgets, resumes
│   └── analyze.py         # exposure, labels, similarity, examples
├── configs/               # models, tasks, budgets
├── results/               # traces and analysis outputs
└── tests/
```

`data/` holds the downloaded benchmark files and is never committed: BrowseComp-Plus asks that its questions and answers never appear as plain text online.

## Limitations

- Ten tasks and one run per configuration support descriptive findings only.
- Phase 1 and Phase 2 change both the models and the thinking level, so differences between phases are not a pure capability effect.
- Provider thinking levels are not directly equivalent.
- Difficulty is measured on GPT-5 and o3, a proxy for how hard tasks are for our models.
- The only hosted Qwen3-32B judge is fp8. It scores the published runs 3.5–5.5 points below the leaderboard, mostly by rejecting answers that differ only in formatting, so every "incorrect" verdict on our runs is checked by hand.
