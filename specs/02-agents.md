# 02 · Agents

Status: Draft · Depends on: 01

## Goal

Three interchangeable search agents that share one tool layer, one budget, and one trace logger. Done when all three complete one mock task and write a valid trace.

## Requirements

### Shared by all agents (`src/agents.py`)

1. The same `search` and `get_document` tools (spec 01), with identical tool definitions.
2. The same task prompt and answer format, ported from the official BrowseComp-Plus template:

   ```text
   Explanation: ... (cite docids in [brackets])
   Exact Answer: ...
   Confidence: ...%
   ```

   Only the strategy instructions differ between agents.
3. The same budget, from `configs/budget.yaml`: at most 30 tool calls (10 of them `get_document`), 40 model calls, 16,000 output tokens per call (thinking included), 15 minutes per run, and a per-run cost cap. When the budget runs out, the agent gets one final message asking for its answer. The 16K cap is the same for every model; the smoke test (spec 03) checks that no model hits it in more than about 5% of calls.
4. Any model from spec 01 plugs into any agent through config alone.
5. Agents receive only `{query_id, question}`. A test confirms no answer or qrel text reaches a model call.
6. **Served model check.** Every `model_call` logs the model ID the provider reports. If it differs from the requested ID (providers silently route retired IDs to other models), the run stops with an `error`.
7. **Degenerate output.** A call that hits the output cap with almost no visible text (under 200 non-whitespace characters, or one line repeated 20+ times) is logged as an `error` of kind `degenerate_output`, with the raw response, and retried up to 2 times. If it persists, the run ends with status `error` and is rerun later, like any API error. These are counted per model and reported, but never labeled as search failures. (Seen once in spec 01: Flash-Lite produced 32,656 output tokens and an almost empty answer.)

### The three strategies

| Agent | Loop |
| --- | --- |
| **ReAct** | Each turn the model either calls a tool or gives the final answer. |
| **Plan-and-Execute** | A planning call writes a numbered plan of sub-questions (JSON). The executor works through the steps with tools, reporting each step as `done` or `blocked`. It replans only when the plan is **exhausted** (every step done or blocked, and no answer yet) or **clearly blocked** (the executor marks the current step `blocked`, or 3 tool calls in a row on that step return no docid it has not already seen). There is no periodic replanning. Each plan version is logged. |
| **Evidence-Driven** | Keeps a ledger: facts found (each with the supporting docid) and open gaps. After each tool result the model updates the ledger (JSON), then searches for the most important gap. It answers when no gaps remain or the budget runs out. |

Planning and ledger calls count toward the model-call budget, so no agent gets extra tool calls.

### Trajectory logging

This is the scientific foundation of the study, so nothing is skipped.

8. One JSONL file per run, `results/<phase>/<run_id>.jsonl`. Each line is one event, with `seq` (event number), `ts`, and `type`:

| `type` | Fields |
| --- | --- |
| `run_start` | run ID, task ID, phase, agent, exact model ID, reasoning settings as sent, budget, git commit, retriever backend and index version, system prompt and tool definitions |
| `model_call` | the **full prompt** exactly as sent (provider-native messages, every call), visible text, tool calls requested, reasoning summary if the provider returns one, requested and served model ID, stop reason, tokens (input, cached input, output, reasoning), latency, cost |
| `search` | query, then for each of the 5 hits: rank, docid, score, snippet |
| `read` | docid, returned text, `truncated` flag |
| `plan` | plan version and steps (Plan-and-Execute) |
| `ledger` | facts with docids, and open gaps (Evidence-Driven) |
| `budget` | which limit was hit |
| `final_answer` | raw final message, plus parsed explanation, exact answer, confidence, and cited docids |
| `error` | kind (`api`, `tool`, `degenerate_output`, `model_mismatch`), message, raw response where there is one, retry count |
| `run_end` | status (`completed`, `budget_exhausted`, `error`), totals for tokens, cost, tool calls, and wall time |

9. Derived per run (written at `run_end`): every docid **retrieved** (appeared in any search result) and every docid **read** (opened).
10. Each event is written and flushed as it happens, so a crash still leaves a usable partial trace.
11. No API keys or auth headers in traces. Reasoning text is stored only as the provider returns it, in a field named `reasoning_summary`, never as chain of thought.
12. A small exporter writes the official BrowseComp-Plus run format (`query_id`, `tool_call_counts`, `status`, `retrieved_docids`, `result`), so the official tooling can read our runs.

## Design notes

- One file, `src/agents.py`: a `Trace` class (an append-only JSONL writer), a `Budget` class, the shared `ToolBox`, and three functions `react(...)`, `plan_execute(...)`, `evidence_driven(...)`, each about 100–150 lines.
- Full prompts make traces large (about 5–10 MB per run, about 1 GB for 135 runs, because each call resends the growing conversation). They are highly repetitive, so committed traces are xz-compressed before the canary encryption (spec 03). *2026-11-27: xz, not gzip: gzip's 32 KB window misses the repetition (smoke test 38.7 MB → gzip 12.1 MB, xz 1.0 MB).*
- The mock model in `src/models.py` replays a scripted sequence of tool calls and a final answer, so all three agents can be tested without API calls.

## Done when

- [ ] All three agents complete one mock task with the mock model and mock retriever, and each writes a trace that passes a schema check.
- [ ] Budget tests: each limit stops the run and triggers the single final-answer request.
- [ ] Replan test: Plan-and-Execute replans on an exhausted plan and on a blocked step, and not otherwise.
- [ ] Degenerate-output and model-mismatch tests: each is logged as an `error` and handled as above.
- [ ] Leakage test passes: no answer or qrel text in any model input.
- [ ] The tool definitions are byte-identical across the three agents (test).
- [ ] A trace cut off mid-run is still readable.
- [ ] With approval: one real run per agent on a non-study task with the cheapest model (about $0.10 in total), traces reviewed by hand.

## Open questions

Resolved 2026-11-26:

1. ~~Replan trigger.~~ Only when the plan is exhausted or clearly blocked (see the strategy table).
2. ~~Full prompts or a hash.~~ Full prompts on every `model_call`.
