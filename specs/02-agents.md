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
3. The same budget, from `configs/budget.yaml`: at most 30 tool calls (10 of them `get_document`), 40 model calls, 32,000 output tokens per call, 15 minutes per run, and a per-run cost cap. When the budget runs out, the agent gets one final message asking for its answer.
4. Any model from spec 01 plugs into any agent through config alone.
5. Agents receive only `{query_id, question}`. A test confirms no answer or qrel text reaches a model call.

### The three strategies

| Agent | Loop |
| --- | --- |
| **ReAct** | Each turn the model either calls a tool or gives the final answer. |
| **Plan-and-Execute** | A planning call writes a numbered plan of sub-questions (JSON). The executor works through the steps with tools. It replans when a step fails or after every 5 tool calls, and logs each plan version. |
| **Evidence-Driven** | Keeps a ledger: facts found (each with the supporting docid) and open gaps. After each tool result the model updates the ledger (JSON), then searches for the most important gap. It answers when no gaps remain or the budget runs out. |

Planning and ledger calls count toward the model-call budget, so no agent gets extra tool calls.

### Trajectory logging

This is the scientific foundation of the study, so nothing is skipped.

6. One JSONL file per run, `results/<phase>/<run_id>.jsonl`. Each line is one event, with `seq` (event number), `ts`, and `type`:

| `type` | Fields |
| --- | --- |
| `run_start` | run ID, task ID, phase, agent, exact model ID, reasoning settings as sent, budget, git commit, retriever backend |
| `model_call` | prompt messages (or a hash for repeated context), visible text, tool calls requested, reasoning summary if the provider returns one, tokens (input, cached input, output, reasoning), latency, cost |
| `search` | query, then for each of the 5 hits: rank, docid, score, snippet |
| `read` | docid, returned text, `truncated` flag |
| `plan` | plan version and steps (Plan-and-Execute) |
| `ledger` | facts with docids, and open gaps (Evidence-Driven) |
| `budget` | which limit was hit |
| `final_answer` | raw final message, plus parsed explanation, exact answer, confidence, and cited docids |
| `error` | API or tool error, with retry count |
| `run_end` | status (`completed`, `budget_exhausted`, `error`), totals for tokens, cost, tool calls, and wall time |

7. Derived per run (written at `run_end`): every docid **retrieved** (appeared in any search result) and every docid **read** (opened).
8. Each event is written and flushed as it happens, so a crash still leaves a usable partial trace.
9. No API keys or auth headers in traces. Reasoning text is stored only as the provider returns it, in a field named `reasoning_summary`, never as chain of thought.
10. A small exporter writes the official BrowseComp-Plus run format (`query_id`, `tool_call_counts`, `status`, `retrieved_docids`, `result`), so the official tooling can read our runs.

## Design notes

- One file, `src/agents.py`: a `Trace` class (an append-only JSONL writer), a `Budget` class, the shared `ToolBox`, and three functions `react(...)`, `plan_execute(...)`, `evidence_driven(...)`, each about 100–150 lines.
- The mock model in `src/models.py` replays a scripted sequence of tool calls and a final answer, so all three agents can be tested without API calls.

## Done when

- [ ] All three agents complete one mock task with the mock model and mock retriever, and each writes a trace that passes a schema check.
- [ ] Budget tests: each limit stops the run and triggers the single final-answer request.
- [ ] Leakage test passes: no answer or qrel text in any model input.
- [ ] The tool definitions are byte-identical across the three agents (test).
- [ ] A trace cut off mid-run is still readable.
- [ ] With approval: one real run per agent on a non-study task with the cheapest model (about $0.10 in total), traces reviewed by hand.

## Open questions

1. Plan-and-Execute replan trigger: every 5 tool calls plus on step failure, or only on failure?
2. Should `model_call` store full prompts every turn (complete but large) or a hash of the repeated prefix (smaller)? The proposal is a hash, with the full prompt stored once per run.
