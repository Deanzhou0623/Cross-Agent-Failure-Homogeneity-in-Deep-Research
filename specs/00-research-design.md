# 00 · Research Design

| Field | Value |
| --- | --- |
| Status | Draft |
| Depends on | None |
| Blocks | All other specs |
| Last updated | 2026-11-25 |

## 1. Goal and scope

Fix the research question, the experimental matrix, what is held constant, what is measured, and how failure homogeneity is analyzed. Every other spec implements this one.

**Research question.** Do different search agents and LLM backbones exhibit shared failure mechanisms when solving difficult information-seeking tasks, after accounting for task difficulty and evidence exposure?

The object of study is the *process* that leads to failure, not final-answer accuracy.

**In scope:** 10 BrowseComp-Plus tasks, 3 agent architectures, 6 backbones across 2 phases (135 trajectories), full trajectory logging, LLM-assisted failure labeling with human verification, descriptive similarity analysis, and a report for supervisor review.

### Non-goals

- Ranking agents or models by accuracy. Accuracy is reported but is not the result.
- Confirmatory hypothesis testing. This is an exploratory pilot; all statistics are descriptive.
- Separating model capability from reasoning level across phases (§8).

## 2. Functional requirements

### Task selection

- **R1.** The 10 study tasks MUST be chosen by the rule in D1, applied once with a recorded seed, before any study trajectory runs. Our own models' outcomes MUST NOT influence selection.
- **R2.** The 5 Phase 2 tasks MUST be chosen by the rule in D2 at the same time as R1, before any Phase 1 trajectory runs.
- **R3.** Selected task IDs, their difficulty stratum, and the selection rationale MUST be stored in `configs/tasks.yaml` and covered by the freeze (R10).

### Controls

- **R4.** These MUST be identical for every trajectory in both phases:
  - corpus, retriever, index version, top-k, snippet length, and document truncation length (D3)
  - tool set and tool schemas (`search`, `get_document`; spec 03)
  - shared task prompt and answer format; only architecture-specific instructions may differ
  - per-trajectory budget (D4)
  - answer judge, judge prompt, and judge settings (D5)
- **R5.** Retrieval MUST run over the full BrowseComp-Plus corpus. It MUST NOT be restricted to the gold or evidence documents of the selected tasks.
- **R6.** Prompt development and debugging MUST use BrowseComp-Plus tasks outside the 10 study tasks.
- **R7.** Reasoning level is set to the provider's closest named setting to High (Phase 1) or Medium (Phase 2). The exact parameter values sent MUST be recorded. Unsupported sampling parameters are left at provider defaults and recorded as such.
- **R8.** Agents MUST NOT receive ground-truth answers, evidence annotations, or gold labels (AGENTS.md §3).

### Run policy

- **R9.** Each cell runs once in the main matrix. Replications follow D8 and are reported separately.
- **R10.** Before the first Phase 1 study trajectory, the following are frozen in a tagged commit (`freeze-phase1`): this spec with all decisions resolved, task lists, configs, prompts, tools, budgets, and the failure taxonomy (§3.5). Later changes go in the deviations log (Appendix B).
- **R11.** Infrastructure failures are rerun under identical settings, up to 3 attempts per cell. Cells that still fail are reported as missing, not as study failures.
- **R12.** No paid study run starts before a smoke test and a pilot (D9) have produced a cost estimate that fits the budget.

## 3. Technical design

### 3.1 Definitions

| Term | Definition |
| --- | --- |
| Task | One BrowseComp-Plus query with its ground-truth answer and evidence annotations |
| Agent | One of three architectures: ReAct, Plan-and-Execute, Evidence-Driven |
| Backbone | One LLM, identified by exact API model ID and reasoning settings |
| Cell | One (task, agent, backbone) combination |
| Trajectory | One complete run of a cell, from task prompt to final answer or budget exhaustion |
| Failure | A trajectory whose final answer the judge scores incorrect, or which gives no answer |
| Infrastructure failure | A trajectory ended by an API error, timeout, or crash. Not a study failure; rerun it |
| Evidence document | A corpus document labeled "needed to answer the query" (`qrel_evidence.txt`) |
| Gold document | An evidence document that also contains the final answer (`qrel_golds.txt`) |

### 3.2 Experimental matrix

| Phase | Tasks | Agents | Backbones | Reasoning level | Trajectories |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 · Failure discovery | 10 (7 hard, 3 moderately hard) | 3 | 3 low-cost | High | 90 |
| 2 · Validation | 5 of the 10, preselected | 3 | 3 mid-tier | Medium | 45 |
| **Total** | | | | | **135** |

### 3.3 Backbones

Exact IDs, reasoning controls, pricing, and token accounting are verified by a smoke test in spec 02 before use.

| Provider | Phase 1 · High | Phase 2 · Medium |
| --- | --- | --- |
| Anthropic | Claude Haiku 5.5 (`claude-haiku-5-5`)¹ | Claude Sonnet 5.5 (`claude-sonnet-5-5`) |
| OpenAI | GPT-6 Luna (`gpt-6-luna`)² | GPT-6.1 Terra (`gpt-6.1-terra`)² |
| Google | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`)³ | Gemini 3.8 Flash (`gemini-3.8-flash`)⁴ |

¹ Input price rises 5× for prompts over 100K tokens; the cost estimate must account for this.
² OpenAI's model pages list `reasoning.effort` values `none` to `max` for both. No GPT-6 Terra exists, so Phase 2 uses GPT-6.1 Terra.
³ Defaults to `minimal` thinking, which Google warns can end multi-step tool use early. `high` MUST be set explicitly, or the study records artificial stopping failures.
⁴ Introductory price ($0.75 / $3.75 per 1M tokens) ends 2026-12-31 and doubles from 2027-01-01.

Backbones are matched by lineup position (each provider's smallest model in Phase 1, its middle model in Phase 2), not by price or measured capability.

### 3.4 Measurements

Per trajectory, as specified in spec 04:

| Measure | Content |
| --- | --- |
| Outcome | Final answer, judge verdict, judge rationale |
| Search behavior | Queries issued, ranked results per query, documents opened, tool-call counts, reformulations |
| Evidence exposure | For each evidence and gold document: **retrieved** (appeared in a result list), **read** (opened with `get_document`), **cited** (docid cited in the final explanation) |
| Planning | Plan and replan events (Plan-and-Execute), evidence ledger and gap updates (Evidence-Driven) |
| Cost | Input, cached-input, output, and reasoning tokens where the provider reports them; latency; estimated cost |
| Annotation | Failure labels citing trajectory event IDs (§3.5) |

Reasoning content is logged only as the provider exposes it (for example, summaries). Raw private chain of thought is never claimed.

Derived per trajectory: evidence recall at the retrieved, read, and cited levels, and an **exposure stratum**:

| Stratum | Condition |
| --- | --- |
| E0 · Not retrieved | No evidence or gold document ever appeared in results |
| E1 · Retrieved, not read | At least one appeared in results; none was opened |
| E2 · Read | At least one evidence or gold document was opened |

Exact definitions are finalized in spec 06.

### 3.5 Failure taxonomy

Each failed trajectory gets one **primary** label (the earliest decisive failure) and any number of **secondary** labels. Every label MUST cite the trajectory event IDs that support it.

| Label | Operational definition |
| --- | --- |
| Query formulation | Queries miss or distort key constraints of the question, so needed documents could not reasonably be retrieved |
| Retrieval | Reasonable queries were issued, but the retriever did not return the needed documents |
| Evidence selection | Needed documents appeared in results but were not opened, or weaker sources were preferred over them |
| Evidence utilization | A needed document was opened, but the agent missed, misread, or failed to extract the relevant fact |
| Reasoning or synthesis | The relevant facts were extracted, but were combined or inferred incorrectly |
| Verification | A candidate answer was not checked against the question's constraints, or was checked incorrectly |
| Premature commitment | The agent fixed on a candidate early, then searched only to confirm it or dismissed contradicting evidence |
| Stopping | The decision to stop was wrong: stopped while evidence was clearly missing, or used up the budget in unproductive loops |
| Other / unclear | None of the above fits; requires a free-text note |

Spec 06 adds worked examples and tie-break rules, for example evidence utilization versus reasoning.

### 3.6 Analysis plan

All analyses are descriptive. Intervals, where reported, are bootstrap intervals over tasks.

- **A1 · Outcomes.** Correctness per cell; accuracy by agent, backbone, phase, and difficulty stratum.
- **A2 · Label distributions.** Primary and secondary label frequencies by agent, backbone, and phase.
- **A3 · Within-task homogeneity.** Comparing trajectories on the same task holds task difficulty constant. For each task, compute pairwise label similarity (D6) for:
  1. same agent, different backbone
  2. same backbone, different agent
  3. different agent and backbone

  Compare each against a permutation baseline that shuffles label sets across tasks while keeping label frequencies. Similarity above baseline indicates mechanisms shared beyond what common labels alone produce.
- **A4 · Exposure-conditioned homogeneity.** Repeat A3 within each exposure stratum, to check whether shared failures only reflect a shared lack of evidence.
- **A5 · Cross-phase persistence.** On the 5 shared tasks, compare Phase 1 and Phase 2 primary labels per (task, agent): persists, disappears (Phase 2 succeeds), or changes.
- **A6 · Exposure and correctness.** Accuracy by exposure stratum, agent, and backbone.
- **A7 · Case studies.** 3–5 trajectories that illustrate the most shared and the most divergent mechanisms, chosen after A3–A5.
- **A8 · Controlled-evidence interventions (optional, D10).** On a few failed cases, supply the gold document and rerun, to separate retrieval-side from reasoning-side failure.

## 4. Implementation tasks

| ID | Task | Implemented in |
| --- | --- | --- |
| T1 | Selection script applying D1/D2 with seed, writing `configs/tasks.yaml` | spec 01 |
| T2 | Experiment config pinning backbones, reasoning settings, budgets | specs 02, 05 |
| T3 | Smoke test, pilot, and cost estimate (R12) | specs 02, 05 |
| T4 | Freeze tag `freeze-phase1` after all decisions resolve | this spec |
| T5 | Analyses A1–A8 | spec 06 |

## 5. Interfaces and data contracts

- `configs/tasks.yaml`: `seed`, `difficulty_source`, `phase1_task_ids[10]`, `phase2_task_ids[5]`, and per task: `stratum` (`hard` | `moderate`), per-baseline correctness, `rationale`. Excluded candidates are listed with their reasons. Schema in spec 01.
- `configs/experiment.yaml`: phases, backbones with reasoning settings, agents, budgets. Schema in spec 05.
- Trajectory logs and annotations: spec 04 and spec 06.

## 6. Testing requirements

This spec has no code of its own. Each requirement is tested where it is implemented: R1–R3 in spec 01, R4–R5 and R8 in specs 01 and 03, R7 in spec 02, R9–R12 in spec 05.

## 7. Acceptance criteria

`Approved` when:

- [ ] Every decision in Appendix A has a resolution.
- [ ] Spec 06 gives an annotation example for every taxonomy label.

`Verified` when:

- [ ] `configs/tasks.yaml` exists with 7 hard + 3 moderately hard tasks, 5 Phase 2 tasks, the seed, and the script that produced it.
- [ ] Spec 01 confirms evidence and gold annotations exist for all 10 tasks.
- [ ] The `freeze-phase1` tag exists and contains every item in R10.

## 8. Dependencies and risks

**Dependencies:** none. All other specs depend on this one.

### Known limitations

- Ten tasks and one run per cell support descriptive findings only.
- High reasoning effort may reduce reasoning-budget limitations but cannot remove weak-model capability differences.
- Provider reasoning levels are not directly equivalent.
- Backbones are matched by lineup position; prices within a phase differ by up to about 3×.
- Phase 1 and Phase 2 differ in both models and reasoning levels, so cross-phase differences cannot be attributed to model capability alone.
- Difficulty is measured on two frontier models (GPT-5, o3) and two retrievers. It is a proxy for difficulty for our backbones and retriever, not a direct measure.
- LLM-assisted annotation may share biases with the backbones; 100% human verification (D7) is the mitigation.

### Risks

| Risk | Mitigation |
| --- | --- |
| Model IDs, prices, or reasoning parameters change mid-study | Verify in spec 02 smoke test; record every parameter sent; log deviations |
| Hosted retriever or judge changes behavior | Cache every search response; pin judge provider; validate both against published numbers (spec 01) |
| Cheap backbones fail every task, leaving little variety | 3 moderately hard tasks; exposure strata separate retrieval from reasoning failures |
| Budget overrun | Hard caps, dry-run estimates, pilot before Phase 1 (D9) |

## Appendix A. Decisions

| ID | Decision | Resolution |
| --- | --- | --- |
| D1 | How the 10 tasks are chosen | **Revised 2026-11-25 per project brief; needs review.** 7 hard + 3 moderately hard. BrowseComp-Plus has no official difficulty levels, and the leaderboard publishes only aggregate accuracy. Difficulty therefore comes from the only public per-query baseline results: the four published runs (GPT-5 and o3, each with BM25 and Qwen3-Embedding-8B), scored by the D5 judge. **Hard:** solved by exactly 1 of the 4 runs. **Moderately hard:** solved by exactly 2 of 4. Tasks solved by 0 runs (possibly unsolvable or mis-keyed) and by 3–4 runs are excluded. Each task MUST have at least one evidence and one gold document. Each stratum is shuffled with seed `20261125` and walked in order. A candidate is skipped if its answer is ambiguous on manual review, with the reason recorded, and topics are spread where a topic field exists. Selection stops at 7 hard and 3 moderately hard. |
| D2 | How the 5 Phase 2 tasks are chosen | **Revised 2026-11-25; needs review.** Stratified draw with seed `20261125`: 3 hard + 2 moderately hard from the 10, drawn together with D1 before any Phase 1 run. Never redrawn. |
| D3 | Retriever, top-k, snippet and document length | **Open.** Proposed: top-k = 5 and snippets ≤ 512 tokens (the official BrowseComp-Plus setup); `get_document` text truncated to 4,096 tokens (project cost control). Retriever backend, chosen after spec 01's validation tests: **(a)** Qwen3-Embedding-8B through the hosted NetMind MCP server, if its terms and rate limits allow, with every response cached; **(b)** local BM25 (Pyserini, Java 21); **(c)** local Qwen3-Embedding-8B on Apple silicon. Recommendation: (a), with (b) as fallback. |
| D4 | Per-trajectory budget | **Revised 2026-11-25 (adds document-read limit); needs review.** Max 30 tool calls, of which at most 10 are `get_document`; max 40 model calls; max 32,000 output tokens per model call; max wall time 15 minutes; per-trajectory cost cap in config. On exhaustion the agent is asked once for a final answer. |
| D5 | Answer judge | **Resolved 2026-11-25.** The official judge `Qwen/Qwen3-32B`, with the official prompt and settings (temperature 0.7, top_p 0.8), through one pinned hosted provider (DeepInfra or OpenRouter) with fallback routing disabled. The provider is recorded with every judgment. The same judge scores the published runs (D1) and both phases. It is not one of the six backbones. |
| D6 | Failure similarity measure | **Resolved 2026-11-25.** Jaccard similarity on label sets, plus primary-label agreement rate. |
| D7 | Annotation process | **Resolved 2026-11-25.** An LLM pre-labels each failed trajectory with event citations (model: D11); the project author verifies every label. A second annotator re-labels 20% if available. LLM–human agreement (Cohen's κ) is reported per label. |
| D8 | Replications | **Resolved 2026-11-25.** 1 run per cell. A separate Phase 1 replication set: 3 tasks × 3 agents × 1 backbone × 2 extra runs = 18 trajectories, to estimate run-to-run variation. |
| D9 | Budget | **Resolved 2026-11-25, provisional.** Hard cap of $30 for all paid API use: Phase 1 $6, Phase 2 $13, judging and annotation $5, smoke test / pilot / debugging $4, reserve $2. Prompt caching MUST be enabled wherever supported. Gemini runs MAY be billed to Google AI Pro developer credits, outside the cap. A pilot of about 3 trajectories per backbone on non-study tasks re-checks the estimate before Phase 1. |
| D10 | Controlled-evidence interventions | **Open.** Proposed: optional; at most 10 runs from the reserve, designed in spec 06, run only after the main analysis. |
| D11 | Failure-labeling model | **Open.** Proposed: a Gemini Pro-tier model paid from Google AI Pro developer credits. It is not one of the backbones and costs $0 of the cap, but it shares a provider with two backbones. It labels condensed traces (spec 04). Alternative: GPT-6.1 Sol, about $8. |

## Appendix B. Deviations log

Record any change made after the `freeze-phase1` tag.

| Date | Change | Reason | Trajectories affected |
| --- | --- | --- | --- |
| | | | |
