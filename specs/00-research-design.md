# 00 · Research Design

| Field | Value |
| --- | --- |
| Status | Draft |
| Depends on | None |
| Blocks | All other specs |
| Last updated | 2026-11-25 |

## 1. Purpose and scope

This spec fixes the research question, the experimental matrix, what is held constant, what is measured, and how failure homogeneity is analyzed. Everything downstream implements it.

**Research question.** Do different search agents and LLM backbones exhibit shared failure mechanisms after accounting for task difficulty and evidence exposure?

### Non-goals

- Ranking agents or models by accuracy. Accuracy is reported, but it is not the result.
- Confirmatory hypothesis testing. This is an exploratory pilot; all statistics are descriptive.
- Separating the effect of model capability from reasoning level across phases (see §9).

## 2. Definitions

| Term | Definition |
| --- | --- |
| Task | One BrowseComp-Plus query with its ground-truth answer and evidence annotations |
| Agent | One of three search-agent architectures: ReAct, Plan-and-Execute, Evidence-Driven |
| Backbone | One LLM, identified by exact API model ID and reasoning settings |
| Cell | One (task, agent, backbone) combination |
| Trajectory | One complete run of a cell, from task prompt to final answer or budget exhaustion |
| Failure | A trajectory whose final answer the judge scores as incorrect, or which gives no answer |
| Infrastructure failure | A trajectory ended by an API error, timeout, or crash. Not a study failure; rerun it |
| Evidence document | A corpus document the benchmark annotates as supporting the answer |
| Gold document | A corpus document the benchmark annotates as containing the answer |

The exact evidence and gold annotations available are confirmed in spec 01.

## 3. Experimental matrix

| Phase | Tasks | Agents | Backbones | Reasoning level | Trajectories |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 · Failure discovery | 10 | 3 | 3 low-cost | High, where supported | 90 |
| 2 · Validation | 5 of the 10 | 3 | 3 mid-tier | Medium, where supported | 45 |
| **Total** | | | | | **135** |

Candidate backbones, with exact IDs confirmed in spec 02:

| Provider | Phase 1 | Phase 2 |
| --- | --- | --- |
| Anthropic | Claude Haiku 5.5 (`claude-haiku-5-5`)³ | Claude Sonnet 5.5 (`claude-sonnet-5-5`) |
| OpenAI | GPT-6 Luna (`gpt-6-luna`)² | GPT-6.1 Terra (`gpt-6.1-terra`) |

² High reasoning support for Luna is not yet confirmed; check in spec 02. No GPT-6 Terra exists, so Phase 2 uses GPT-6.1 Terra.

³ Haiku 5.5 input pricing rises 5× for prompts over 100K tokens; spec 02's cost estimate must account for this.

⁴ Defaults to `minimal` thinking, which Google warns can end multi-step tool use early. The `high` thinking level MUST be set explicitly, or the model will produce artificial stopping failures.

⁵ Introductory price ($0.75 / $3.75 per 1M tokens) ends 2026-12-31; from 2027-01-01 it doubles to $1.50 / $7.50.
| Google | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`)⁴ | Gemini 3.8 Flash (`gemini-3.8-flash`)⁵ |

## 4. Requirements

### Task selection

- **R1.** The 10 study tasks MUST be chosen by the rule in decision D1, applied once with a recorded seed, before any study trajectory runs.
- **R2.** The 5 Phase 2 tasks MUST be chosen by the rule in decision D2 at the same time as R1, before any Phase 1 trajectory runs. Phase 1 outcomes MUST NOT influence this choice.
- **R3.** The task IDs for both phases MUST be stored in `configs/` and covered by the freeze (R9).

### Controls

- **R4.** These MUST be identical for every trajectory in both phases:
  - corpus, retriever, index version, top-k, and document truncation length (D3)
  - tool set and tool schemas (search and document fetch, defined in spec 03)
  - shared system prompt text and answer format; only architecture-specific instructions may differ
  - per-trajectory budget (D4)
  - answer judge and judging prompt (D5)
- **R5.** Prompt development and debugging MUST use BrowseComp-Plus tasks outside the 10 study tasks.
- **R6.** Reasoning level is set to the provider's closest named setting to High (Phase 1) or Medium (Phase 2). The exact parameter values sent MUST be recorded. Sampling parameters a provider does not support for reasoning models are left at provider defaults and recorded as such.
- **R7.** Agents MUST NOT receive ground-truth answers, evidence annotations, or gold labels (see AGENTS.md §3).

### Run policy

- **R8.** Each cell runs once in the main matrix. Replications follow decision D8 and are reported separately.
- **R9.** Before the first Phase 1 study trajectory, the following are frozen in a tagged commit (`freeze-phase1`): this spec with all decisions resolved, task lists, configs, prompts, tools, budgets, and the failure taxonomy (§6). Later changes go in the deviations log (§10).
- **R10.** Infrastructure failures are rerun under identical settings, up to 3 attempts per cell. Cells that still fail are reported as missing, not as study failures.

## 5. Measurements

Per trajectory, as specified in spec 04:

| Measure | Content |
| --- | --- |
| Outcome | Final answer, judge verdict, judge rationale |
| Search behavior | Queries issued, results returned per query, documents fetched, number of tool calls, reformulations |
| Evidence exposure | For each evidence and gold document: whether it was **retrieved** (appeared in any result list), **read** (fetched in full), and **cited** (referenced in the final answer) |
| Planning | Plan and replan events (Plan-and-Execute), evidence-gap statements (Evidence-Driven) |
| Cost | Input, output, and reasoning tokens where available; latency; API cost |
| Annotation | Failure labels with trajectory step references (§6) |

Derived per trajectory: evidence recall at the retrieved, read, and cited levels, and an **exposure stratum**:

| Stratum | Condition |
| --- | --- |
| E0 · Not retrieved | No gold or evidence document was ever retrieved |
| E1 · Retrieved, not read | At least one was retrieved, none was read |
| E2 · Read | At least one gold or evidence document was read |

Exact recall definitions and stratum thresholds are finalized in spec 06.

## 6. Failure taxonomy

Each failed trajectory gets one **primary** label (the earliest decisive failure) and any number of **secondary** labels. Every label MUST cite the trajectory step or steps that support it.

| Label | Operational definition |
| --- | --- |
| Query formulation | Queries miss or distort key constraints of the question, so relevant documents could not reasonably be retrieved |
| Retrieval | Reasonable queries were issued, but the retriever did not return the needed documents |
| Evidence selection | Needed documents were retrieved but not read, or were read and then ignored in favor of weaker sources |
| Reasoning or synthesis | Needed evidence was read, but the agent drew a wrong inference or combined facts incorrectly |
| Verification | A candidate answer was not checked against the question's constraints, or was checked incorrectly |
| Premature commitment | The agent fixed on a candidate early and afterward searched only to confirm it, or dismissed contradicting evidence |
| Stopping | The decision to stop was wrong: stopped while evidence was clearly missing, or used up the budget in unproductive loops |
| Other / unclear | None of the above fits; requires a free-text note |

The annotation guide in spec 06 adds worked examples and tie-break rules, for example between premature commitment and verification.

## 7. Analysis plan

All analyses are exploratory and descriptive. Where intervals are reported, they are bootstrap intervals over tasks.

- **A1 · Outcomes.** Correctness for each cell; accuracy by agent, by backbone, and by phase.
- **A2 · Label distributions.** Primary and secondary label frequencies by agent, by backbone, and by phase.
- **A3 · Within-task homogeneity.** Comparing trajectories on the same task holds task difficulty constant. For each task, compute pairwise similarity of failure labels (D6) for:
  - same agent, different backbone
  - same backbone, different agent
  - different agent and backbone

  Compare each against a permutation baseline that shuffles labels across tasks while keeping each label's overall frequency. Similarity above the baseline indicates shared mechanisms beyond what common labels alone would produce.
- **A4 · Exposure-conditioned homogeneity.** Repeat A3 within each exposure stratum (E0–E2), to check whether shared failures only reflect shared lack of evidence.
- **A5 · Cross-phase persistence.** On the 5 shared tasks, for each (task, agent), compare Phase 1 and Phase 2 primary labels. Report how often a mechanism persists, disappears (the stronger model succeeds), or changes.
- **A6 · Exposure and correctness.** Accuracy by exposure stratum, by agent and backbone.
- **A7 · Case studies.** 3–5 trajectories chosen to illustrate the most shared and the most divergent mechanisms, chosen after A3–A5.

## 8. Acceptance criteria

This spec is `Approved` when:

- [ ] Every decision in §11 has its Resolution column filled in.
- [ ] Spec 06 confirms each taxonomy label has an annotation example.
- [ ] Spec 01 confirms the evidence and gold annotations needed for §5 exist in BrowseComp-Plus.

It is `Verified` when:

- [ ] Task lists for both phases are in `configs/` with the seed and selection script that produced them.
- [ ] The `freeze-phase1` tag exists and contains every item listed in R9.

## 9. Known limitations

- Ten tasks and one run per cell support descriptive findings only.
- High reasoning effort may reduce reasoning-budget limitations but cannot remove weak-model capability differences.
- Provider reasoning levels are not directly equivalent.
- Backbones are matched by each provider's lineup position, not by price or measured capability; prices within a phase differ by up to about 3×.
- Phase 1 and Phase 2 differ in both models and reasoning levels, so cross-phase differences cannot be attributed to model capability alone.
- LLM-assisted annotation may share biases with the backbones being annotated; manual verification (D7) is the mitigation.

## 10. Deviations log

Record any change made after the `freeze-phase1` tag.

| Date | Change | Reason | Trajectories affected |
| --- | --- | --- | --- |
| | | | |

## 11. Open decisions

Each row has a proposed default. Accept it, replace it, or mark it for discussion.

| ID | Decision | Proposed default | Resolution |
| --- | --- | --- | --- |
| D1 | How the 10 difficult tasks are chosen | Use per-query accuracy from published BrowseComp-Plus baseline runs, if available (confirm in spec 01). Keep queries that most baselines answer incorrectly, exclude any with ambiguous answers, then sample 10 with a fixed seed, spread across topics where possible. | **Resolved 2026-11-25, pending spec 01 check.** Hard tasks only; BrowseComp-Plus has no official difficulty levels, so difficulty is derived. Per-query accuracy comes from the `per_query_metrics` of published leaderboard submissions. A task is eligible if at most 1/3 of submissions answer it correctly **and** at least one does, so it is hard but shown to be solvable. Exclude tasks with ambiguous answers. Sample 10 eligible tasks with seed `20261125`, spread across topics where possible. **Fallback** if per-query results are not available: run one fixed, cheap baseline 3 times on each task in a random pool of 50, and treat a task as eligible if it is solved in exactly 1 of the 3 runs; the pool is then used only for selection, and debugging (R5) uses other tasks. |
| D2 | How the 5 Phase 2 tasks are chosen | Random 5 of the 10, fixed seed, drawn together with D1 | **Resolved 2026-11-25.** Uniform random sample of 5 from the 10 study tasks, seed `20261125`. Drawn once by the selection script right after D1, before any Phase 1 run. The 5 task IDs are committed to `configs/` and frozen (R9); they are never redrawn. |
| D3 | Retriever, top-k, document truncation | The benchmark's dense retriever with its prebuilt index if one is available, otherwise BM25. Top-k = 5. Fetched documents truncated to a fixed token length set in spec 01. | |
| D4 | Per-trajectory budget | Max 30 tool calls in total; max wall time 15 minutes; per-trajectory cost cap set in config. On budget exhaustion, the agent is asked once for a final answer. | **Resolved 2026-11-25.** Accepted as proposed. |
| D5 | Answer judge | The judge model and prompt from the official BrowseComp-Plus evaluation, fixed for both phases, with the model ID recorded. | **Resolved 2026-11-25.** Accepted. One judge model, version, and prompt for both phases, never changed between them. |
| D6 | Failure similarity measure | Jaccard similarity on full label sets, plus primary-label agreement rate | **Resolved 2026-11-25.** Accepted as proposed. |
| D7 | Annotation process | An LLM pre-labels each failed trajectory with step citations; the project author verifies every label. A second annotator re-labels 20% if available. Report LLM–human agreement (Cohen's κ) per label. | **Resolved 2026-11-25.** Accepted as proposed. |
| D8 | Replications | 1 run per cell in the main matrix. Separate replication set in Phase 1: 3 tasks × 3 agents × 1 backbone × 2 extra runs = 18 trajectories, to estimate run-to-run variation. | **Resolved 2026-11-25.** Accepted as proposed. |
| D9 | Total paid-API budget | To be set by the project author, in USD, for each phase | **Resolved 2026-11-25, provisional.** Hard cap of $30 for all paid API use, split as: Phase 1 $6, Phase 2 $12, judging and annotation $6, debugging and pilot $3, reserve $3. Prompt caching MUST be enabled for every provider that supports it; the 2026-11-25 estimate is about $20–30 with caching and $50–65 without. Before Phase 1, a pilot of about 3 trajectories per backbone on non-study tasks measures real token use, and the caps are re-checked against it. If the estimate exceeds a cap, revisit D3/D4 (document length, tool-call limit) or the budget before running. |
