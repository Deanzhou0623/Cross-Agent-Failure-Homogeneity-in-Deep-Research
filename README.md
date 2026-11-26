# Cross-Agent Failure Homogeneity in Deep Research

An exploratory study of whether search agents and LLM backbones share failure mechanisms on **BrowseComp-Plus**. The focus is not only whether agents fail on the same questions, but whether they fail for similar underlying reasons.

> **Research question:** Do different search agents and LLM backbones exhibit shared failure mechanisms after accounting for task difficulty and evidence exposure?

## Objectives

1. Identify common failure mechanisms across search agents.
2. Compare failure patterns across LLM backbones.
3. Examine whether common failures persist with stronger models.
4. Analyze how evidence exposure and search behavior relate to final-answer correctness.
5. Produce reproducible trajectory-level evidence to guide further research.

## Experimental Design

All agents share the same fixed corpus and retrieval API. Scoring uses the benchmark's answers and evidence annotations.

| Phase | Questions | Agents | Backbones | Thinking¹ | Trajectories |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 · Failure discovery | 10 predefined difficult tasks | 3 | 3 low-cost | High | 90 |
| 2 · Validation | 5 of the 10, selected in advance | 3 | 3 mid-tier | Medium | 45 |
| **Total** | | | | | **135** |

¹ Where supported; provider thinking levels are not directly equivalent.

Phase 1 uses High thinking to reduce avoidable reasoning-budget limitations. Debugging, replications, and evidence interventions are additional runs.

### Candidate Models

| Provider | Phase 1 · High¹ | Phase 2 · Medium¹ |
| --- | --- | --- |
| Anthropic | Claude Haiku 5.5 (`claude-haiku-5-5`) | Claude Sonnet 5.5 (`claude-sonnet-5-5`) |
| OpenAI | GPT-6 Luna (`gpt-6-luna`) | GPT-6.1 Terra (`gpt-6.1-terra`) |
| Google | Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) | Gemini 3.8 Flash (`gemini-3.8-flash`) |

Exact API model IDs and supported reasoning settings must be verified before execution.

### Agent Architectures

| Agent | Strategy |
| --- | --- |
| ReAct | Iterative reasoning and tool use |
| Plan-and-Execute | Planning, execution, and replanning |
| Evidence-Driven | Retrieval guided by evidence gaps |

## Evaluation

Score answers using the official benchmark evaluation procedure. Annotate failed trajectories with multiple labels where appropriate:

- Query formulation
- Retrieval
- Evidence selection
- Reasoning or synthesis
- Verification
- Premature commitment
- Stopping

Compare failure mechanisms across backbones and architectures on the same tasks, across phases, and by relevant evidence exposure. Use trace-supported annotations, LLM-assisted judging, and manual verification.

## Reproducibility

Record exact model IDs and settings, task and agent IDs, queries and search metadata, retrieved versus actually read documents, tool actions, planning updates, final answers, correctness, available token usage, latency, costs, and failure annotations.

## Project Structure

```text
.
├── README.md                      # Overview and research goals
├── AGENTS.md                      # SDD implementation rules
├── specs/
│   ├── 00-research-design.md      # Research question and controls
│   ├── 01-benchmark.md            # BrowseComp-Plus setup
│   ├── 02-model-providers.md      # Adapters and thinking levels
│   ├── 03-agent-harnesses.md      # Agent architectures
│   ├── 04-trajectory-logging.md   # Execution traces
│   ├── 05-experiment-runner.md    # Two-phase orchestration
│   ├── 06-failure-analysis.md     # Taxonomy and judging
│   └── 07-results-report.md       # Analysis and deliverables
├── src/                           # Implementations
├── configs/                       # Model and experiment settings
├── tests/                         # Unit and integration tests
└── results/                       # Traces, annotations, and reports
```

This is the planned layout; implementation files will be added during development.

## Development

Follow **Spec-Driven Development (SDD)** in dependency order: specification → implementation tasks → tests → acceptance criteria → review.

Do not run the full paid experiment until all infrastructure and logging specifications pass.

## Deliverables and Limitations

Deliver a reproducible repository, complete configurations, structured trajectories, failure similarity analysis, qualitative case studies, and a final report with follow-up experiments.

This is an exploratory pilot, not a large-scale statistical study. High thinking may reduce reasoning-budget limitations but cannot remove weak-model capability differences. Because the two phases differ in both models and reasoning levels, cross-phase differences cannot be attributed to model capability alone. A follow-up study may use matched-evidence interventions to test the mechanisms the pilot suggests.
