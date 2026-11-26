# AGENTS.md

Rules for anyone, human or coding agent, implementing this repository. Read [README.md](README.md) and the relevant spec in [specs/](specs/) before changing code.

## 1. Spec-Driven Development

No implementation without an approved spec.

### Spec lifecycle

Each spec has a `Status` field in its header:

| Status | Meaning | Who changes it |
| --- | --- | --- |
| `Draft` | Being written; open decisions remain | Author |
| `Approved` | All decisions resolved; ready to implement | Human reviewer |
| `Implemented` | Code and tests exist for every requirement | Implementer |
| `Verified` | All acceptance criteria pass and were reviewed | Human reviewer |

Implement only `Approved` specs. Never mark your own work `Verified`.

### Required spec sections

Every spec in `specs/` contains:

1. **Purpose and scope**, including non-goals
2. **Dependencies** on other specs
3. **Requirements**, numbered (`R1`, `R2`, …), using MUST / SHOULD / MAY
4. **Interfaces**: data schemas, function signatures, file formats
5. **Implementation tasks**, each traceable to requirements
6. **Tests**, each traceable to requirements
7. **Acceptance criteria**, checkable and binary
8. **Open decisions**, empty before approval

### Dependency order

```text
00-research-design
├── 01-benchmark ─────────────┐
├── 02-model-providers ───────┤
└── 04-trajectory-logging ────┼──> 03-agent-harnesses ──> 05-experiment-runner
                              │                                  │
                              └──> 06-failure-analysis <─────────┘
                                          │
                                          v
                                   07-results-report
```

The logging schema (04) is defined before the harnesses (03) because the harnesses emit it. Do not start a spec until its dependencies are at least `Approved`, and do not build on code from a spec that is not `Implemented`.

### Review gate

When you finish a spec's tasks, write a short review note in the PR or a commit message that maps each acceptance criterion to its evidence (test name, output file, or command). Stop and wait for human review before starting the next spec.

## 2. Cost and paid-API safety

- Tests MUST NOT call paid APIs. Use the fake provider defined in spec 02.
- Any command that calls a paid API MUST support `--dry-run`, which prints the planned calls and an estimated cost without making them.
- Paid runs require explicit human approval for each invocation. A coding agent never starts one on its own.
- Every paid run MUST enforce the cost cap set in `configs/`. Stop when the cap is reached, never warn and continue.
- Do not run the full Phase 1 or Phase 2 matrix until specs 01–05 are `Verified`.
- Debugging and prompt development MUST use tasks outside the 10 study tasks (see spec 00, R5).

## 3. Experimental integrity

- Agents MUST NOT have access to ground-truth answers, evidence annotations, or gold-document labels. These are used only by scoring and analysis code. Keep a test that fails if they leak into any agent input.
- All three agent architectures use the same tools, corpus, retriever, budgets, and answer format. Architecture-specific logic lives only in the harness.
- Model IDs, reasoning settings, budgets, prompts, and task lists live in `configs/`, never hard-coded.
- After the Phase 1 freeze (spec 00, R9), any change to prompts, budgets, tools, task lists, or taxonomy is a deviation. Record it in the deviations log in spec 00 before running anything under it.
- Never edit or delete raw outputs in `results/`. Derived data is regenerated from raw data by scripts.
- Infrastructure failures (API errors, timeouts, crashes) are not study failures. Log them, exclude them from analysis, and rerun under the same settings.

## 4. Reproducibility

Every run writes a manifest with: git commit hash, dirty-tree flag, config file hashes, exact model IDs and provider parameters as sent, retriever and index version, random seeds, and start/end timestamps. The full per-run record is defined in spec 04.

## 5. Secrets

- API keys come from environment variables only. `.env` files are gitignored.
- Keys and auth headers MUST NOT appear in logs, traces, manifests, or error messages. Logging code redacts them.

## 6. Code conventions

- Python 3.11+, type hints on public functions, `pytest` for tests.
- Keep provider adapters, harnesses, logging, and analysis in separate modules under `src/`. Harnesses depend on a provider interface, not a specific SDK.
- Trajectories are stored as JSON Lines, one event per line, following spec 04.
- One spec per branch and PR where practical. Reference the spec and requirement IDs (for example `03/R4`) in commit messages.
