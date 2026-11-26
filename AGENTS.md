# AGENTS.md

Rules for anyone, human or coding agent, implementing this repository. Read [README.md](README.md) and the relevant spec in [specs/](specs/) before changing code.

Priorities, in order: scientific validity, reproducibility, complete trajectory logging, low experimental cost.

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

1. Goal and scope
2. Functional requirements (numbered `R1`, `R2`, …, using MUST / SHOULD / MAY)
3. Technical design
4. Implementation tasks (`T1`, `T2`, …, each traceable to requirements)
5. Interfaces and data contracts
6. Testing requirements (each traceable to requirements)
7. Acceptance criteria (`AC1`, `AC2`, …, checkable and binary)
8. Dependencies and risks

Open decisions go in a final "Open decisions" section, which must be empty before approval.

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

The logging schema (04) comes before the harnesses (03) because the harnesses emit it. Do not start a spec until its dependencies are `Approved`, and do not build on code from a spec that is not `Implemented`.

### Workflow for each spec

1. Read the spec and its dependencies.
2. Propose an implementation plan.
3. Implement the smallest testable components.
4. Write and run the tests.
5. Verify every acceptance criterion and record the evidence.
6. Update documentation.
7. Report completion and any deviations.

Each spec is delivered as its own branch (`spec/NN-name`) and pull request. The PR description maps each acceptance criterion to its evidence (test name, output file, or command). Stop and wait for human review before starting the next spec.

## 2. Cost and paid-API safety

- Tests MUST NOT call paid APIs. Use the fake provider (spec 02).
- Any command that calls a paid API MUST support `--dry-run`, which prints the planned calls and an estimated cost without making them.
- Paid runs require explicit human approval for each invocation. A coding agent never starts one on its own.
- Every paid run enforces the per-run and total caps in `configs/`. It stops when a cap is reached, never warns and continues.
- Before the 135-trajectory matrix: a smoke test, a pilot, and a cost estimate that fits the budget (spec 00, R12).
- Do not run the Phase 1 or Phase 2 matrix until specs 01–05 are `Verified`.
- Debugging and prompt development use tasks outside the 10 study tasks (spec 00, R6).

## 3. Experimental integrity

- Agents MUST NOT have access to ground-truth answers, evidence annotations, or gold-document labels. These are used only by scoring and analysis code. Keep a test that fails if they reach any agent input.
- Retrieval runs over the full corpus, never only the selected tasks' documents.
- All three agent architectures use the same tools, corpus, retriever, budgets, and answer format. Architecture-specific logic lives only in the harness.
- Model IDs, reasoning settings, budgets, prompts, and task lists live in `configs/`, never hard-coded.
- After the Phase 1 freeze (spec 00, R10), any change to prompts, budgets, tools, task lists, or taxonomy is a deviation. Record it in spec 00, Appendix B, before running anything under it.
- Never edit or delete raw outputs in `results/`. Derived data is regenerated from raw data by scripts.
- Infrastructure failures (API errors, timeouts, crashes) are not study failures. Log them, exclude them from analysis, and rerun under the same settings.
- Log reasoning only as the provider exposes it. Never describe a reasoning summary as the model's chain of thought.

## 4. Benchmark data handling

- BrowseComp-Plus asks that its data never appear as plain text online. Decrypted queries, answers, corpus text, and published run files live only under `data/`, which is gitignored.
- Never commit, paste into issues or PRs, or send to any service other than the configured model, retriever, and judge APIs any decrypted question, answer, or document text.
- Only task IDs, docids, and aggregate metrics may be committed. Trajectory logs under `results/` contain question and document text, so they are committed only in encrypted form or kept out of git, as decided in spec 04.

## 5. Reproducibility

Every run writes a manifest with: git commit hash, dirty-tree flag, config file hashes, exact model IDs and provider parameters as sent, retriever and index version, random seeds, and start/end timestamps. The full per-run record is defined in spec 04.

## 6. Secrets

- API keys come from environment variables only. `.env` files are gitignored.
- Keys and auth headers MUST NOT appear in logs, traces, manifests, or error messages. Logging code redacts them.

## 7. Code conventions

- Python 3.11, managed with `uv`; type hints on public functions; `pytest` for tests.
- Keep the benchmark layer, provider adapters, harnesses, logging, runner, and analysis in separate packages under `src/`. Harnesses depend on the provider interface, not on a vendor SDK.
- Trajectories are JSON Lines, one event per line, following spec 04.
- Reference spec and requirement IDs (for example `03/R4`) in commit messages.
