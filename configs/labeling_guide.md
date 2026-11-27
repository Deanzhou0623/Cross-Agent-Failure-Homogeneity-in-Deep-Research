# Labeling guide (spec 04)

Label only runs that are still incorrect after the verdict check. Give every label that applies, mark one as **primary** (the earliest failure that, if fixed, would most likely have led to the right answer), and cite the trace events (`seq`) that show it. Notes are short and must not quote the question, the answer, queries, or documents. Describe instead: "read the gold doc at #14 but answered with the name of a different person from it".

Markers in the condensed traces: `[G]` gold document (contains the answer), `[E]` other evidence document. Exposure: E0 no evidence retrieved, E1 retrieved but not read, E2 read.

| Label | Use when | Example |
| --- | --- | --- |
| `query_formulation` | Queries drop, distort, or over-combine key constraints, so the needed documents were unlikely to come back | Every query stuffs all six clues into one long string; none targets the most distinctive one |
| `retrieval` | Queries are reasonable and targeted, but the retriever still never returns the needed documents | Focused queries on each clue; no evidence document in any result list (E0) |
| `evidence_selection` | Evidence documents appear in results but are not opened, or weaker sources are preferred | A gold doc is ranked 2nd at #8; the agent opens the 1st and 4th results instead |
| `evidence_utilization` | An evidence document is opened, but the relevant fact is missed or misread | Reads the gold doc at #14, then states it does not mention the needed detail |
| `reasoning` | The needed facts are in hand, but they are combined or inferred wrongly | Has both dates from read documents but computes the wrong interval |
| `verification` | The candidate is not checked against the question's constraints, or checked wrongly | Answers with a candidate that fails a stated constraint the agent never re-checked |
| `premature_commitment` | Fixes on a candidate early, then only searches to confirm it | Names a candidate at #3, every later query adds that name |
| `stopping` | Stops while evidence is clearly missing, or uses up the budget going in circles | Repeats near-identical queries until the tool budget runs out; forced final answer |
| `other` | None of the above; a note is required | |

## Tie-breaks

- **Retrieval vs query formulation:** look at the queries. If a careful searcher would have written them, it is `retrieval`; if they miss or distort the constraints, `query_formulation`. Both can apply.
- **Evidence selection vs utilization:** was the needed document opened? Not opened → `selection`; opened → `utilization`.
- **Utilization vs reasoning:** is the needed fact stated in the agent's own text or ledger? Not stated → `utilization`; stated but combined wrongly → `reasoning`.
- **Premature commitment vs verification:** commitment is about how the search was steered; verification is about the final check. A run that locked on early and never checked usually gets both, with `premature_commitment` primary.
- **Stopping:** add it when a forced final answer (budget limit) came while gaps were open, or when the agent gave up with "unable to determine" and budget left. It is primary only when the search was on track.
- **No answer:** an empty or "None" answer is still labeled by why the evidence was not found or used.
