"""Spec 04: evidence exposure, verdict check, failure labels, similarity, and the failure summary.

Committed outputs (results/analysis/, results/review/*.csv) never quote questions, answers, queries,
or documents. Labeling inputs and review files do, so they stay local (gitignored).

    python -m src.analyze                  # regenerate results/analysis/ from traces and reviewed labels
    python -m src.analyze --draft          # same, filling unreviewed runs with pre-labels (marked DRAFT)
    python -m src.analyze prelabel-inputs  # condensed traces of failed runs -> results/review/inputs/
    python -m src.analyze add-labels --rater first|second FILE.json
    python -m src.analyze second-sample    # the 20% sample for the blind second rater
    python -m src.analyze review-files     # checklists -> results/review/runs/ (reviewed ones are kept)
    python -m src.analyze export-labels    # reviewed checklists -> results/review/labels.csv
    python -m src.analyze select-evidence  # failed runs for the controlled-evidence check
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src import agents as ag
from src import benchmark as bm

RESULTS = bm.ROOT / "results"
REVIEW = RESULTS / "review"
OUT = RESULTS / "analysis"
STUDY = ("phase1", "phase2")
LABELS = ["query_formulation", "retrieval", "evidence_selection", "evidence_utilization", "reasoning",
          "verification", "premature_commitment", "stopping", "other"]
LABELER = "claude-opus-5-5 (Claude Code session)"
SEED, N_PERM, SECOND_SHARE = 0, 10_000, 0.2
LABEL_FIELDS = ["run_id", "labels", "primary", "seq", "note", "labeler", "date"]
PAIR_KINDS = ("same agent, different model", "same model, different agent", "neither")


# --------------------------------------------------------------------------- #
# Runs, verdicts, exposure
# --------------------------------------------------------------------------- #

def _read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()] if path.exists() else []


def exposure(events: list[dict], task: bm.TaskRecord) -> dict:
    """Which evidence/gold documents were retrieved, read, and cited; E0 / E1 / E2."""
    target = set(task.evidence_docids) | set(task.gold_docids)
    retrieved = {h["docid"] for e in events if e["type"] == "search" for h in e["hits"]}
    read = {e["docid"] for e in events if e["type"] == "read" and e["found"]}
    final = next((e for e in reversed(events) if e["type"] == "final_answer"), None)
    cited = set(map(str, (final or {}).get("cited_docids") or []))
    ex = {"retrieved": sorted(target & retrieved), "read": sorted(target & read), "cited": sorted(target & cited)}
    ex["level"] = "E2" if ex["read"] else "E1" if ex["retrieved"] else "E0"
    return ex


def load_runs(results: Path = RESULTS, tasks: dict | None = None, phases=STUDY,
              overrides: Path | None = None) -> list[dict]:
    """One row per planned run: judge verdict, final verdict after the hand check, exposure."""
    tasks = tasks if tasks is not None else bm.load_tasks()
    over = {r["run_id"]: r for r in _read_csv(overrides or results / "review" / "verdict_overrides.csv")}
    runs = []
    for phase in phases:
        summary = {r["run_id"]: r for r in _read_csv(results / phase / "summary.csv")}
        for j in _jsonl(results / phase / "judgments.jsonl"):
            s = summary.get(j["run_id"], {})
            r = {"run_id": j["run_id"], "phase": phase, "task_id": j["task_id"], "agent": j["agent"],
                 "model": j["model"], "stratum": s.get("stratum", ""), "judge_correct": bool(j["correct"])}
            o = over.get(j["run_id"])
            r["correct"] = (o["final_verdict"] == "correct") if o else r["judge_correct"]
            r["overridden"] = bool(o) and r["correct"] != r["judge_correct"]
            events = ag.read_trace(results / j["trace"]) if j.get("trace") else []
            r["events"] = events
            final = next((e for e in reversed(events) if e["type"] == "final_answer"), {})
            r["exact_answer"] = (final.get("exact_answer") or "").strip()
            r["forced"] = bool(final.get("forced"))
            r["exposure"] = exposure(events, tasks[j["task_id"]])
            runs.append(r)
    return sorted(runs, key=lambda r: r["run_id"])


def failed(runs: list[dict]) -> list[dict]:
    return [r for r in runs if not r["correct"]]


# --------------------------------------------------------------------------- #
# Labeling inputs (local only: they quote the question, answer, and documents)
# --------------------------------------------------------------------------- #

def _cut(text, n: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[:n] + " …"


def condense(run: dict, task: bm.TaskRecord) -> str:
    gold, ev = set(task.gold_docids), set(task.evidence_docids)

    def tag(d: str) -> str:
        return f"{d}[G]" if d in gold else f"{d}[E]" if d in ev else d

    ex = run["exposure"]
    out = [f"# {run['run_id']}", f"agent: {run['agent']} · model: {run['model']} · phase: {run['phase']}",
           f"QUESTION: {task.question}", f"CORRECT ANSWER: {task.answer}",
           f"evidence docs: {', '.join(map(tag, task.evidence_docids))}  ([G] = gold, contains the answer)",
           f"exposure: {ex['level']} · retrieved {ex['retrieved']} · read {ex['read']} · cited {ex['cited']}",
           f"final answer: {run['exact_answer']!r} · forced: {run['forced']}", ""]
    seen_facts: set[str] = set()
    for e in run["events"]:
        t, n = e["type"], f"#{e['seq']}"
        if t == "model_call":
            bits = [f"{n} model ({e['purpose']})"]
            if e.get("reasoning_summary"):
                bits.append(f"thinking: {_cut(e['reasoning_summary'], 250)}")
            if e.get("text"):
                bits.append(f"text: {_cut(e['text'], 300)}")
            out.append(" | ".join(bits))
        elif t == "search":
            hits = e["hits"]
            out.append(f"{n} search {e['query']!r} -> {', '.join(tag(h['docid']) for h in hits)}")
            out += [f"    {tag(h['docid'])}: {_cut(h.get('snippet'), 200)}" for h in hits if h["docid"] in ev]
        elif t == "read":
            mark = e["docid"] in ev
            out.append(f"{n} read {tag(e['docid'])} found={e['found']}: {_cut(e['text'], 1500 if mark else 150)}")
        elif t == "plan":
            if e.get("steps"):
                out.append(f"{n} plan v{e['version']} ({e.get('reason') or 'initial'}): "
                           + " / ".join(_cut(s["goal"], 100) for s in e["steps"]))
            elif e.get("update"):
                u = e["update"]
                out.append(f"{n} plan step {u['step']} {u['status']} ({u['by']}): {_cut(u.get('finding'), 150)}")
        elif t == "ledger":
            if e.get("parse_error"):
                out.append(f"{n} ledger: (not parsed)")
                continue
            new = [str(f.get("fact") if isinstance(f, dict) else f) for f in e.get("facts") or []]
            new = [f for f in new if f not in seen_facts]
            seen_facts.update(new)
            out.append(f"{n} ledger: +{len(new)} facts, gaps: " + " / ".join(_cut(g, 100) for g in e.get("gaps") or []))
            out += [f"    + {_cut(f, 150)}" for f in new]
        elif t == "context":
            out.append(f"{n} context docs: {', '.join(map(tag, e['docids']))}")
        elif t == "budget":
            out.append(f"{n} budget limit: {e['limit']}")
        elif t == "error":
            out.append(f"{n} error {e['kind']}: {_cut(e['message'], 120)}")
        elif t == "final_answer":
            out.append(f"{n} FINAL (forced={e['forced']}) exact={e['exact_answer']!r} conf={e['confidence']} "
                       f"cited={e['cited_docids']} | {_cut(e['explanation'], 1200)}")
        elif t == "run_end":
            out.append(f"{n} end: {e['status']}")
    return "\n".join(out) + "\n"


def write_prelabel_inputs(runs: list[dict], tasks: dict, review: Path = REVIEW) -> list[Path]:
    paths = []
    for r in failed(runs):
        p = review / "inputs" / f"{r['run_id']}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(condense(r, tasks[r["task_id"]]), encoding="utf-8")
        paths.append(p)
    return paths


def second_sample(runs: list[dict]) -> list[str]:
    ids = sorted(r["run_id"] for r in failed(runs))
    return sorted(random.Random(SEED).sample(ids, round(SECOND_SHARE * len(ids))))


# --------------------------------------------------------------------------- #
# Labels: validation, pre-labels, review checklists, export
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def check_label(row: dict, run: dict, task: bm.TaskRecord) -> list[str]:
    """Problems with one label row (empty if valid)."""
    labels, seqs = row["labels"], row["seq"]
    probs = [f"unknown label {l!r}" for l in labels if l not in LABELS]
    if not labels:
        probs.append("no labels")
    if row["primary"] not in labels:
        probs.append("primary is not one of the labels")
    if "other" in labels and not row.get("note"):
        probs.append("'other' needs a note")
    n_events = len(run["events"])
    probs += [f"seq {s} is not an event of this run" for s in seqs if not 0 <= s < n_events]
    if not seqs:
        probs.append("no cited events")
    note = _norm(row.get("note", ""))
    if len(_norm(task.answer)) >= 3 and _norm(task.answer) in note:
        probs.append("note quotes the answer")
    words = re.findall(r"\w+", task.question.lower())
    if any(" ".join(words[i:i + 6]) in " ".join(re.findall(r"\w+", row.get("note", "").lower()))
           for i in range(max(0, len(words) - 5))):
        probs.append("note quotes the question")
    return probs


def _row_out(row: dict) -> dict:
    return {**row, "labels": ";".join(row["labels"]), "seq": ";".join(map(str, row["seq"]))}


def _row_in(row: dict) -> dict:
    return {**row, "labels": [l for l in row["labels"].split(";") if l],
            "seq": [int(s) for s in str(row["seq"]).split(";") if s]}


def read_labels(path: Path) -> dict[str, dict]:
    return {r["run_id"]: _row_in(r) for r in _read_csv(path)}


def add_labels(items: list[dict], path: Path, runs: list[dict], tasks: dict) -> list[str]:
    """Validate and merge label proposals into a CSV. Returns problems; nothing is written if any."""
    by_id = {r["run_id"]: r for r in failed(runs)}
    rows, probs = read_labels(path), []
    today = dt.date.today().isoformat()
    for it in items:
        run = by_id.get(it["run_id"])
        if run is None:
            probs.append(f"{it['run_id']}: not a failed run")
            continue
        row = {"run_id": it["run_id"], "labels": list(it["labels"]), "primary": it["primary"],
               "seq": [int(s) for s in it["seq"]], "note": it.get("note", "").strip(),
               "labeler": it.get("labeler", LABELER), "date": it.get("date", today)}
        probs += [f"{it['run_id']}: {p}" for p in check_label(row, run, tasks[run["task_id"]])]
        rows[row["run_id"]] = row
    if not probs:
        _write_csv(path, [_row_out(rows[k]) for k in sorted(rows)], LABEL_FIELDS)
    return probs


REVIEW_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)


def _review_block(p: Path) -> dict | None:
    m = REVIEW_BLOCK.search(p.read_text(encoding="utf-8"))
    data = yaml.safe_load(m.group(1)) if m else None
    return data if isinstance(data, dict) else None


def _is_reviewed(p: Path) -> bool:
    data = _review_block(p)
    return bool(data) and str(data.get("reviewed")).lower() in ("yes", "true")


def write_review_files(runs: list[dict], tasks: dict, review: Path = REVIEW) -> int:
    pre = read_labels(review / "prelabels.csv")
    n = 0
    for r in failed(runs):
        p = review / "runs" / f"{r['run_id']}.md"
        if p.exists() and _is_reviewed(p):
            continue
        lab = pre.get(r["run_id"], {"labels": [], "primary": "", "seq": [], "note": ""})
        lines = {int(m.group(1)): line for line in condense(r, tasks[r["task_id"]]).split("\n")
                 if (m := re.match(r"#(\d+) ", line))}
        cited = "\n".join(f"- {lines.get(s, f'#{s} (no summary line)')}" for s in lab["seq"])
        block = yaml.safe_dump({"reviewed": "no", "labels": lab["labels"], "primary": lab["primary"],
                                "seq": lab["seq"], "note": lab["note"]}, sort_keys=False, allow_unicode=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# Review: {r['run_id']}\n\nProposed by {lab.get('labeler', LABELER)}. Labels: "
                     f"{', '.join(LABELS)}. See configs/labeling_guide.md.\n\n## Cited events\n\n{cited}\n\n"
                     f"## Your review\n\nAccept, edit, or replace, then set `reviewed: yes`. "
                     f"The note must not quote the question or answer.\n\n```yaml\n{block}```\n\n"
                     f"## Condensed trace\n\n```text\n{condense(r, tasks[r['task_id']])}```\n", encoding="utf-8")
        n += 1
    return n


def export_labels(runs: list[dict], tasks: dict, review: Path = REVIEW, reviewer: str = "human") -> tuple[int, list[str]]:
    items, probs = [], []
    for p in sorted((review / "runs").glob("*.md")):
        if not _is_reviewed(p):
            continue
        data = _review_block(p)
        items.append({"run_id": p.stem, "labels": data.get("labels") or [], "primary": data.get("primary") or "",
                      "seq": data.get("seq") or [], "note": data.get("note") or "", "labeler": reviewer})
    out = review / "labels.csv"
    probs = add_labels(items, out, runs, tasks)
    return len(items), probs


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def kappa(x: list[bool], y: list[bool]) -> float | None:
    n = len(x)
    if not n:
        return None
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def pair_kind(a: dict, b: dict) -> str | None:
    if a["agent"] == b["agent"] and a["model"] != b["model"]:
        return PAIR_KINDS[0]
    if a["model"] == b["model"] and a["agent"] != b["agent"]:
        return PAIR_KINDS[1]
    if a["model"] != b["model"] and a["agent"] != b["agent"]:
        return PAIR_KINDS[2]
    return None


def similarity(rows: list[dict], strata_key=None, n_perm: int = N_PERM, seed: int = SEED) -> list[dict]:
    """Mean Jaccard of label sets for same-task pairs, by pair kind, against a shuffled baseline.

    rows: failed runs with 'labels'. The baseline permutes label sets across runs (and so across tasks),
    within each stratum if strata_key is given; pairs are then restricted to the same stratum.
    """
    idx = range(len(rows))
    pairs = {k: [] for k in PAIR_KINDS}
    for i, j in itertools.combinations(idx, 2):
        a, b = rows[i], rows[j]
        if a["task_id"] != b["task_id"] or (strata_key and strata_key(a) != strata_key(b)):
            continue
        if kind := pair_kind(a, b):
            pairs[kind].append((i, j))
    groups = defaultdict(list)
    for i in idx:
        groups[strata_key(rows[i]) if strata_key else None].append(i)
    labels = [frozenset(r["labels"]) for r in rows]
    answers = [_norm(r.get("exact_answer", "")) for r in rows]

    def means(lab):
        return {k: sum(jaccard(lab[i], lab[j]) for i, j in ps) / len(ps) if ps else None for k, ps in pairs.items()}

    obs = means(labels)
    rng = random.Random(seed)
    ge, tot = Counter(), defaultdict(float)
    for _ in range(n_perm):
        perm = list(labels)
        for members in groups.values():
            shuffled = [labels[i] for i in members]
            rng.shuffle(shuffled)
            for i, lab in zip(members, shuffled):
                perm[i] = lab
        for k, v in means(perm).items():
            if v is not None:
                tot[k] += v
                ge[k] += v >= obs[k] - 1e-12
    out = []
    for k in PAIR_KINDS:
        ps = pairs[k]
        same = [answers[i] == answers[j] and answers[i] not in ("", "none") for i, j in ps]
        out.append({"pair_kind": k, "pairs": len(ps),
                    "mean_jaccard": None if obs[k] is None else round(obs[k], 3),
                    "baseline": round(tot[k] / n_perm, 3) if ps else None,
                    "p_value": round((ge[k] + 1) / (n_perm + 1), 4) if ps else None,
                    "same_final_answer": f"{sum(same)}/{len(ps)}"})
    return out


def phase_table(runs: list[dict], labels: dict) -> list[dict]:
    """On the shared tasks: does each task x agent's Phase 1 primary label persist, disappear, or change?"""
    shared = sorted({r["task_id"] for r in runs if r["phase"] == "phase2"}, key=int)
    out = []
    for t, a in itertools.product(shared, sorted({r["agent"] for r in runs})):
        cell = {p: [r for r in runs if r["phase"] == p and r["task_id"] == t and r["agent"] == a] for p in STUDY}
        prim = {p: Counter(labels[r["run_id"]]["primary"] for r in cell[p]
                           if not r["correct"] and r["run_id"] in labels) for p in STUDY}
        fails = {p: sum(not r["correct"] for r in cell[p]) for p in STUDY}
        if not fails["phase1"]:
            outcome = "no Phase 1 failure" + (" (new in Phase 2)" if fails["phase2"] else "")
        elif not fails["phase2"]:
            outcome = "disappears"
        elif set(prim["phase1"]) & set(prim["phase2"]):
            outcome = "persists"
        else:
            outcome = "changes"
        fmt = lambda c: "; ".join(f"{k} ×{v}" for k, v in sorted(c.items())) or "—"
        out.append({"task_id": t, "agent": a, "phase1_failed": f"{fails['phase1']}/{len(cell['phase1'])}",
                    "phase1_primary": fmt(prim["phase1"]), "phase2_failed": f"{fails['phase2']}/{len(cell['phase2'])}",
                    "phase2_primary": fmt(prim["phase2"]), "outcome": outcome})
    return out


def accuracy(runs: list[dict], key) -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        groups[key(r)].append(r["correct"])
    return [{"group": g, "correct": sum(v), "runs": len(v), "accuracy": round(sum(v) / len(v), 3)}
            for g, v in sorted(groups.items())]


# --------------------------------------------------------------------------- #
# Controlled-evidence check
# --------------------------------------------------------------------------- #

def select_evidence(runs: list[dict], n: int = 10, seed: int = SEED) -> list[dict]:
    """Up to n failed E0/E1 runs, spread over models (round robin), then tasks and agents, fixed seed."""
    pool = [r for r in failed(runs) if r["exposure"]["level"] in ("E0", "E1")]
    rng = random.Random(seed)
    by_model = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["run_id"]):
        by_model[r["model"]].append(r)
    for rs in by_model.values():
        rng.shuffle(rs)
    picked, agents_used, tasks_used = [], Counter(), Counter()
    while len(picked) < n and any(by_model.values()):
        for m in sorted(by_model):
            if by_model[m] and len(picked) < n:
                rs = by_model[m]
                rs.sort(key=lambda r: (tasks_used[r["task_id"]], agents_used[r["agent"]]))   # stable among ties
                r = rs.pop(0)
                picked.append(r)
                agents_used[r["agent"]] += 1
                tasks_used[r["task_id"]] += 1
    return [{"source_run_id": r["run_id"], "task_id": r["task_id"], "agent": r["agent"], "model": r["model"],
             "exposure": r["exposure"]["level"]} for r in picked]


def evidence_results(results: Path = RESULTS) -> list[dict]:
    path = bm.CONFIGS / "evidence_check.yaml"
    if not path.exists():
        return []
    judged = {j["run_id"]: j for j in _jsonl(results / "evidence" / "judgments.jsonl")}
    out = []
    for s in yaml.safe_load(path.read_text())["runs"]:
        rid = f"evidence-{s['task_id']}-{s['agent']}-{s['model']}"
        j = judged.get(rid)
        out.append({"source_run_id": s["source_run_id"], "exposure": s["exposure"], "model": s["model"],
                    "agent": s["agent"], "rerun": rid,
                    "rerun_correct": "" if j is None else ("yes" if j["correct"] else "no")})
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _md_table(rows: list[dict], fields: list[str] | None = None) -> str:
    if not rows:
        return "_none_\n"
    fields = fields or list(rows[0])
    fmt = lambda v: "—" if v is None or v == "" else str(v)
    return "\n".join(["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
                     + ["| " + " | ".join(fmt(r.get(f)) for f in fields) + " |" for r in rows]) + "\n"


def _outline(r: dict, lab: dict) -> str:
    """A trace excerpt without question, answer, query, or document text."""
    ex, seqs = r["exposure"], set(lab["seq"])
    ev = set(ex["retrieved"])
    parts = []
    for e in r["events"]:
        t = e["type"]
        if t == "search":
            hit = [h["docid"] for h in e["hits"] if h["docid"] in ev]
            parts.append(f"#{e['seq']} search" + (f" → evidence {', '.join(hit)}" if hit else ""))
        elif t == "read":
            parts.append(f"#{e['seq']} read {e['docid']}" + (" (evidence)" if e["docid"] in ex["read"] else ""))
        elif t in ("plan", "budget", "final_answer", "context") or (t == "ledger" and e["seq"] in seqs):
            parts.append(f"#{e['seq']} {t}" + (f" ({e['limit']})" if t == "budget" else "")
                         + (" (forced)" if t == "final_answer" and e["forced"] else ""))
    keep = [p for p in parts if int(p.split()[0][1:]) in seqs or "evidence" in p or "final" in p or "budget" in p]
    return " · ".join(keep[:14]) + (" · …" if len(keep) > 14 else "")


def _chart_labels(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({r["model"] for r in rows})
    counts = {m: Counter(r["primary"] for r in rows if r["model"] == m) for m in models}
    used = [l for l in LABELS if any(c[l] for c in counts.values())]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / max(1, len(models))
    for k, m in enumerate(models):
        ax.bar([i + k * w for i in range(len(used))], [counts[m][l] for l in used], w, label=m)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(used))], used, rotation=30, ha="right")
    ax.set_ylabel("failed runs (primary label)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def _chart_exposure(runs: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for k, phase in enumerate(STUDY):
        acc = {a["group"]: a for a in accuracy([r for r in runs if r["phase"] == phase], lambda r: r["exposure"]["level"])}
        levels = ["E0", "E1", "E2"]
        ax.bar([i + k * 0.4 for i in range(3)], [acc[l]["accuracy"] if l in acc else 0 for l in levels], 0.4,
               label=phase)
    ax.set_xticks([0.2, 1.2, 2.2], ["E0 not retrieved", "E1 retrieved", "E2 read"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def build(results: Path = RESULTS, tasks: dict | None = None, draft: bool = False, out: Path = OUT,
          n_perm: int = N_PERM, charts: bool = True) -> dict:
    tasks = tasks if tasks is not None else bm.load_tasks()
    runs = load_runs(results, tasks)
    review = results / "review"
    reviewed = read_labels(review / "labels.csv")
    pre = read_labels(review / "prelabels.csv")
    second = read_labels(review / "second_rater.csv")
    fails = failed(runs)
    labels = {r["run_id"]: reviewed.get(r["run_id"]) or (pre.get(r["run_id"]) if draft else None) for r in fails}
    labels = {k: v for k, v in labels.items() if v}
    n_reviewed = sum(r["run_id"] in reviewed for r in fails)
    out.mkdir(parents=True, exist_ok=True)

    # per-run table
    run_rows = []
    for r in runs:
        lab = labels.get(r["run_id"], {})
        ex = r["exposure"]
        run_rows.append({"run_id": r["run_id"], "phase": r["phase"], "task_id": r["task_id"], "stratum": r["stratum"],
                         "agent": r["agent"], "model": r["model"], "judge_correct": int(r["judge_correct"]),
                         "correct": int(r["correct"]), "overridden": int(r["overridden"]), "exposure": ex["level"],
                         "evidence_retrieved": len(ex["retrieved"]), "evidence_read": len(ex["read"]),
                         "evidence_cited": len(ex["cited"]), "forced": int(r["forced"]),
                         "labels": ";".join(lab.get("labels", [])), "primary": lab.get("primary", ""),
                         "label_source": "" if not lab else ("reviewed" if r["run_id"] in reviewed else "pre-label")})
    _write_csv(out / "runs.csv", run_rows, list(run_rows[0]))

    acc = {name: accuracy(runs, key) for name, key in {
        "phase_model": lambda r: f"{r['phase']} {r['model']}", "phase_agent": lambda r: f"{r['phase']} {r['agent']}",
        "phase_stratum": lambda r: f"{r['phase']} {r['stratum']}",
        "phase_exposure": lambda r: f"{r['phase']} {r['exposure']['level']}"}.items()}
    for name, rows in acc.items():
        _write_csv(out / f"accuracy_{name}.csv", rows, ["group", "correct", "runs", "accuracy"])
    overrides = [{"group": f"{p} {m}", "overridden": sum(r["overridden"] for r in runs if r["phase"] == p and r["model"] == m),
                  "judged_incorrect": sum(not r["judge_correct"] for r in runs if r["phase"] == p and r["model"] == m)}
                 for p in STUDY for m in sorted({r["model"] for r in runs if r["phase"] == p})]

    labeled = [{**r, **labels[r["run_id"]]} for r in fails if r["run_id"] in labels]
    freq = []
    for l in LABELS:
        row = {"label": l}
        for p in STUDY:
            ps = [x for x in labeled if x["phase"] == p]
            row[f"{p}_primary"] = sum(x["primary"] == l for x in ps)
            row[f"{p}_any"] = sum(l in x["labels"] for x in ps)
        freq.append(row)
    _write_csv(out / "label_frequency.csv", freq, list(freq[0]))

    sims = []
    for p in STUDY:
        ps = [x for x in labeled if x["phase"] == p]
        for s in similarity(ps, n_perm=n_perm):
            sims.append({"phase": p, "exposure": "all", **s})
        for s in similarity(ps, strata_key=lambda x: x["exposure"]["level"], n_perm=n_perm):
            sims.append({"phase": p, "exposure": "same level", **s})
        for lvl in ("E0", "E1", "E2"):
            sub = [x for x in ps if x["exposure"]["level"] == lvl]
            for s in similarity(sub, n_perm=n_perm):
                sims.append({"phase": p, "exposure": lvl, **s})
    _write_csv(out / "similarity.csv", sims, list(sims[0]))

    kap = []
    for name, other in (("first vs second rater", second), ("first vs reviewed", reviewed)):
        ids = sorted(set(pre) & set(other))
        for l in LABELS:
            k = kappa([l in pre[i]["labels"] for i in ids], [l in other[i]["labels"] for i in ids])
            kap.append({"comparison": name, "label": l, "runs": len(ids), "kappa": k})
        ids_p = sum(pre[i]["primary"] == other[i]["primary"] for i in ids)
        kap.append({"comparison": name, "label": "(primary agrees)", "runs": len(ids),
                    "kappa": f"{ids_p}/{len(ids)}" if ids else None})
    _write_csv(out / "agreement.csv", kap, ["comparison", "label", "runs", "kappa"])

    phases = phase_table(runs, labels)
    _write_csv(out / "phase1_vs_phase2.csv", phases, list(phases[0]) if phases else ["task_id"])
    evid = evidence_results(results)
    if evid:
        _write_csv(out / "evidence_check.csv", evid, list(evid[0]))

    if charts and labeled:
        _chart_labels(labeled, out / "label_frequency.png")
    if charts:
        _chart_exposure(runs, out / "accuracy_by_exposure.png")

    # examples: tasks with the most shared and the most divergent failures (within a phase)
    task_sim = []
    for (p, t), grp in itertools.groupby(sorted(labeled, key=lambda x: (x["phase"], x["task_id"])),
                                         key=lambda x: (x["phase"], x["task_id"])):
        grp = list(grp)
        if len(grp) >= 3:
            js = [jaccard(a["labels"], b["labels"]) for a, b in itertools.combinations(grp, 2)]
            task_sim.append((sum(js) / len(js), p, t, grp))
    task_sim.sort(key=lambda x: (-x[0], x[1], x[2]))
    examples = [("most shared", x) for x in task_sim[:3]] + [("most divergent", x) for x in task_sim[::-1][:2]]
    seen, ex_md = set(), []
    for kind, (m, p, t, grp) in examples:
        if (p, t) in seen:
            continue
        seen.add((p, t))
        ex_md.append(f"#### {kind}: task {t}, {p} (mean Jaccard {m:.2f}, {len(grp)} failed runs)\n")
        for x in grp[:4]:
            ex_md.append(f"- **{x['run_id']}** · {x['exposure']['level']} · primary `{x['primary']}`, labels "
                         f"`{', '.join(x['labels'])}`. {x.get('note', '')}\n  - {_outline(x, x)}")
        ex_md.append("")

    sim_all = [s for s in sims if s["exposure"] in ("all", "same level")]
    status = (f"**DRAFT:** {n_reviewed}/{len(fails)} failed runs have reviewed labels; the rest use pre-labels.\n"
              if draft and n_reviewed < len(fails) else f"Labels: {len(labels)}/{len(fails)} failed runs reviewed.\n")
    top = Counter(x["primary"] for x in labeled).most_common(3)
    md = [
        "# Failure summary", "",
        "Generated by `python -m src.analyze`. No question, answer, query, or document text appears here.", "",
        status,
        "## Accuracy (after the verdict check)", "",
        _md_table(acc["phase_model"]), _md_table(acc["phase_agent"]), _md_table(acc["phase_stratum"]),
        "### Verdict check", "",
        f"{sum(r['overridden'] for r in runs)} of {sum(not r['judge_correct'] for r in runs)} \"incorrect\" verdicts "
        "were overridden by hand (`results/review/verdict_overrides.csv`).", "",
        _md_table([o for o in overrides if o["judged_incorrect"]]),
        "## Evidence exposure", "",
        "E0: no evidence document retrieved · E1: retrieved, none read · E2: at least one read.", "",
        _md_table(acc["phase_exposure"]), "![accuracy by exposure](accuracy_by_exposure.png)", "",
        "## Failure labels", "",
        "Most frequent primary labels: " + (", ".join(f"`{l}` ({n})" for l, n in top) or "—") + ".", "",
        _md_table(freq), "![label frequency](label_frequency.png)" if labeled else "", "",
        "## Shared failures", "",
        f"Mean Jaccard of label sets for same-task pairs of failed runs within a phase, against a shuffled baseline "
        f"({n_perm:,} permutations, seed {SEED}). `same level` restricts pairs and permutations to one exposure "
        "level. `same_final_answer`: pairs giving the same wrong answer. Per-level tables: `similarity.csv`.", "",
        _md_table(sim_all),
        "## Phase 1 against Phase 2 (shared tasks)", "", _md_table(phases),
        "## Label agreement (Cohen's κ per label)", "",
        "The second rater is a fresh-context Claude subagent, blind to the first labels; same model family, so this "
        "κ is an upper bound on reliability.", "",
        _md_table([k for k in kap if k["runs"]]) if any(k["runs"] for k in kap) else "_not available yet_\n",
        "## Controlled-evidence check", "",
        "Failed E0/E1 runs rerun once with up to "
        f"{ag.load_budget().get('evidence_max_context_docs', 3)} gold documents in the first prompt.", "",
        _md_table(evid) if evid else "_not run yet_\n",
        "## Example cases", "", "Trace outlines show event numbers and evidence documents only.", "",
        *ex_md,
    ]
    (out / "failure_summary.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    return {"runs": len(runs), "failed": len(fails), "labeled": len(labels), "reviewed": n_reviewed,
            "overridden": sum(r["overridden"] for r in runs)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.analyze")
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["all", "prelabel-inputs", "add-labels", "second-sample", "review-files",
                             "export-labels", "select-evidence"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--rater", choices=["first", "second"], default="first")
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args(argv)
    tasks = bm.load_tasks()
    if args.cmd == "all":
        print(json.dumps(build(tasks=tasks, draft=args.draft), indent=2))
        return
    runs = load_runs(tasks=tasks)
    if args.cmd == "prelabel-inputs":
        print(f"{len(write_prelabel_inputs(runs, tasks))} condensed traces -> {REVIEW / 'inputs'}")
    elif args.cmd == "add-labels":
        path = REVIEW / ("prelabels.csv" if args.rater == "first" else "second_rater.csv")
        probs = add_labels(json.loads(Path(args.file).read_text()), path, runs, tasks)
        if probs:
            sys.exit("not written:\n" + "\n".join(probs))
        print(f"ok -> {path} ({len(read_labels(path))} runs)")
    elif args.cmd == "second-sample":
        print("\n".join(second_sample(runs)))
    elif args.cmd == "review-files":
        print(f"{write_review_files(runs, tasks)} review files -> {REVIEW / 'runs'}")
    elif args.cmd == "export-labels":
        n, probs = export_labels(runs, tasks)
        if probs:
            sys.exit("not written:\n" + "\n".join(probs))
        print(f"{n} reviewed runs -> {REVIEW / 'labels.csv'}")
    elif args.cmd == "select-evidence":
        picked = select_evidence(runs)
        path = bm.CONFIGS / "evidence_check.yaml"
        path.write_text("# Controlled-evidence check (spec 04, item 14), from `python -m src.analyze select-evidence`\n"
                        + yaml.safe_dump({"runs": picked}, sort_keys=False))
        print(f"{len(picked)} runs -> {path}")


if __name__ == "__main__":
    main()
