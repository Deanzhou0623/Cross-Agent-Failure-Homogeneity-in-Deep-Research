"""Experiment runner (spec 03): plan, cap, run, resume, judge, summarize, pack, verify.

    python -m src.runner run --phase smoke|1|2 --dry-run
    python -m src.runner run --phase 1 --confirm [--workers 3] [--only AGENT/MODEL/TASK]
    python -m src.runner judge --phase 1        # results/<phase>/judgments.jsonl
    python -m src.runner summarize --phase 1    # results/<phase>/summary.csv + cost_report.json
    python -m src.runner pack --phase 1         # xz + canary-encrypt traces and judgments for committing
    python -m src.runner unpack --phase 1       # decrypt them back
    python -m src.runner verify --phase 1       # judge/summary are reproducible; replay cache serves every search

Layout: results/<phase>/<run_id>/attempt-<n>.jsonl, run_id = <phase>-<task>-<agent>-<model>.
A run is done when one attempt ends `completed` or `budget_exhausted`. Traces are never modified.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import lzma
import hashlib
import io
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from src import agents as ag
from src import benchmark as bm
from src import models

RESULTS = bm.ROOT / "results"
PHASES = {"smoke": "smoke", "1": "phase1", "2": "phase2", "phase1": "phase1", "phase2": "phase2"}
DONE = ("completed", "budget_exhausted")
CONFIG_FILES = ("models.yaml", "budget.yaml", "tasks.yaml", "benchmark.yaml")


@dataclass(frozen=True)
class RunSpec:
    phase: str
    task_id: str
    agent: str
    model: str

    @property
    def run_id(self) -> str:
        return f"{self.phase}-{self.task_id}-{self.agent}-{self.model}"


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

def _yaml(name: str) -> dict:
    return yaml.safe_load((bm.CONFIGS / name).read_text())


def smoke_task_ids() -> list[str]:
    """2 hard + 1 moderately hard candidates that were never screened: realistic difficulty, never study tasks."""
    screening = _yaml("task_screening.yaml")["candidates"]
    pending = [c for c in screening if c["decision"] == "pending"]
    hard = [c["query_id"] for c in pending if c["stratum"] == "hard"][:2]
    moderate = [c["query_id"] for c in pending if c["stratum"] == "moderate"][:1]
    study = set(_yaml("tasks.yaml")["phase1_task_ids"])
    ids = hard + moderate
    assert len(ids) == 3 and not set(ids) & study
    return ids


def plan_runs(phase: str) -> list[RunSpec]:
    mcfg = _yaml("models.yaml")
    agents = list(ag.AGENTS)
    if phase == "smoke":   # 6 models x 3 agents, agent j on smoke task j
        tasks = smoke_task_ids()
        return [RunSpec("smoke", tasks[j], a, m) for p in ("phase1", "phase2") for m in mcfg[p]
                for j, a in enumerate(agents)]
    tasks = _yaml("tasks.yaml")[f"{phase}_task_ids"]
    return [RunSpec(phase, t, a, m) for t in tasks for a in agents for m in mcfg[phase]]


def select(specs: list[RunSpec], only: str | None) -> list[RunSpec]:
    """--only AGENT/MODEL/TASK, each part may be '*'."""
    if not only:
        return specs
    a, m, t = (only.split("/") + ["*", "*"])[:3]
    return [s for s in specs if a in ("*", s.agent) and m in ("*", s.model) and t in ("*", s.task_id)]


# --------------------------------------------------------------------------- #
# Trace bookkeeping and spending
# --------------------------------------------------------------------------- #

def run_dir(spec: RunSpec, results: Path = RESULTS) -> Path:
    return results / spec.phase / spec.run_id


def attempts(spec: RunSpec, results: Path = RESULTS) -> list[Path]:
    return sorted(run_dir(spec, results).glob("attempt-*.jsonl"), key=lambda p: int(p.stem.split("-")[1]))


def _last_event(path: Path) -> dict | None:
    events = ag.read_trace(path)
    return events[-1] if events else None


def done_trace(spec: RunSpec, results: Path = RESULTS) -> Path | None:
    for p in attempts(spec, results):
        end = _last_event(p)
        if end and end["type"] == "run_end" and end["status"] in DONE:
            return p
    return None


def run_state(spec: RunSpec, max_attempts: int, results: Path = RESULTS) -> str:
    if done_trace(spec, results):
        return "done"
    return "missing" if len(attempts(spec, results)) >= max_attempts else "todo"


_COST_CACHE: dict[tuple, dict] = {}


def trace_cost(path: Path) -> dict:
    """Spend recorded in one trace, by billing bucket (works on partial traces too)."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _COST_CACHE:
        out = {"total": 0.0, "google_credits": 0.0}
        events = ag.read_trace(path)
        bucket = "google_credits" if events and events[0].get("provider") == "gemini" else "total"
        for e in events:
            if e["type"] == "model_call":
                out[bucket] += e.get("cost_usd") or 0.0
            elif e["type"] == "search" and not e.get("cache_hit"):
                out["total"] += (e.get("usage") or {}).get("embedding_cost_usd", 0.0)   # OpenRouter
        _COST_CACHE[key] = out
    return _COST_CACHE[key]


def spending(budget: dict, results: Path = RESULTS, judge_cache: Path | None = bm.DATA / "cache" / "judge") -> dict:
    """All recorded spend: every trace under results/, every judge call, plus prior checks."""
    spent = {"total": budget["prior_spend_usd"]["total"], "google_credits": budget["prior_spend_usd"]["google_credits"]}
    by_phase: dict[str, float] = {}
    traces = list(results.glob("*/*/attempt-*.jsonl")) + list(results.glob("debug/*.jsonl"))  # debug: flat files
    for p in traces:
        c = trace_cost(p)
        spent["total"] += c["total"]
        spent["google_credits"] += c["google_credits"]
        phase = "debug" if p.parent.name == "debug" else p.parent.parent.name
        by_phase[phase] = by_phase.get(phase, 0.0) + c["total"] + c["google_credits"]
    if judge_cache and judge_cache.exists():
        for p in judge_cache.glob("*.json"):
            spent["total"] += json.loads(p.read_text(encoding="utf-8")).get("cost_usd") or 0.0
    return {**{k: round(v, 4) for k, v in spent.items()}, "by_phase": {k: round(v, 4) for k, v in by_phase.items()}}


def per_run_cap(spec: RunSpec, budget: dict, mcfg: dict) -> float:
    return budget["cost_cap_usd"].get(spec.phase) or budget["cost_cap_usd"][mcfg["phase"]]


def cap_problem(spec: RunSpec, spent: dict, budget: dict, mcfg: dict, inflight_room: float = 0.0) -> str | None:
    """Stop before a run that could cross a cap: room needed = this run's cap + the caps of runs in flight."""
    caps = budget["caps_usd"]
    room = inflight_room + per_run_cap(spec, budget, mcfg)
    if spent["by_phase"].get(spec.phase, 0.0) + room > caps[spec.phase]:
        return f"{spec.phase} cap ${caps[spec.phase]}"
    bucket = "google_credits" if mcfg.get("billing") == "google_credits" else "total"
    if spent[bucket] + room > caps[bucket]:
        return f"{bucket} cap ${caps[bucket]}"
    return None


# --------------------------------------------------------------------------- #
# Estimates
# --------------------------------------------------------------------------- #

def smoke_cost_per_run(results: Path = RESULTS) -> dict[str, float]:
    """Mean cost of finished smoke runs, by model key."""
    costs: dict[str, list[float]] = {}
    for spec in plan_runs("smoke"):
        p = done_trace(spec, results)
        if p:
            costs.setdefault(spec.model, []).append(_last_event(p)["totals"]["cost_usd"])
    return {m: sum(v) / len(v) for m, v in costs.items()}


def estimate(specs: list[RunSpec], budget: dict, results: Path = RESULTS) -> dict:
    per_model = smoke_cost_per_run(results)
    rows, total = {}, 0.0
    for s in specs:
        mphase = models.load_model(s.model)["phase"]
        unit = per_model.get(s.model, budget["fallback_estimate_usd_per_run"][mphase])
        r = rows.setdefault(s.model, {"runs": 0, "usd_per_run": round(unit, 4),
                                       "source": "smoke" if s.model in per_model else "fallback"})
        r["runs"] += 1
        total += unit
    return {"runs": len(specs), "estimated_usd": round(total, 2), "by_model": rows}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def config_hashes() -> dict[str, str]:
    return {n: hashlib.sha256((bm.CONFIGS / n).read_bytes()).hexdigest()[:16] for n in CONFIG_FILES}


def git_dirty(paths=("src", "configs")) -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=bm.ROOT, capture_output=True, text=True)
    return bool(out.stdout.strip())


class Runner:
    def __init__(self, specs: list[RunSpec], *, budget: dict | None = None, results: Path = RESULTS,
                 tasks: dict | None = None, load_model=models.load_model, make_retriever=None,
                 judge_cache: Path | None = bm.DATA / "cache" / "judge", log=print):
        self.specs, self.results, self.log = specs, results, log
        self.budget = budget or ag.load_budget()
        self.tasks = tasks if tasks is not None else bm.load_tasks()
        self.load_model = load_model
        self.make_retriever = make_retriever or self._default_retriever
        self.judge_cache = judge_cache
        self.lock = threading.Condition()
        self.inflight_room = 0.0          # sum of per-run caps of runs in flight
        self.stop_reason: str | None = None
        self._retriever = None

    def _default_retriever(self):
        if self._retriever is None:
            cfg = bm.load_config()
            self._retriever = bm.Retriever(bm.make_backend(cfg), cfg)
        return self._retriever

    def _claim(self, spec: RunSpec, mcfg: dict) -> bool:
        """Reserve room for one run. If only in-flight runs block a cap, wait for them; else stop."""
        with self.lock:
            while True:
                if self.stop_reason:
                    return False
                spent = spending(self.budget, self.results, self.judge_cache)
                if not cap_problem(spec, spent, self.budget, mcfg, self.inflight_room):
                    self.inflight_room += per_run_cap(spec, self.budget, mcfg)
                    return True
                problem = cap_problem(spec, spent, self.budget, mcfg, 0.0)
                if problem:
                    self.stop_reason = f"stopped before {spec.run_id}: {problem} (spent {spent})"
                    self.lock.notify_all()
                    return False
                self.lock.wait()            # a run in flight will finish and free its room

    def run_one(self, spec: RunSpec) -> str:
        max_attempts = self.budget["max_attempts"]
        state = run_state(spec, max_attempts, self.results)
        if state != "todo":
            return state
        mcfg = self.load_model(spec.model)
        task = self.tasks[spec.task_id].view()
        meta = {"config_sha256": config_hashes(), "git_dirty": git_dirty(), "billing": mcfg.get("billing", "default"),
                "run_spec": spec.__dict__}
        while len(attempts(spec, self.results)) < max_attempts:
            if not self._claim(spec, mcfg):
                return "stopped"
            n = len(attempts(spec, self.results)) + 1
            try:
                summary = ag.run_agent(spec.agent, task, mcfg, self.make_retriever(),
                                       run_dir(spec, self.results) / f"attempt-{n}.jsonl", run_id=spec.run_id,
                                       phase=spec.phase, budget=self.budget, meta={**meta, "attempt": n})
            finally:
                with self.lock:
                    self.inflight_room -= per_run_cap(spec, self.budget, mcfg)
                    self.lock.notify_all()
            self.log(f"{spec.run_id} attempt {n}: {summary['status']} ${summary['cost_usd']:.4f}"
                     + (f" ({summary['error']['kind']})" if summary.get("error") else ""))
            if summary["status"] in DONE:
                return "done"
            if (summary.get("error") or {}).get("kind") == "model_mismatch":
                with self.lock:
                    self.stop_reason = f"model mismatch in {spec.run_id}: {summary['error']['message']}"
                return "stopped"
        return "missing"

    def run_all(self, workers: int = 1) -> dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            states = list(pool.map(self.run_one, self.specs))
        counts: dict[str, int] = {}
        for st in states:
            counts[st] = counts.get(st, 0) + 1
        return {"states": counts, "stop_reason": self.stop_reason,
                "spent": spending(self.budget, self.results, self.judge_cache)}


# --------------------------------------------------------------------------- #
# After a phase: judge, summarize, pack, verify
# --------------------------------------------------------------------------- #

def judge_phase(phase: str, results: Path = RESULTS, judge=None, tasks=None) -> list[dict]:
    budget = ag.load_budget()
    judge = judge or bm.Judge(bm.load_config())
    tasks = tasks if tasks is not None else bm.load_tasks()
    rows = []
    for spec in sorted(plan_runs(phase), key=lambda s: s.run_id):
        p = done_trace(spec, results)
        row = {"run_id": spec.run_id, "task_id": spec.task_id, "agent": spec.agent, "model": spec.model,
               "state": "done" if p else run_state(spec, budget["max_attempts"], results)}
        final = next((e for e in ag.read_trace(p) if e["type"] == "final_answer"), None) if p else None
        if final is None:
            row.update(correct=False, judge=None, note="no final answer")
        else:
            verdict = judge.judge(tasks[spec.task_id], final["raw"])
            row.update(correct=bool(verdict["correct"]), trace=str(p.relative_to(results)), judge=verdict)
        rows.append(row)
    out = results / phase / "judgments.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return rows


SUMMARY_FIELDS = ["run_id", "phase", "task_id", "stratum", "agent", "model", "model_id", "state", "status",
                  "correct", "attempts", "model_calls", "tool_calls", "search", "get_document", "retrieved_docs",
                  "read_docs", "input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens",
                  "cost_usd", "wall_seconds"]


def summarize_phase(phase: str, results: Path = RESULTS) -> dict:
    budget = ag.load_budget()
    strata = {t["query_id"]: t["stratum"] for t in _yaml("tasks.yaml")["tasks"]}
    judgments = {json.loads(l)["run_id"]: json.loads(l)
                 for l in (results / phase / "judgments.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()}
    rows, actual = [], {}
    for spec in sorted(plan_runs(phase), key=lambda s: s.run_id):
        p = done_trace(spec, results)
        n_att = len(attempts(spec, results))
        row = {"run_id": spec.run_id, "phase": phase, "task_id": spec.task_id, "stratum": strata.get(spec.task_id, ""),
               "agent": spec.agent, "model": spec.model, "attempts": n_att,
               "state": "done" if p else run_state(spec, budget["max_attempts"], results),
               "correct": int(bool(judgments.get(spec.run_id, {}).get("correct")))}
        all_cost = sum(sum(trace_cost(a).values()) for a in attempts(spec, results))
        if p:
            events = ag.read_trace(p)
            end, tot = events[-1], events[-1]["totals"]
            tok = tot["tokens"]
            row.update(model_id=events[0]["model_id"], status=end["status"], model_calls=tot["model_calls"],
                       tool_calls=tot["tool_calls"], search=tot["search"], get_document=tot["get_document"],
                       retrieved_docs=len(end["retrieved_docids"]), read_docs=len(end["read_docids"]),
                       input_tokens=tok.get("input", 0), cached_input_tokens=tok.get("cached_input", 0),
                       cache_write_tokens=tok.get("cache_write", 0), output_tokens=tok.get("output", 0),
                       wall_seconds=tot["wall_seconds"])
        row["cost_usd"] = round(all_cost, 6)           # includes failed attempts
        a = actual.setdefault(spec.model, {"runs": 0, "usd": 0.0})
        a["runs"] += 1
        a["usd"] += all_cost
        rows.append(row)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
    (results / phase / "summary.csv").write_text(buf.getvalue(), encoding="utf-8")
    est_path = results / phase / "estimate.json"
    est = json.loads(est_path.read_text()) if est_path.exists() else {"by_model": {}}
    report = {m: {"runs": v["runs"], "actual_usd": round(v["usd"], 4),
                  "actual_usd_per_run": round(v["usd"] / v["runs"], 4),
                  "estimated_usd_per_run": est["by_model"].get(m, {}).get("usd_per_run")} for m, v in actual.items()}
    report = {"by_model": report, "actual_usd": round(sum(v["usd"] for v in actual.values()), 4),
              "estimated_usd": est.get("estimated_usd"), "correct": sum(r["correct"] for r in rows), "runs": len(rows)}
    (results / phase / "cost_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _xor(data: bytes) -> bytes:
    import numpy as np

    key = np.frombuffer(bm._derive_key(bm.CANARY, len(data)), dtype=np.uint8)
    return (np.frombuffer(data, dtype=np.uint8) ^ key).tobytes()


def _plain_files(phase: str, results: Path) -> list[Path]:
    return sorted((results / phase).glob("*/attempt-*.jsonl")) + sorted((results / phase).glob("judgments.jsonl"))


def pack(phase: str, results: Path = RESULTS) -> int:
    """xz, then the BrowseComp-Plus canary XOR: <file>.xz.enc next to each trace and judgments.jsonl.

    xz rather than gzip: each model call repeats the whole growing prompt, and gzip's 32 KB window
    cannot see that repetition (smoke test: 38.7 MB raw -> 12.1 MB gzip, 1.0 MB xz).
    """
    n = 0
    for p in _plain_files(phase, results):
        enc = p.with_name(p.name + ".xz.enc")
        blob = _xor(lzma.compress(p.read_bytes(), preset=9 | lzma.PRESET_EXTREME))
        if not enc.exists() or enc.read_bytes() != blob:
            enc.write_bytes(blob)
            n += 1
    return n


def unpack(phase: str, results: Path = RESULTS) -> int:
    n = 0
    for enc in sorted((results / phase).rglob("*.jsonl.xz.enc")):
        plain = enc.with_name(enc.name.removesuffix(".xz.enc"))
        if not plain.exists():
            plain.write_bytes(lzma.decompress(_xor(enc.read_bytes())))
            n += 1
    return n


def verify(phase: str, results: Path = RESULTS) -> dict:
    """Judge + summary regenerate byte-identically, and the replay cache serves every search and read."""
    before = {n: (results / phase / n).read_bytes() for n in ("judgments.jsonl", "summary.csv")}
    judge_phase(phase, results)
    summarize_phase(phase, results)
    identical = {n: (results / phase / n).read_bytes() == b for n, b in before.items()}
    cfg = bm.load_config()
    replayed, mismatched = 0, []
    for spec in plan_runs(phase):
        p = done_trace(spec, results)
        for e in ag.read_trace(p) if p else []:
            if e["type"] not in ("search", "read"):
                continue
            r = bm.Retriever(None, cfg, replay=True)
            if e["type"] == "search":
                r.backend_name, r.index_version = e["backend"], e["index_version"]
                ok = r.search(e["query"])["hits"] == e["hits"]
            else:
                start = ag.read_trace(p)[0]["retriever"]
                r.backend_name, r.index_version = start["backend"], start["index_version"]
                ok = r.get_document(e["docid"])["text"] == e["text"]
            replayed += 1
            if not ok:
                mismatched.append((spec.run_id, e["seq"]))
    return {"identical": identical, "replayed": replayed, "replay_mismatches": mismatched}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _confirm(est: dict, given: float | None) -> bool:
    msg = f"About to start {est['runs']} paid runs, estimated ${est['estimated_usd']:.2f}."
    if sys.stdin.isatty() and given is None:
        print(msg)
        return input("Type the estimated cost to proceed: ").strip().lstrip("$") == f"{est['estimated_usd']:.2f}"
    if given is None or round(given, 2) != round(est["estimated_usd"], 2):
        print(msg + f" Re-run with --confirm-cost {est['estimated_usd']:.2f} to proceed.")
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--phase", required=True, choices=sorted(PHASES))
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--confirm", action="store_true")
    r.add_argument("--confirm-cost", type=float, help="non-interactive confirmation: the estimated cost")
    r.add_argument("--only", help="AGENT/MODEL/TASK, each part may be *")
    r.add_argument("--workers", type=int, default=1)
    for name in ("judge", "summarize", "pack", "unpack", "verify"):
        sub.add_parser(name).add_argument("--phase", required=True, choices=sorted(PHASES))
    args = ap.parse_args(argv)
    bm.load_env()
    phase = PHASES[args.phase]

    if args.cmd == "run":
        budget = ag.load_budget()
        specs = select(plan_runs(phase), args.only)
        todo = [s for s in specs if run_state(s, budget["max_attempts"]) == "todo"]
        est = estimate(todo, budget)
        states = {}
        for s in specs:
            st = run_state(s, budget["max_attempts"])
            states[st] = states.get(st, 0) + 1
        print(json.dumps({"phase": phase, "planned": len(specs), "states": states, "estimate_for_todo": est,
                          "spent_so_far": spending(budget), "caps": budget["caps_usd"]}, indent=2))
        if args.dry_run:
            for s in todo:
                print("  ", s.run_id)
            return
        if not args.confirm:
            sys.exit("paid runs need --confirm (and the cost confirmation); nothing was run")
        if phase in ("phase1", "phase2") and git_dirty():
            sys.exit("src/ or configs/ has uncommitted changes: commit and tag before a study phase")
        if not _confirm(est, args.confirm_cost):
            sys.exit("not confirmed; nothing was run")
        (RESULTS / phase).mkdir(parents=True, exist_ok=True)
        est_path = RESULTS / phase / "estimate.json"
        if not est_path.exists():
            est_path.write_text(json.dumps(est, indent=2) + "\n")
        print(json.dumps(Runner(todo).run_all(args.workers), indent=2))
    elif args.cmd == "judge":
        rows = judge_phase(phase)
        print(f"{sum(r['correct'] for r in rows)}/{len(rows)} correct -> results/{phase}/judgments.jsonl")
    elif args.cmd == "summarize":
        print(json.dumps(summarize_phase(phase), indent=2))
    elif args.cmd == "pack":
        print(f"{pack(phase)} files encrypted")
    elif args.cmd == "unpack":
        print(f"{unpack(phase)} files decrypted")
    elif args.cmd == "verify":
        print(json.dumps(verify(phase), indent=2))


if __name__ == "__main__":
    main()
