"""Offline tests for spec 03: mock model + fake retriever, no network or keys."""

import json

import pytest

from src import agents as ag
from src import benchmark as bm
from src import runner as rn
from tests.test_agents import FINAL, mock
from tests.test_benchmark import CFG, FakeBackend, FakeTokenizer

DOCS = {str(100 + i): f"doc {100 + i} paris seine " * 40 for i in range(10)}
TASKS = {q: bm.TaskRecord(q, f"question {q}?", "Seine", ["100"], ["101"]) for q in ("t1", "t2")}
SPECS = [rn.RunSpec("phase1", t, a, "mockm") for t in TASKS for a in ("react", "evidence_driven")]


def mcfg(script):
    return {**mock(script), "phase": "phase1"}


def make_runner(tmp_path, script, *, budget=None, specs=SPECS, cost_per_call=0.0):
    b = {**ag.load_budget(), "prior_spend_usd": {"total": 0.0, "google_credits": 0.0}, **(budget or {})}
    retr = bm.Retriever(FakeBackend(DOCS), CFG, cache_dir=tmp_path / "cache", tokenizer=FakeTokenizer())

    def load_model(key):
        m = mcfg(script)
        if cost_per_call:   # mock usage is 100 in + 20 out per call
            m["price"] = {"input": cost_per_call * 1e6 / 120, "output": cost_per_call * 1e6 / 120}
        return m

    return rn.Runner(specs, budget=b, results=tmp_path / "results", tasks=TASKS, load_model=load_model,
                     make_retriever=lambda: retr, judge_cache=None, log=lambda *_: None)


def test_dry_run_plans_90_and_45():
    p1, p2, smoke = rn.plan_runs("phase1"), rn.plan_runs("phase2"), rn.plan_runs("smoke")
    assert len(p1) == 90 and len(p2) == 45 and len(smoke) == 18
    assert len({s.run_id for s in p1 + p2 + smoke}) == 153
    study = set(rn._yaml("tasks.yaml")["phase1_task_ids"])
    assert not {s.task_id for s in smoke} & study
    assert {s.task_id for s in p2} <= {s.task_id for s in p1}
    assert rn.select(p1, "react/haiku/*") == [s for s in p1 if s.agent == "react" and s.model == "haiku"]


def test_resume_skips_completed_runs(tmp_path):
    runner = make_runner(tmp_path, [{"text": FINAL}])
    assert runner.run_all()["states"] == {"done": 4}
    again = make_runner(tmp_path, [{"text": FINAL}])
    assert again.run_all()["states"] == {"done": 4}
    for s in SPECS:
        assert len(rn.attempts(s, tmp_path / "results")) == 1     # nothing re-run


def test_errors_retry_then_missing(tmp_path):
    bad = {"text": "\n", "stop_reason": "max_tokens"}               # persistent degenerate output -> error
    runner = make_runner(tmp_path, [bad], specs=SPECS[:1])
    assert runner.run_all()["states"] == {"missing": 1}
    assert len(rn.attempts(SPECS[0], tmp_path / "results")) == 4      # 1 try + 3 retries
    assert make_runner(tmp_path, [{"text": FINAL}], specs=SPECS[:1]).run_all()["states"] == {"missing": 1}
    # --retry-missing: a fresh set of attempts; the earlier ones stay on disk and numbering continues
    retry = make_runner(tmp_path, [{"text": FINAL}], specs=SPECS[:1])
    retry.retry_missing = True
    assert retry.run_all()["states"] == {"done": 1}
    assert len(rn.attempts(SPECS[0], tmp_path / "results")) == 5
    assert rn.done_trace(SPECS[0], tmp_path / "results").name == "attempt-5.jsonl"


def test_model_mismatch_stops_the_runner(tmp_path):
    runner = make_runner(tmp_path, [{"text": FINAL, "served_model": "other"}])
    out = runner.run_all()
    assert "model mismatch" in out["stop_reason"] and out["states"].get("stopped", 0) == 4
    assert len(rn.attempts(SPECS[0], tmp_path / "results")) == 1


def test_phase_cap_stops_cleanly(tmp_path):
    # each run costs $0.20 (one call); a run needs $0.40 of room (the per-run cap); phase cap $0.70:
    # run 1 (0 + 0.4), run 2 (0.2 + 0.4) fit; run 3 (0.4 + 0.4 > 0.7) does not
    b = {"caps_usd": {**ag.load_budget()["caps_usd"], "phase1": 0.7}}
    runner = make_runner(tmp_path, [{"text": FINAL}], budget=b, cost_per_call=0.20)
    out = runner.run_all()
    assert out["states"] == {"done": 2, "stopped": 2} and "phase1 cap" in out["stop_reason"]
    assert out["spent"]["by_phase"]["phase1"] == pytest.approx(0.40)


def test_total_and_gemini_caps(tmp_path):
    b = ag.load_budget()
    spent = {"total": 29.8, "google_credits": 0.0, "by_phase": {}}
    m = {"phase": "phase1"}
    assert "total cap" in rn.cap_problem(SPECS[0], spent, b, m)
    spent = {"total": 0.0, "google_credits": 14.8, "by_phase": {}}
    assert "google_credits cap" in rn.cap_problem(SPECS[0], spent, b, {**m, "billing": "google_credits"})
    assert rn.cap_problem(SPECS[0], spent, b, m) is None              # non-Gemini runs are unaffected
    # in-flight room counts each running run's own cap, not this run's cap times the worker count
    smoke = rn.RunSpec("smoke", "t1", "react", "sonnet")
    spent = {"total": 0.5, "google_credits": 0.4, "by_phase": {"smoke": 0.47}}
    assert rn.cap_problem(smoke, spent, b, {"phase": "phase2"}, inflight_room=0.8) is None      # 0.47+0.8+1.5 <= 4
    assert "smoke cap" in rn.cap_problem(smoke, spent, b, {"phase": "phase2"}, inflight_room=3.0)


class FakeJudge:
    def judge(self, task, response):
        return {"correct": "Seine" in response, "extracted_final_answer": "x", "cost_usd": 0.0}


def test_judge_summary_pack_are_reproducible(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, [{"tool_calls": [["search", {"query": "paris"}]]}, {"text": FINAL}])
    runner.run_all()
    res = tmp_path / "results"
    monkeypatch.setattr(rn, "plan_runs", lambda phase: SPECS)
    monkeypatch.setattr(rn, "_yaml", lambda n: {"tasks": [{"query_id": q, "stratum": "hard"} for q in TASKS]})
    rows = rn.judge_phase("phase1", res, judge=FakeJudge(), tasks=TASKS)
    assert len(rows) == 4 and all(r["correct"] for r in rows)
    rep = rn.summarize_phase("phase1", res)
    assert rep["runs"] == 4 and rep["correct"] == 4
    first = {n: (res / "phase1" / n).read_bytes() for n in ("judgments.jsonl", "summary.csv")}
    rn.judge_phase("phase1", res, judge=FakeJudge(), tasks=TASKS)
    rn.summarize_phase("phase1", res)
    assert first == {n: (res / "phase1" / n).read_bytes() for n in first}
    header = (res / "phase1" / "summary.csv").read_text().splitlines()[0]
    assert header.split(",") == rn.SUMMARY_FIELDS and "Seine" not in (res / "phase1" / "summary.csv").read_text()

    n = rn.pack("phase1", res)
    assert n == 5 and rn.pack("phase1", res) == 0                      # 4 traces + judgments; idempotent
    enc = sorted((res / "phase1").rglob("*.xz.enc"))
    plain = {e: e.with_name(e.name.removesuffix(".xz.enc")).read_bytes() for e in enc}
    assert all(b"Seine" not in e.read_bytes() and b"paris" not in e.read_bytes() for e in enc)
    for e in enc:
        e.with_name(e.name.removesuffix(".xz.enc")).unlink()
    assert rn.unpack("phase1", res) == 5
    assert all(e.with_name(e.name.removesuffix(".xz.enc")).read_bytes() == b for e, b in plain.items())


def test_replay_cache_serves_logged_searches(tmp_path):
    runner = make_runner(tmp_path, [{"tool_calls": [["search", {"query": "paris"}]]},
                                    {"tool_calls": [["get_document", {"docid": "100"}]]}, {"text": FINAL}],
                         specs=SPECS[:1])
    runner.run_all()
    trace = rn.done_trace(SPECS[0], tmp_path / "results")
    events = ag.read_trace(trace)
    replay = bm.Retriever(None, CFG, replay=True, cache_dir=tmp_path / "cache", tokenizer=FakeTokenizer())
    s = next(e for e in events if e["type"] == "search")
    replay.backend_name, replay.index_version = s["backend"], s["index_version"]
    assert replay.search(s["query"])["hits"] == s["hits"]
    r = next(e for e in events if e["type"] == "read")
    assert replay.get_document(r["docid"])["text"] == r["text"]


def test_run_start_records_config_hashes_and_billing(tmp_path):
    make_runner(tmp_path, [{"text": FINAL}], specs=SPECS[:1]).run_all()
    start = ag.read_trace(rn.done_trace(SPECS[0], tmp_path / "results"))[0]
    assert set(start["meta"]["config_sha256"]) == set(rn.CONFIG_FILES)
    assert start["meta"]["attempt"] == 1 and start["meta"]["billing"] == "default"
    assert start["meta"]["run_spec"]["agent"] == "react"


def test_confirmation_requires_the_estimate(capsys):
    est = {"runs": 90, "estimated_usd": 5.4}
    assert rn._confirm(est, None) is False
    assert rn._confirm(est, 5.0) is False
    assert rn._confirm(est, 5.40) is True


def test_runner_waits_for_in_flight_runs_instead_of_stopping(tmp_path):
    # phase cap $0.75, per-run cap $0.40, runs cost $0.05: two workers cannot both reserve $0.40 at once,
    # but spending stays low, so the runner must wait for in-flight runs, not stop, and finish all 4
    b = {"caps_usd": {**ag.load_budget()["caps_usd"], "phase1": 0.75}}
    runner = make_runner(tmp_path, [{"text": FINAL}], budget=b, cost_per_call=0.05)
    out = runner.run_all(workers=3)
    assert out["states"] == {"done": 4} and out["stop_reason"] is None


def test_evidence_phase_puts_gold_docs_into_the_prompt(tmp_path, monkeypatch):
    spec = rn.RunSpec("evidence", "t1", "react", "mockm")
    b = {"caps_usd": {**ag.load_budget()["caps_usd"]}}
    runner = make_runner(tmp_path, [{"text": FINAL}], budget=b, specs=[spec])
    assert runner.run_all()["states"] == {"done": 1}
    events = ag.read_trace(rn.done_trace(spec, tmp_path / "results"))
    assert events[0]["meta"]["context_docids"] == ["101"]          # the task's gold doc
    assert [e["docids"] for e in events if e["type"] == "context"] == [["101"]]
