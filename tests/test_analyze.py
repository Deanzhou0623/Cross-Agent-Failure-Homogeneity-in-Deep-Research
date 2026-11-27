"""Offline tests for spec 04 on a small synthetic set of labeled runs."""

import json

import pytest

from src import agents as ag
from src import analyze as an
from src import benchmark as bm
from tests.test_agents import DOCS, FINAL, TASK, mock
from tests.test_benchmark import CFG, FakeBackend, FakeTokenizer

T = bm.TaskRecord("t1", "Which quiet river flows past the old stone mill near the town?", "Seine",
                  ["100", "101"], ["101"])


def ev(*events):
    out = [{"seq": 0, "type": "run_start"}]
    for e in events:
        out.append({"seq": len(out), **e})
    return out


def test_exposure_levels():
    s = {"type": "search", "query": "q", "hits": [{"docid": "101"}, {"docid": "7"}]}
    r = {"type": "read", "docid": "101", "found": True, "text": ""}
    f = {"type": "final_answer", "cited_docids": ["101"], "exact_answer": "x", "forced": False}
    assert an.exposure(ev({**s, "hits": [{"docid": "7"}]}), T)["level"] == "E0"
    assert an.exposure(ev(s), T)["level"] == "E1"
    e = an.exposure(ev(s, r, f), T)
    assert e["level"] == "E2" and e["read"] == ["101"] and e["cited"] == ["101"]


def test_kappa():
    assert an.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert an.kappa([True, True, False, False], [False, False, True, True]) == -1.0
    assert an.kappa([True, True], [True, True]) is None          # no variation: undefined


def _row(i, task, agent, model, labels):
    return {"run_id": f"r{i}", "task_id": task, "agent": agent, "model": model, "labels": labels,
            "exact_answer": "x" if i < 2 else f"a{i}", "exposure": {"level": "E0"}}


def test_similarity_beats_the_shuffled_baseline():
    # same model, different agents share labels within each task; tasks have disjoint label sets
    rows = []
    for t, lab in (("1", ["retrieval"]), ("2", ["reasoning"]), ("3", ["stopping"]), ("4", ["verification"])):
        for a in ("react", "plan_execute", "evidence_driven"):
            rows.append(_row(len(rows), t, a, "m1", lab))
    sim = {s["pair_kind"]: s for s in an.similarity(rows, n_perm=500)}
    same_model = sim["same model, different agent"]
    assert same_model["pairs"] == 12 and same_model["mean_jaccard"] == 1.0
    assert same_model["baseline"] < 0.5 and same_model["p_value"] < 0.01
    assert same_model["same_final_answer"] == "1/12"
    assert sim["neither"]["pairs"] == 0 and sim["neither"]["mean_jaccard"] is None
    again = {s["pair_kind"]: s for s in an.similarity(rows, n_perm=500)}
    assert again == sim                                          # fixed seed


def test_check_label_rejects_bad_rows():
    runs = {"events": ev({"type": "search", "query": "q", "hits": []})}
    ok = {"labels": ["retrieval"], "primary": "retrieval", "seq": [1], "note": "no evidence in any result"}
    assert an.check_label(ok, runs, T) == []
    assert "seq 9 is not an event of this run" in an.check_label({**ok, "seq": [9]}, runs, T)
    assert "primary is not one of the labels" in an.check_label({**ok, "primary": "stopping"}, runs, T)
    assert "unknown label 'luck'" in an.check_label({**ok, "labels": ["luck"], "primary": "luck"}, runs, T)
    assert "note quotes the answer" in an.check_label({**ok, "note": "said seine late"}, runs, T)
    assert "note quotes the question" in an.check_label({**ok, "note": "the old stone mill near the town"}, runs, T)


def _runs(tmp_path):
    base = {"phase": "phase1", "task_id": "t1", "agent": "react", "model": "m", "correct": False,
            "exact_answer": "x", "forced": False, "exposure": {"level": "E1", "retrieved": ["101"], "read": [], "cited": []},
            "events": ev({"type": "search", "query": "q", "hits": [{"docid": "101"}]},
                         {"type": "final_answer", "exact_answer": "x", "confidence": 5, "cited_docids": [],
                          "explanation": "e", "forced": False})}
    return [{**base, "run_id": "phase1-t1-react-m"}, {**base, "run_id": "phase1-t1-plan_execute-m", "agent": "plan_execute"}]


def test_review_round_trip(tmp_path):
    runs, tasks = _runs(tmp_path), {"t1": T}
    pre = [{"run_id": "phase1-t1-react-m", "labels": ["evidence_selection"], "primary": "evidence_selection",
            "seq": [1], "note": "gold doc in results, never opened"}]
    assert an.add_labels(pre, tmp_path / "prelabels.csv", runs, tasks) == []
    assert an.write_review_files(runs, tasks, review=tmp_path) == 2
    p = tmp_path / "runs" / "phase1-t1-react-m.md"
    text = p.read_text()
    assert "#1 search" in text and "evidence_selection" in text
    p.write_text(text.replace("reviewed: 'no'", "reviewed: 'yes'").replace("reviewed: no", "reviewed: yes"))
    n, probs = an.export_labels(runs, tasks, review=tmp_path)
    assert (n, probs) == (1, [])
    lab = an.read_labels(tmp_path / "labels.csv")
    assert lab["phase1-t1-react-m"]["labels"] == ["evidence_selection"] and lab["phase1-t1-react-m"]["labeler"] == "human"
    assert an.write_review_files(runs, tasks, review=tmp_path) == 1     # the reviewed file is kept


def test_phase_table_outcomes():
    def r(phase, agent, ok, model="m"):
        return {"run_id": f"{phase}-1-{agent}-{model}", "phase": phase, "task_id": "1", "agent": agent, "model": model,
                "correct": ok}
    runs = [r("phase1", "react", False), r("phase2", "react", False, "n"),
            r("phase1", "plan_execute", False), r("phase2", "plan_execute", True, "n")]
    labels = {"phase1-1-react-m": {"primary": "retrieval"}, "phase2-1-react-n": {"primary": "retrieval"},
              "phase1-1-plan_execute-m": {"primary": "stopping"}}
    out = {x["agent"]: x["outcome"] for x in an.phase_table(runs, labels)}
    assert out == {"react": "persists", "plan_execute": "disappears"}


def test_context_docs_go_into_the_first_prompt_and_trace(retriever, tmp_path):
    b = ag.load_budget()
    path = tmp_path / "ctx.jsonl"
    ag.run_agent("react", TASK, mock([{"text": FINAL}]), retriever, path, run_id="t", phase="debug", budget=b,
                 context_docids=["101", "nope"])
    events = ag.read_trace(path)
    assert ag.validate_trace(events) == []
    ctx = [e for e in events if e["type"] == "context"]
    assert len(ctx) == 1 and ctx[0]["docids"] == ["101"]          # unknown docids are skipped
    first = json.dumps(next(e for e in events if e["type"] == "model_call")["prompt"])
    assert ag.CONTEXT_INTRO in first and "doc 101" in first
    assert events[-1]["read_docids"] == []                       # injected docs do not count as reads


@pytest.fixture
def retriever(tmp_path):
    return bm.Retriever(FakeBackend(DOCS), CFG, cache_dir=tmp_path / "cache", tokenizer=FakeTokenizer())
