"""Offline tests for spec 02: mock model + fake retriever, no network or keys."""

import json

import pytest

from src import agents as ag
from src import benchmark as bm
from tests.test_benchmark import CFG, FakeBackend, FakeTokenizer

DOCS = {str(100 + i): f"doc {100 + i} " + ("paris seine river " if i < 3 else f"filler{i} ") * 50 for i in range(12)}
SECRET = "SECRET-ANSWER-XYZ"
TASK = {"query_id": "q1", "question": "Which river flows through Paris?"}
FINAL = "Explanation: It is the Seine [100].\nExact Answer: Seine\nConfidence: 90%"


@pytest.fixture
def retriever(tmp_path):
    return bm.Retriever(FakeBackend(DOCS), CFG, cache_dir=tmp_path / "cache", tokenizer=FakeTokenizer())


def mock(script):
    return {"provider": "mock", "model_id": "mock-1", "key": "mock", "reasoning": {"effort": "high"},
            "price": {"input": 0, "output": 0}, "script": script}


def run(agent, script, retriever, tmp_path, **budget):
    b = {**ag.load_budget(), **budget}
    path = tmp_path / f"{agent}.jsonl"
    summary = ag.run_agent(agent, TASK, mock(script), retriever, path, run_id="t", phase="debug", budget=b)
    events = ag.read_trace(path)
    assert ag.validate_trace(events) == [], ag.validate_trace(events)
    return summary, events


def of(events, type_):
    return [e for e in events if e["type"] == type_]


SCRIPTS = {
    "react": [{"tool_calls": [["search", {"query": "paris seine"}]]},
              {"tool_calls": [["get_document", {"docid": "100"}]]},
              {"text": FINAL}],
    "plan_execute": [{"text": '{"plan": [{"step": 1, "goal": "find the river"}, {"step": 2, "goal": "confirm"}]}'},
                     {"tool_calls": [["search", {"query": "paris seine"}]]},
                     {"text": '{"step": 1, "status": "done", "finding": "Seine"}'},
                     {"tool_calls": [["get_document", {"docid": "100"}]]},
                     {"text": FINAL}],
    "evidence_driven": [{"text": '{"facts": [], "gaps": ["river name"]}', "tool_calls": [["search", {"query": "paris seine"}]]},
                        {"text": '{"facts": [{"fact": "Seine", "docid": "100"}], "gaps": ["confirm"]}',
                         "tool_calls": [["get_document", {"docid": "100"}]]},
                        {"text": '{"facts": [{"fact": "Seine", "docid": "100"}], "gaps": []}\n' + FINAL}],
}


@pytest.mark.parametrize("agent", list(ag.AGENTS))
def test_each_agent_completes_a_mock_task(agent, retriever, tmp_path):
    summary, events = run(agent, SCRIPTS[agent], retriever, tmp_path)
    assert summary["status"] == "completed"
    fa = of(events, "final_answer")[0]
    assert fa["exact_answer"] == "Seine" and fa["confidence"] == 90.0 and fa["cited_docids"] == ["100"]
    end = events[-1]
    assert "100" in end["retrieved_docids"] and end["read_docids"] == ["100"]
    assert end["totals"]["search"] == 1 and end["totals"]["get_document"] == 1
    s = of(events, "search")[0]
    assert len(s["hits"]) == 5 and all({"rank", "docid", "score", "snippet"} <= set(h) for h in s["hits"])
    assert of(events, "read")[0]["text"]
    calls = of(events, "model_call")
    assert all(c["served_model"] == c["requested_model"] == "mock-1" for c in calls)
    assert calls[-1]["prompt"][-1]["role"] in ("tool", "user")     # full prompt as sent, every call
    if agent == "plan_execute":
        assert [p["version"] for p in of(events, "plan") if "steps" in p] == [1]   # no replan needed
    if agent == "evidence_driven":
        assert of(events, "ledger")[-1]["gaps"] == []


def test_tool_definitions_identical_across_agents(retriever, tmp_path):
    starts = [run(a, SCRIPTS[a], retriever, tmp_path / a)[1][0] for a in ag.AGENTS]
    assert len({json.dumps(s["tools"], sort_keys=True) for s in starts}) == 1
    assert len({s["query_template"] for s in starts}) == 1
    assert len({s["system_prompt"] for s in starts}) == 3        # only the strategy differs


def _endless_search():
    return [{"tool_calls": [["search", {"query": f"q{i}"}]]} for i in range(60)] + [{"text": FINAL}]


@pytest.mark.parametrize("limit,budget", [
    ("model_calls", {"max_model_calls": 3}),
    ("tool_calls", {"max_tool_calls": 2}),
    ("cost", {"cost_cap_usd": {"debug": 0.0}}),
    ("wall_time", {"max_wall_minutes": 0}),
])
def test_each_limit_triggers_one_final_request(limit, budget, retriever, tmp_path):
    summary, events = run("react", _endless_search(), retriever, tmp_path, **budget)
    assert summary["status"] == "budget_exhausted"
    assert [b["limit"] for b in of(events, "budget")] == [limit]
    finals = [c for c in of(events, "model_call") if c["purpose"] == "final"]
    assert len(finals) == 1 and finals[0]["request"]["tool_choice"] == "none"
    assert finals[0]["prompt"][-1]["content"] == ag.FINAL_REQUEST
    assert of(events, "final_answer")[0]["forced"] is True
    if limit == "tool_calls":
        assert summary["tool_calls"] == 2


def test_get_document_limit_refuses_reads_but_run_continues(retriever, tmp_path):
    script = [{"tool_calls": [["get_document", {"docid": "100"}]]},
              {"tool_calls": [["get_document", {"docid": "101"}]]},
              {"tool_calls": [["search", {"query": "paris"}]]}, {"text": FINAL}]
    summary, events = run("react", script, retriever, tmp_path, max_get_document=1)
    assert summary["status"] == "completed" and summary["get_document"] == 1 and summary["search"] == 1
    assert of(events, "budget")[0]["limit"] == "get_document"


def test_agents_never_see_answers_or_qrels(retriever, tmp_path):
    record = bm.TaskRecord("q1", TASK["question"], SECRET, ["100"], ["101"])
    path = tmp_path / "leak.jsonl"
    ag.run_agent("react", record.view(), mock(SCRIPTS["react"]), retriever, path, run_id="t", phase="debug")
    body = path.read_text()
    assert SECRET not in body and "evidence_docids" not in body and "gold" not in body
    with pytest.raises(ValueError):
        ag.run_agent("react", {**record.view(), "answer": SECRET}, mock([]), retriever, tmp_path / "x.jsonl",
                     run_id="t", phase="debug")


def test_truncated_trace_is_readable(retriever, tmp_path):
    _, events = run("react", SCRIPTS["react"], retriever, tmp_path)
    path = tmp_path / "react.jsonl"
    raw = path.read_text()
    cut = tmp_path / "cut.jsonl"
    cut.write_text(raw[: raw.rindex('{"seq"') + 25])          # crash mid-way through the last event
    partial = ag.read_trace(cut)
    assert len(partial) == len(events) - 1 and partial[0]["type"] == "run_start"


def test_replan_on_exhausted_plan(retriever, tmp_path):
    script = [{"text": '{"plan": [{"step": 1, "goal": "find it"}]}'},
              {"tool_calls": [["search", {"query": "paris"}]]},
              {"text": '{"step": 1, "status": "done", "finding": "not yet"}'},
              {"text": '{"plan": [{"step": 1, "goal": "read the top document"}]}'},
              {"text": FINAL}]
    summary, events = run("plan_execute", script, retriever, tmp_path)
    plans = [p for p in of(events, "plan") if "steps" in p]
    assert [p["version"] for p in plans] == [1, 2] and plans[0]["reason"] is None
    assert "done or blocked" in plans[1]["reason"] and summary["status"] == "completed"


def test_replan_on_blocked_step_reported_and_automatic(retriever, tmp_path):
    reported = [{"text": '{"plan": [{"step": 1, "goal": "a"}, {"step": 2, "goal": "b"}]}'},
                {"text": '{"step": 1, "status": "blocked", "finding": "nothing"}'},
                {"text": '{"plan": [{"step": 1, "goal": "c"}]}'}, {"text": FINAL}]
    _, events = run("plan_execute", reported, retriever, tmp_path / "r")
    plans = [p for p in of(events, "plan") if "steps" in p]
    assert len(plans) == 2 and "reported blocked" in plans[1]["reason"]

    same = {"tool_calls": [["search", {"query": "paris seine"}]]}      # 1st call is productive, then 3 are not
    auto = [{"text": '{"plan": [{"step": 1, "goal": "a"}]}'}, same, same, same, same,
            {"text": '{"plan": [{"step": 1, "goal": "b"}]}'}, {"text": FINAL}]
    _, events = run("plan_execute", auto, retriever, tmp_path / "a")
    plans = [p for p in of(events, "plan") if "steps" in p]
    assert len(plans) == 2 and "3 tool calls in a row" in plans[1]["reason"]
    assert any(p.get("update", {}).get("by") == "auto" for p in of(events, "plan"))
    replan_call = [c for c in of(events, "model_call") if c["purpose"] == "plan"][1]
    assert replan_call["prompt"][-2]["role"] == "tool"          # tool results flushed with the replan request


def test_degenerate_output_is_retried_and_rolled_back(retriever, tmp_path):
    bad = {"text": "\n" * 50, "stop_reason": "max_tokens", "output_tokens": 16000}
    summary, events = run("react", [bad, {"text": FINAL}], retriever, tmp_path)
    assert summary["status"] == "completed"
    err = of(events, "error")[0]
    assert err["kind"] == "degenerate_output" and err["retry"] == 1
    calls = of(events, "model_call")
    assert calls[0]["prompt"] == calls[1]["prompt"]          # the degenerate reply was rolled back


def test_persistent_degenerate_output_ends_run_as_error(retriever, tmp_path):
    bad = {"text": "x\n" * 30, "stop_reason": "MAX_TOKENS"}
    summary, events = run("react", [bad], retriever, tmp_path)
    assert summary["status"] == "error" and events[-1]["error"]["kind"] == "degenerate_output"
    assert len(of(events, "error")) == 3


def test_model_mismatch_stops_the_run(retriever, tmp_path):
    summary, events = run("react", [{"text": FINAL, "served_model": "mock-2"}], retriever, tmp_path)
    assert summary["status"] == "error" and of(events, "error")[0]["kind"] == "model_mismatch"
    assert not of(events, "final_answer")


def test_tool_errors_are_logged_and_returned(retriever, tmp_path):
    script = [{"tool_calls": [["search", {}], ["browse", {"url": "x"}]]}, {"text": FINAL}]
    summary, events = run("react", script, retriever, tmp_path)
    assert [e["kind"] for e in of(events, "error")] == ["tool", "tool"] and summary["status"] == "completed"


def test_export_official_format(retriever, tmp_path):
    _, events = run("react", SCRIPTS["react"], retriever, tmp_path)
    out = ag.export_official(events)
    assert out["query_id"] == "q1" and out["status"] == "completed"
    assert out["tool_call_counts"] == {"search": 1, "get_document": 1}
    assert [r["type"] for r in out["result"]] == ["tool_call", "tool_call", "output_text"]
    assert json.loads(out["result"][0]["output"])[0]["docid"] in out["retrieved_docids"]
    p = tmp_path / "export.jsonl"  # the reader used for the published runs also reads our export
    p.write_text(json.dumps(out) + "\n")
    assert bm.published_run_responses(p) == {"q1": FINAL}


def test_api_keys_are_redacted(retriever, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    task = {"query_id": "q1", "question": "Is sk-test-0123456789abcdef a key?"}
    path = tmp_path / "keys.jsonl"
    ag.run_agent("react", task, mock([{"text": FINAL}]), retriever, path, run_id="t", phase="debug")
    assert "sk-test-0123456789abcdef" not in path.read_text() and "[REDACTED]" in path.read_text()


def test_parse_final_and_json():
    p = ag.parse_final("**Explanation:** A [1, 2] and [3].\n**Exact Answer:** Loire\n**Confidence:** 75%")
    assert p["exact_answer"] == "Loire" and p["cited_docids"] == ["1", "2", "3"] and p["confidence"] == 75.0
    assert ag.parse_final("no format")["exact_answer"] is None
    assert ag.extract_json('Ledger:\n```json\n{"gaps": []}\n```\nthen') == {"gaps": []}
    assert ag.extract_json("nothing here") is None
