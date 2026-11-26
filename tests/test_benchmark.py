"""Offline tests for spec 01. No network, no API keys, no benchmark data needed."""

import json
import subprocess
from pathlib import Path

import pytest

from src import benchmark as bm
from src import models

CFG = bm.load_config()


class FakeTokenizer:
    """Whitespace tokenizer standing in for Qwen3-0.6B."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self._Enc(text.split())

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class FakeBackend(bm.Backend):
    name = "fake"
    index_version = "v1"

    def __init__(self, docs: dict[str, str]):
        self.docs = docs
        self.calls = 0

    def search(self, query, k):
        self.calls += 1
        words = set(query.lower().split())
        ranked = sorted(self.docs, key=lambda d: (-len(words & set(self.docs[d].lower().split())), d))
        return [{"docid": d, "score": 1.0 / (i + 1), "text": self.docs[d]} for i, d in enumerate(ranked[:k])]

    def get_document(self, docid):
        self.calls += 1
        return self.docs.get(docid)


DOCS = {f"d{i}": " ".join([f"word{i}"] * 600 + ["paris", "seine"] * (i % 2)) for i in range(20)}


@pytest.fixture
def retriever(tmp_path):
    return bm.Retriever(FakeBackend(DOCS), CFG, cache_dir=tmp_path, tokenizer=FakeTokenizer())


def test_crypto_roundtrip():
    text = "Ünïcode question?"
    assert bm.decrypt(bm.encrypt(text)) == text


def test_search_returns_k_hits_with_truncated_snippets(retriever):
    resp = retriever.search("paris seine")
    assert len(resp["hits"]) == CFG["retrieval"]["k"] == 5
    assert [h["rank"] for h in resp["hits"]] == [1, 2, 3, 4, 5]
    assert all(h["snippet_tokens"] <= CFG["retrieval"]["snippet_max_tokens"] for h in resp["hits"])
    assert resp["cache_hit"] is False


def test_get_document_truncates(retriever):
    doc = retriever.get_document("d1")
    assert doc["found"] and doc["original_tokens"] == 602
    short_cfg = {**CFG, "retrieval": {**CFG["retrieval"], "document_max_tokens": 100}}
    r = bm.Retriever(FakeBackend(DOCS), short_cfg, cache_dir=retriever.cache_dir / "short", tokenizer=FakeTokenizer())
    cut = r.get_document("d1")
    assert cut["truncated"] and cut["returned_tokens"] == 100 and len(cut["text"].split()) == 100


def test_missing_document(retriever):
    assert retriever.get_document("nope")["found"] is False


def test_cache_replay_is_identical(retriever, tmp_path):
    first = retriever.search("paris seine")
    calls = retriever.backend.calls
    again = retriever.search("paris seine")
    assert retriever.backend.calls == calls and again["cache_hit"] is True
    replay = bm.Retriever(None, CFG, replay=True, cache_dir=tmp_path, tokenizer=FakeTokenizer())
    replay.backend_name, replay.index_version = "fake", "v1"
    strip = lambda r: {k: v for k, v in r.items() if k != "cache_hit"}
    assert strip(replay.search("paris seine")) == strip(first)
    with pytest.raises(KeyError):
        replay.search("never searched")


def test_judge_parser():
    ok = bm.parse_judge_response("extracted_final_answer: Seine\nreasoning: matches\ncorrect: yes\nconfidence: 90%")
    assert ok["correct"] is True and ok["extracted_final_answer"] == "Seine" and ok["confidence"] == 90.0
    bold = bm.parse_judge_response("**extracted_final_answer:** Loire\n**reasoning:** no\n**correct:** no")
    assert bold["correct"] is False
    bad = bm.parse_judge_response("I cannot tell.")
    assert bad["parse_error"] is True and bad["correct"] is None
    assert bm.parse_judge_response("")["parse_error"] is True


def test_grader_template_matches_official_shape():
    assert "{question}" in bm.GRADER_TEMPLATE and "{correct_answer}" in bm.GRADER_TEMPLATE
    assert "0|\\%| and 100|\\%|" in bm.GRADER_TEMPLATE


def _toy_tasks_and_correctness():
    tasks, corr = {}, {}
    pattern = {"hard": [1, 0, 0, 0], "moderate": [1, 1, 0, 0], "none": [0, 0, 0, 0], "easy": [1, 1, 1, 1]}
    i = 0
    for kind, n in (("hard", 15), ("moderate", 8), ("none", 5), ("easy", 5)):
        for _ in range(n):
            qid = str(i)
            tasks[qid] = bm.TaskRecord(qid, f"q{i}", f"a{i}", ["e"], ["g"] if i != 3 else [])
            corr[qid] = {f"run{j}": bool(v) for j, v in enumerate(pattern[kind])}
            i += 1
    return tasks, corr


def test_selection_is_deterministic_and_stratified():
    tasks, corr = _toy_tasks_and_correctness()
    a = bm.build_candidates(corr, tasks, seed=20261125)
    b = bm.build_candidates(corr, tasks, seed=20261125)
    assert a == b
    assert "3" not in a["hard"]  # no gold document -> excluded
    assert len(a["hard"]) == 14 and len(a["moderate"]) == 8
    screening = [{"query_id": q, "stratum": s, "decision": "accept"} for s in ("hard", "moderate") for q in a[s]]
    screening[0]["decision"] = "reject"
    screening[0]["reason"] = "ambiguous"
    scfg = CFG["selection"]
    r1 = bm.finalize_selection(screening, corr, tasks, scfg)
    r2 = bm.finalize_selection(screening, corr, tasks, scfg)
    assert r1 == r2
    assert len(r1["phase1_task_ids"]) == 10 and len(r1["phase2_task_ids"]) == 5
    strata = {t["query_id"]: t["stratum"] for t in r1["tasks"]}
    assert sum(s == "hard" for s in strata.values()) == 7
    assert [strata[q] for q in r1["phase2_task_ids"]].count("hard") == 3
    assert set(r1["phase2_task_ids"]) <= set(r1["phase1_task_ids"])
    assert r1["excluded"][0]["query_id"] == a["hard"][0]


def test_finalize_requires_review():
    tasks, corr = _toy_tasks_and_correctness()
    a = bm.build_candidates(corr, tasks, seed=1)
    screening = [{"query_id": q, "stratum": s, "decision": "pending"} for s in ("hard", "moderate") for q in a[s]]
    with pytest.raises(ValueError, match="pending"):
        bm.finalize_selection(screening, corr, tasks, CFG["selection"])


def test_task_view_has_no_answer_or_qrels():
    t = bm.TaskRecord("1", "question?", "secret answer", ["e1"], ["g1"])
    view = t.view()
    assert set(view) == {"query_id", "question"}
    assert "secret answer" not in json.dumps(view)


def test_published_run_parsing(tmp_path):
    p = tmp_path / "run.jsonl"
    rows = [{"query_id": "1", "status": "completed", "result": [{"type": "output_text", "output": "Exact Answer: x"}]},
            {"query_id": "2", "status": "max_tokens", "result": [{"type": "output_text", "output": "partial"}]},
            {"query_id": "3", "status": "completed", "result": [{"type": "tool_call", "output": "..."}]}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert bm.published_run_responses(p) == {"1": "Exact Answer: x", "2": None, "3": None}


def test_cost_computation():
    price = {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50,
             "long_context_threshold": 100000, "long_input": 0.50, "long_output": 2.50}
    small = models.compute_cost(price, {"input": 1000, "cached_input": 9000, "cache_write": 0, "output": 1000})
    assert small == pytest.approx((1000 * 0.10 + 9000 * 0.01 + 1000 * 0.50) / 1e6)
    big = models.compute_cost(price, {"input": 200000, "cached_input": 0, "cache_write": 0, "output": 0})
    assert big == pytest.approx(200000 * 0.50 / 1e6)


def test_mock_conversation_tool_loop():
    mcfg = {"provider": "mock", "model_id": "mock-1", "script": [
        {"tool_calls": [["search", {"query": "paris"}]]}, {"text": "Exact Answer: Seine"}]}
    conv = models.open_conversation(mcfg, system="s")
    conv.add_user("q")
    r1 = conv.step([models.CHECK_TOOL])
    assert r1.tool_calls[0].name == "search" and r1.cost_usd == 0.0
    conv.add_tool_results([(r1.tool_calls[0].id, "search", "result")])
    assert "Seine" in conv.step([models.CHECK_TOOL]).text


def test_all_six_models_configured():
    keys = models.all_model_keys()
    assert len(keys) == 6
    for k in keys:
        m = models.load_model(k)
        assert m["provider"] in ("anthropic", "openai", "gemini") and m["model_id"]
        expected = "high" if m["phase"] == "phase1" else "medium"
        assert expected in m["reasoning"].values()


def test_no_benchmark_text_in_tracked_files():
    queries = bm.DATA / "queries.jsonl"
    if not queries.exists():
        pytest.skip("benchmark data not downloaded")
    answers = {json.loads(l)["answer"].strip() for l in queries.read_text(encoding="utf-8").splitlines()}
    answers = {a for a in answers if len(a) >= 6}
    tracked = subprocess.run(["git", "ls-files"], cwd=bm.ROOT, capture_output=True, text=True).stdout.split()
    for rel in tracked:
        path = bm.ROOT / rel
        if path.suffix in {".py", ".md", ".yaml", ".yml", ".txt", ".csv", ".json", ".toml"} and path.is_file():
            body = path.read_text(encoding="utf-8", errors="ignore")
            leaked = [a for a in answers if a in body]
            assert not leaked, f"{rel} contains benchmark answer text"
