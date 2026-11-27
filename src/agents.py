"""Three search agents sharing one tool layer, one budget, and one trace logger (spec 02).

    summary = run_agent("react", task.view(), load_model("haiku"), retriever,
                        Path("results/debug/x.jsonl"), run_id="x", phase="debug")

Every event is appended to a JSONL trace and flushed immediately. The trace is the
single source of truth: retrieved/read sets and the official-format export are
derived from it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import yaml

from src import models
from src.benchmark import ROOT

# --------------------------------------------------------------------------- #
# Shared prompt and tools (identical for every agent)
# --------------------------------------------------------------------------- #

# Official BrowseComp-Plus QUERY_TEMPLATE (texttron/BrowseComp-Plus, MIT License).
QUERY_TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and get_document tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and get_document tools multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

SEARCH_TOOL = {
    "name": "search",
    "description": "Performs a web search: supply a string 'query'; the tool retrieves the top 5 results for the "
                   "query, returning their docid, score, and snippet. The snippet contains the document's contents "
                   "(may be truncated based on token limits).",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "The query string for the search."}},
                   "required": ["query"]},
}
GET_DOCUMENT_TOOL = {
    "name": "get_document",
    "description": "Retrieve the full text of a document by its docid (may be truncated based on token limits).",
    "parameters": {"type": "object",
                   "properties": {"docid": {"type": "string", "description": "The docid of the document."}},
                   "required": ["docid"]},
}
TOOLS = [SEARCH_TOOL, GET_DOCUMENT_TOOL]

FINAL_REQUEST = ("The research budget is used up. Do not call any more tools. Give your final answer now, "
                 "in the required format (Explanation / Exact Answer / Confidence).")

# Only these strategy instructions differ between agents.
STRATEGIES = {
    "react": (
        "Strategy (ReAct): work step by step. In each turn, think about what you know and what is still "
        "missing, then either call a tool or, once you can answer, give the final answer in the required format."
    ),
    "plan_execute": (
        "Strategy (Plan-and-Execute): first write a plan of sub-questions, then carry it out one step at a time "
        "with the tools.\n"
        'When asked for a plan, reply only with JSON: {"plan": [{"step": 1, "goal": "..."}, ...]}.\n'
        "While executing a step, use the tools. When the step is finished, reply without a tool call, with JSON: "
        '{"step": N, "status": "done" or "blocked", "finding": "..."}.\n'
        "As soon as you can answer the question, give the final answer in the required format instead."
    ),
    "evidence_driven": (
        "Strategy (Evidence-Driven): keep an evidence ledger. In every reply, first write the updated ledger as "
        'JSON: {"facts": [{"fact": "...", "docid": "..."}], "gaps": ["..."]}, giving each fact with the docid '
        "that supports it, and the open gaps, most important first. Then call a tool to close the most important "
        "gap. When no gaps remain, give the final answer in the required format instead of calling a tool."
    ),
}
PLAN_FIRST = "Before using any tools, write your plan as JSON only."
EXECUTE_STEP = "Execute step {step}: {goal}"
REPLAN = "Replan: {reason}.\nFindings so far:\n{findings}\nWrite a revised plan as JSON only, covering what is still needed."
PLAN_NUDGE = "Reply with the step status JSON, call a tool, or give the final answer in the required format."
LEDGER_DONE = "No gaps remain in your ledger. Give the final answer in the required format."
LEDGER_NUDGE = "Call a tool for the most important gap, or give the final answer in the required format."

KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
            "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY")


def load_budget(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "configs" / "budget.yaml").read_text())


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #

class Trace:
    """Append-only JSONL writer. Each event is flushed as it happens; API keys are redacted."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("x", encoding="utf-8")   # never overwrite a raw trace
        self.seq = 0
        self._secrets = [v for k in KEY_VARS if len(v := os.environ.get(k, "")) >= 8]

    def log(self, type_: str, **fields) -> None:
        rec = {"seq": self.seq, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
               "type": type_, **fields}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        for s in self._secrets:
            line = line.replace(s, "[REDACTED]")
        self.f.write(line + "\n")
        self.f.flush()
        self.seq += 1

    def close(self) -> None:
        self.f.close()


def read_trace(path: Path) -> list[dict]:
    """Read a trace; a final line cut off by a crash is skipped."""
    events = []
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 2:          # only the last line may be partial
                raise
    return events


REQUIRED = {
    "run_start": {"run_id", "task_id", "phase", "agent", "model_id", "reasoning", "budget", "git_commit",
                  "retriever", "system_prompt", "tools"},
    "model_call": {"purpose", "prompt", "text", "tool_calls", "reasoning_summary", "requested_model",
                   "served_model", "stop_reason", "usage", "latency_ms", "cost_usd", "attempt"},
    "search": {"call_id", "arguments", "query", "hits"},
    "read": {"call_id", "arguments", "docid", "found", "text", "truncated"},
    "plan": {"version"},
    "ledger": {"facts", "gaps"},
    "budget": {"limit"},
    "final_answer": {"raw", "explanation", "exact_answer", "confidence", "cited_docids", "forced"},
    "error": {"kind", "message", "retry"},
    "run_end": {"status", "totals", "retrieved_docids", "read_docids"},
}


def validate_trace(events: list[dict]) -> list[str]:
    """Schema check: returns a list of problems (empty if valid)."""
    problems = []
    for i, e in enumerate(events):
        if e.get("seq") != i:
            problems.append(f"event {i}: seq {e.get('seq')}")
        missing = REQUIRED.get(e.get("type"), {"<unknown type>"}) - set(e)
        if missing:
            problems.append(f"event {i} ({e.get('type')}): missing {sorted(missing)}")
    if not events or events[0]["type"] != "run_start":
        problems.append("first event is not run_start")
    if events and events[-1]["type"] != "run_end":
        problems.append("last event is not run_end")
    return problems


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _field(name: str, text: str) -> str | None:
    m = re.search(rf"\**{name}\**\s*:\s*\**(.*?)(?=\n\s*\**(?:Explanation|Exact Answer|Confidence)\**\s*:|\Z)",
                  text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def parse_final(text: str) -> dict:
    explanation = _field("Explanation", text)
    conf = _field("Confidence", text)
    num = re.search(r"\d+(?:\.\d+)?", conf or "")
    cited = []
    for group in re.findall(r"\[([\d,\s]+)\]", explanation or text):
        cited += [d.strip() for d in group.split(",") if d.strip()]
    return {"explanation": explanation, "exact_answer": _field("Exact Answer", text),
            "confidence": float(num.group()) if num else None, "cited_docids": list(dict.fromkeys(cited))}


def is_final(text: str) -> bool:
    return bool(re.search(r"Exact Answer\**\s*:", text or "", re.IGNORECASE))


def extract_json(text: str) -> dict | None:
    """The last JSON object in the text (fenced or bare), or None."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    if not candidates and text and "{" in text:
        candidates = [text[text.index("{"): text.rindex("}") + 1]] if "}" in text else []
    for c in reversed(candidates):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _hit_cap(stop_reason) -> bool:
    s = str(stop_reason or "").lower()
    return "max_tokens" in s or "max_output" in s or s in ("incomplete", "length")


def is_degenerate(r: models.Response, cap: int, dcfg: dict) -> bool:
    """Hit the output cap with no tool call and almost no visible text (or one line repeated)."""
    if r.tool_calls or not (_hit_cap(r.stop_reason) or (r.usage.get("output") or 0) >= cap):
        return False
    visible = re.sub(r"\s", "", r.text or "")
    lines = [ln.strip() for ln in (r.text or "").splitlines() if ln.strip()]
    repeated = max(Counter(lines).values(), default=0)
    return len(visible) < dcfg["min_visible_chars"] or repeated >= dcfg["max_repeated_line"]


def _search_output(hits: list[dict]) -> str:
    return json.dumps([{"docid": h["docid"], "score": h["score"], "snippet": h["snippet"]} for h in hits],
                      indent=2, ensure_ascii=False)


def _read_output(docid: str, found: bool, text: str) -> str:
    return json.dumps({"docid": docid, "text": text}, ensure_ascii=False) if found else f"Document {docid} not found."


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# One run: budget, model calls, tools, final answer
# --------------------------------------------------------------------------- #

class RunError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class Run:
    def __init__(self, *, run_id: str, phase: str, agent: str, task: dict, mcfg: dict, retriever,
                 trace_path: Path, budget: dict | None = None):
        self.b = budget or load_budget()
        self.run_id, self.phase, self.agent, self.task = run_id, phase, agent, task
        self.mcfg = {**mcfg, "max_output_tokens": self.b["max_output_tokens"]}
        self.retriever = retriever
        self.trace = Trace(trace_path)
        self.system = STRATEGIES[agent]
        self.t0 = time.monotonic()
        self.n_model = self.n_tools = self.n_search = self.n_read = 0
        self.cost = 0.0
        self.tokens = Counter()
        self.retrieved: dict[str, None] = {}      # ordered sets
        self.read: dict[str, None] = {}
        self.pending: list[tuple[str, str, str]] = []
        self.status: str | None = None
        self.error: dict | None = None

    # -- lifecycle ----------------------------------------------------------- #
    def start(self) -> None:
        r = self.retriever
        self.trace.log("run_start", run_id=self.run_id, task_id=self.task["query_id"], phase=self.phase,
                       agent=self.agent, model_key=self.mcfg.get("key"), model_id=self.mcfg["model_id"],
                       provider=self.mcfg["provider"], reasoning=self.mcfg.get("reasoning"),
                       max_output_tokens=self.mcfg["max_output_tokens"], price=self.mcfg.get("price"),
                       budget={**self.b, "cost_cap_usd": self.cost_cap}, git_commit=_git_commit(),
                       retriever={"backend": r.backend_name, "index_version": r.index_version, **r.rcfg},
                       system_prompt=self.system, query_template=QUERY_TEMPLATE, tools=TOOLS)

    @property
    def cost_cap(self) -> float:
        return self.b["cost_cap_usd"][self.phase]

    def open(self) -> models.Conversation:
        return models.open_conversation(self.mcfg, system=self.system)

    def prompt(self) -> str:
        return QUERY_TEMPLATE.format(Question=self.task["question"])

    def end(self) -> dict:
        totals = {"model_calls": self.n_model, "tool_calls": self.n_tools, "search": self.n_search,
                  "get_document": self.n_read, "cost_usd": round(self.cost, 6), "tokens": dict(self.tokens),
                  "wall_seconds": round(time.monotonic() - self.t0, 1)}
        self.trace.log("run_end", status=self.status or "error", error=self.error, totals=totals,
                       retrieved_docids=list(self.retrieved), read_docids=list(self.read))
        self.trace.close()
        return {"run_id": self.run_id, "status": self.status, "trace": str(self.trace.path), **totals}

    # -- budget -------------------------------------------------------------- #
    def exhausted(self) -> str | None:
        if self.n_model >= self.b["max_model_calls"]:
            return "model_calls"
        if self.n_tools >= self.b["max_tool_calls"]:
            return "tool_calls"
        if time.monotonic() - self.t0 >= self.b["max_wall_minutes"] * 60:
            return "wall_time"
        if self.cost >= self.cost_cap:
            return "cost"
        return None

    # -- model calls --------------------------------------------------------- #
    def call(self, conv: models.Conversation, purpose: str, tool_choice: str = "auto",
             text: str | None = None) -> models.Response:
        """Flush pending tool results (+ optional user text), call the model, log everything."""
        if self.pending:
            conv.add_tool_results(self.pending, text=text)
            self.pending = []
        elif text:
            conv.add_user(text)
        mark = conv.mark()
        dcfg = self.b["degenerate"]
        for attempt in range(1, dcfg["retries"] + 2):
            prompt = json.loads(json.dumps(conv.native_history(), default=str))
            try:
                r = conv.step(TOOLS, tool_choice=tool_choice)
            except Exception as e:
                self.trace.log("error", kind="api", message=f"{type(e).__name__}: {e}", retry=models.MAX_ATTEMPTS,
                               purpose=purpose)
                raise RunError("api", str(e)) from e
            self.n_model += 1
            self.cost += r.cost_usd
            self.tokens.update({k: v for k, v in r.usage.items() if isinstance(v, int)})
            self.trace.log("model_call", purpose=purpose, attempt=attempt, prompt=prompt, text=r.text,
                           tool_calls=[c.__dict__ for c in r.tool_calls], reasoning_summary=r.reasoning_summary,
                           requested_model=self.mcfg["model_id"], served_model=r.model_id,
                           stop_reason=r.stop_reason, usage=r.usage, latency_ms=round(r.latency_ms, 1),
                           cost_usd=r.cost_usd, request=r.request)
            if not models.served_matches(self.mcfg["model_id"], r.model_id):
                msg = f"requested {self.mcfg['model_id']}, served {r.model_id}"
                self.trace.log("error", kind="model_mismatch", message=msg, retry=0)
                raise RunError("model_mismatch", msg)
            if not is_degenerate(r, self.mcfg["max_output_tokens"], dcfg):
                return r
            self.trace.log("error", kind="degenerate_output", message="output cap hit with almost no visible text",
                           retry=attempt, raw_response=r.text, usage=r.usage)
            conv.rollback(mark)
        raise RunError("degenerate_output", f"degenerate output after {dcfg['retries']} retries")

    # -- tools --------------------------------------------------------------- #
    def execute(self, calls: list[models.ToolCall]) -> tuple[str | None, list[bool]]:
        """Run tool calls in order; results wait in self.pending for the next model call.

        Returns (limit hit or None, per-call productive flags). A call is productive if a search
        surfaces a docid not retrieved before, or a get_document opens a document not read before.
        """
        productive, limit = [], None
        for c in calls:
            if self.n_tools >= self.b["max_tool_calls"]:
                limit = "tool_calls"
                out, ok = "Not executed: the tool-call budget is used up.", False
            elif c.name == "get_document" and self.n_read >= self.b["max_get_document"]:
                self.trace.log("budget", limit="get_document", call_id=c.id)
                out, ok = "Not executed: the get_document budget is used up. You can still search.", False
            else:
                out, ok = self._run_tool(c)
            self.pending.append((c.id, c.name, out))
            productive.append(ok)
        return limit, productive

    def _run_tool(self, c: models.ToolCall) -> tuple[str, bool]:
        self.n_tools += 1
        args = c.arguments or {}
        try:
            if c.name == "search":
                query = args.get("query")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("search needs a non-empty 'query' string")
                self.n_search += 1
                resp = self.retriever.search(query)
                new = [h["docid"] for h in resp["hits"] if h["docid"] not in self.retrieved]
                self.retrieved.update(dict.fromkeys(h["docid"] for h in resp["hits"]))
                usage = resp.get("usage") or {}
                self.cost += usage.get("embedding_cost_usd", 0.0) if not resp.get("cache_hit") else 0.0
                self.trace.log("search", call_id=c.id, arguments=args, query=query, hits=resp["hits"],
                               new_docids=new, backend=resp.get("backend"), index_version=resp.get("index_version"),
                               cache_hit=resp.get("cache_hit"), latency_ms=resp.get("latency_ms"), usage=usage)
                return _search_output(resp["hits"]), bool(new)
            if c.name == "get_document":
                docid = str(args.get("docid") or "").strip()
                if not docid:
                    raise ValueError("get_document needs a 'docid'")
                self.n_read += 1
                doc = self.retriever.get_document(docid)
                new = doc["found"] and docid not in self.read
                if doc["found"]:
                    self.read[docid] = None
                self.trace.log("read", call_id=c.id, arguments=args, docid=docid, found=doc["found"],
                               text=doc["text"], truncated=doc["truncated"], returned_tokens=doc["returned_tokens"],
                               original_tokens=doc["original_tokens"], cache_hit=doc.get("cache_hit"),
                               latency_ms=doc.get("latency_ms"))
                return _read_output(docid, doc["found"], doc["text"]), bool(new)
            raise ValueError(f"unknown tool {c.name!r}")
        except ValueError as e:
            self.trace.log("error", kind="tool", message=str(e), retry=0, call_id=c.id, tool=c.name, arguments=args)
            return f"Error: {e}", False
        except Exception as e:      # retriever / embedding API failure: not a study failure
            self.trace.log("error", kind="api", message=f"{type(e).__name__}: {e}", retry=0, call_id=c.id)
            raise RunError("api", str(e)) from e

    # -- endings ------------------------------------------------------------- #
    def final(self, text: str, status: str) -> None:
        self.trace.log("final_answer", raw=text, forced=status == "budget_exhausted", **parse_final(text))
        self.status = status

    def force_final(self, conv: models.Conversation, limit: str) -> None:
        """Budget hit: one extra call with tools disabled, asking for the answer."""
        self.trace.log("budget", limit=limit)
        r = self.call(conv, "final", tool_choice="none", text=FINAL_REQUEST)
        self.final(r.text, "budget_exhausted")


# --------------------------------------------------------------------------- #
# The three strategies
# --------------------------------------------------------------------------- #

def react(run: Run) -> None:
    conv = run.open()
    conv.add_user(run.prompt())
    while True:
        if limit := run.exhausted():
            return run.force_final(conv, limit)
        r = run.call(conv, "act")
        if not r.tool_calls:
            return run.final(r.text, "completed")
        limit, _ = run.execute(r.tool_calls)
        if limit:
            return run.force_final(conv, limit)


def plan_execute(run: Run) -> None:
    conv = run.open()
    conv.add_user(run.prompt() + "\n\n" + PLAN_FIRST)
    findings: list[str] = []
    version = 0

    def make_plan(text: str | None, reason: str | None) -> list[dict] | None:
        """One planning call (tools disabled). Returns steps, or None if the model answered instead."""
        nonlocal version
        r = run.call(conv, "plan", tool_choice="none", text=text)
        if is_final(r.text):
            run.final(r.text, "completed")
            return None
        version += 1
        data = extract_json(r.text) or {}
        steps = [{"step": i + 1, "goal": str(s.get("goal") if isinstance(s, dict) else s)}
                 for i, s in enumerate(data.get("plan") or [])]
        run.trace.log("plan", version=version, reason=reason, steps=steps, parse_error=not steps)
        return steps

    def replan(reason: str) -> list[dict] | None:
        return make_plan(REPLAN.format(reason=reason, findings="\n".join(findings) or "(none)"), reason)

    plan = make_plan(None, None)
    i, unproductive, new_step, nudge = 0, 0, True, None
    while plan is not None:
        if limit := run.exhausted():
            return run.force_final(conv, limit)
        if i >= len(plan):
            plan = replan("every step of the plan is done or blocked and the question is not answered yet")
            i, unproductive, new_step = 0, 0, True
            continue
        step = plan[i]
        # A new step gets its instruction; a continuing step gets only the pending tool results.
        r = run.call(conv, "act", text=EXECUTE_STEP.format(**step) if new_step else nudge)
        new_step, nudge = False, None
        if r.tool_calls:
            limit, flags = run.execute(r.tool_calls)
            if limit:
                return run.force_final(conv, limit)
            for ok in flags:
                unproductive = 0 if ok else unproductive + 1
            if unproductive >= run.b["blocked_after_unproductive_calls"]:
                why = f"{unproductive} tool calls in a row found no new documents"
                findings.append(f"Step {step['step']} (blocked): {why}")
                run.trace.log("plan", version=version, update={"step": step["step"], "status": "blocked", "by": "auto"})
                plan = replan(f"step {step['step']} is blocked ({why})")
                i, unproductive, new_step = 0, 0, True
            continue
        if is_final(r.text):
            return run.final(r.text, "completed")
        report = extract_json(r.text) or {}
        status = report.get("status")
        if status in ("done", "blocked"):
            findings.append(f"Step {step['step']} ({status}): {report.get('finding', '')}")
            run.trace.log("plan", version=version, update={"step": step["step"], "status": status, "by": "model",
                                                           "finding": report.get("finding")})
            if status == "blocked":
                plan = replan(f"step {step['step']} was reported blocked")
                i = 0
            else:
                i += 1
            unproductive, new_step = 0, True
        else:
            nudge = PLAN_NUDGE


def evidence_driven(run: Run) -> None:
    conv = run.open()
    conv.add_user(run.prompt())
    next_text = None
    while True:
        if limit := run.exhausted():
            return run.force_final(conv, limit)
        r = run.call(conv, "act", text=next_text)
        next_text = None
        ledger = extract_json(r.text)
        has_ledger = isinstance(ledger, dict) and ("facts" in ledger or "gaps" in ledger)
        if has_ledger:
            run.trace.log("ledger", facts=ledger.get("facts") or [], gaps=ledger.get("gaps") or [], parse_error=False)
        elif r.tool_calls:
            run.trace.log("ledger", facts=None, gaps=None, parse_error=True)
        if r.tool_calls:
            limit, _ = run.execute(r.tool_calls)
            if limit:
                return run.force_final(conv, limit)
            continue
        if is_final(r.text):
            return run.final(r.text, "completed")
        next_text = LEDGER_DONE if has_ledger and not ledger.get("gaps") else LEDGER_NUDGE


AGENTS: dict[str, Callable[[Run], None]] = {"react": react, "plan_execute": plan_execute,
                                            "evidence_driven": evidence_driven}


def run_agent(agent: str, task: dict, mcfg: dict, retriever, trace_path: Path, *, run_id: str,
              phase: str, budget: dict | None = None) -> dict:
    """Run one agent on one task. task is a TaskView: {query_id, question} only."""
    if set(task) != {"query_id", "question"}:
        raise ValueError("agents receive only {query_id, question}")
    run = Run(run_id=run_id, phase=phase, agent=agent, task=task, mcfg=mcfg, retriever=retriever,
              trace_path=trace_path, budget=budget)
    run.start()
    try:
        AGENTS[agent](run)
    except RunError as e:
        run.status, run.error = "error", {"kind": e.kind, "message": str(e)}
    except Exception as e:      # a crash in our code: logged, the run is rerun
        run.trace.log("error", kind="crash", message=f"{type(e).__name__}: {e}", retry=0)
        run.status, run.error = "error", {"kind": "crash", "message": str(e)}
    return run.end()


# --------------------------------------------------------------------------- #
# Export to the official BrowseComp-Plus run format
# --------------------------------------------------------------------------- #

def export_official(events: list[dict]) -> dict:
    start, end = events[0], events[-1]
    result = []
    for e in events:
        if e["type"] == "model_call" and e.get("reasoning_summary"):
            result.append({"type": "reasoning", "tool_name": None, "arguments": None,
                           "output": [{"text": e["reasoning_summary"]}]})
        elif e["type"] == "search":
            result.append({"type": "tool_call", "tool_name": "search", "arguments": json.dumps(e["arguments"]),
                           "output": _search_output(e["hits"])})
        elif e["type"] == "read":
            result.append({"type": "tool_call", "tool_name": "get_document", "arguments": json.dumps(e["arguments"]),
                           "output": _read_output(e["docid"], e["found"], e["text"])})
        elif e["type"] == "final_answer":
            result.append({"type": "output_text", "tool_name": None, "arguments": None, "output": e["raw"]})
    counts = end["totals"]
    status = {"completed": "completed", "budget_exhausted": "incomplete"}.get(end["status"], "error")
    return {"query_id": start["task_id"], "status": status,
            "tool_call_counts": {"search": counts["search"], "get_document": counts["get_document"]},
            "search_counts": counts["search"], "retrieved_docids": end["retrieved_docids"], "result": result}


# --------------------------------------------------------------------------- #
# CLI: debug runs on non-study tasks only (the study runner is spec 03)
# --------------------------------------------------------------------------- #

def debug_task(seed: int, index: int = 0) -> str:
    """A question every published run solved: never one of the 10 study tasks."""
    import random

    study = set(yaml.safe_load((ROOT / "configs" / "tasks.yaml").read_text())["phase1_task_ids"])
    from src.benchmark import load_correctness

    easy = sorted(q for q, r in load_correctness().items() if all(r.values()) and q not in study)
    random.Random(f"{seed}-debug").shuffle(easy)
    return easy[index]


def main(argv: list[str] | None = None) -> None:
    import argparse

    from src import benchmark as bm

    ap = argparse.ArgumentParser(prog="python -m src.agents")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("try", help="one real run on a non-study task (spends money)")
    t.add_argument("--agent", choices=list(AGENTS), required=True)
    t.add_argument("--model", default="haiku")
    t.add_argument("--task-index", type=int, default=0)
    args = ap.parse_args(argv)

    bm.load_env()
    cfg = bm.load_config()
    qid = debug_task(cfg["selection"]["seed"], args.task_index)
    task = bm.load_tasks()[qid]
    mcfg = models.load_model(args.model)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"debug-{qid}-{args.agent}-{args.model}-{stamp}"
    backend = bm.make_backend(cfg)
    retriever = bm.Retriever(backend, cfg)
    summary = run_agent(args.agent, task.view(), mcfg, retriever, ROOT / "results" / "debug" / f"{run_id}.jsonl",
                        run_id=run_id, phase="debug")
    events = read_trace(Path(summary["trace"]))
    final = next((e for e in events if e["type"] == "final_answer"), None)
    if final:
        summary["judge_correct"] = bm.Judge(cfg).judge(task, final["raw"])["correct"]
    summary["schema_problems"] = validate_trace(events)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
