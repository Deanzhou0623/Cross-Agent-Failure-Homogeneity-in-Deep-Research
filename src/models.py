"""Provider wrappers (spec 01, items 13-16).

Each provider gets a small Conversation class that keeps the provider-native
message history (so thinking blocks, reasoning items, and thought signatures
round-trip exactly as each API requires) and returns a normalized Response.

    conv = open_conversation(load_model("haiku"), system="...")
    conv.add_user("question ...")
    resp = conv.step(tools=TOOLS)          # Response(text, tool_calls, usage, cost_usd, ...)
    conv.add_tool_results([(call.id, call.name, "result text"), ...])

Refusal fallbacks to other models are deliberately NOT enabled: a silent model
switch would confound the experiment. Refusals are logged via stop_reason.

CLI:  python -m src.models check [--model haiku]   # one real call per model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
MAX_ATTEMPTS = 3


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Response:
    text: str
    tool_calls: list[ToolCall]
    reasoning_summary: str | None      # only what the provider exposes; never raw chain of thought
    usage: dict                         # input, cached_input, cache_write, output, reasoning (None if not reported)
    cost_usd: float
    latency_ms: float
    stop_reason: str | None
    model_id: str
    request: dict = field(default_factory=dict)   # exact non-message parameters sent


def load_model(key: str, path: Path | None = None) -> dict:
    cfg = yaml.safe_load((path or ROOT / "configs" / "models.yaml").read_text())
    for phase in ("phase1", "phase2"):
        if key in cfg.get(phase, {}):
            return {**cfg[phase][key], "key": key, "phase": phase,
                    "max_output_tokens": cfg.get("max_output_tokens", 32000)}
    raise KeyError(f"unknown model {key!r} in configs/models.yaml")


def all_model_keys(path: Path | None = None) -> list[str]:
    cfg = yaml.safe_load((path or ROOT / "configs" / "models.yaml").read_text())
    return [k for phase in ("phase1", "phase2") for k in cfg.get(phase, {})]


def compute_cost(price: dict, usage: dict) -> float:
    """USD from normalized usage. 'input' excludes cached and cache-write tokens."""
    prompt_total = (usage.get("input") or 0) + (usage.get("cached_input") or 0) + (usage.get("cache_write") or 0)
    long = price.get("long_context_threshold") and prompt_total > price["long_context_threshold"]
    p_in = price["long_input"] if long else price["input"]
    p_out = price["long_output"] if long else price["output"]
    scale = p_in / price["input"]
    return ((usage.get("input") or 0) * p_in
            + (usage.get("cached_input") or 0) * price.get("cached_input", price["input"]) * scale
            + (usage.get("cache_write") or 0) * price.get("cache_write", price["input"]) * scale
            + (usage.get("output") or 0) * p_out) / 1e6


def _retry(fn):
    """Retry transient API errors (rate limits, 5xx, network) with backoff. Others raise."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(), attempt
        except Exception as e:  # provider SDKs use different exception types
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or "connection" in type(e).__name__.lower() \
                or "timeout" in type(e).__name__.lower()
            if not transient or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 ** attempt)


class Conversation:
    provider = "base"

    def __init__(self, mcfg: dict, system: str):
        self.mcfg = mcfg
        self.system = system
        self.model_id = mcfg["model_id"]

    def add_user(self, text: str) -> None:
        raise NotImplementedError

    def add_tool_results(self, results: list[tuple[str, str, str]], text: str | None = None) -> None:
        """Append tool outputs (call_id, name, output), plus an optional user text in the same turn."""
        raise NotImplementedError

    def step(self, tools: list[dict], tool_choice: str = "auto") -> Response:
        """One model call. tool_choice="none" keeps the tools defined but forbids calling them."""
        raise NotImplementedError

    def native_history(self) -> list:
        raise NotImplementedError

    def _store(self) -> list:
        raise NotImplementedError

    def mark(self) -> int:
        return len(self._store())

    def rollback(self, mark: int) -> None:
        """Drop everything appended after mark (used to retry a degenerate call)."""
        del self._store()[mark:]


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #

class AnthropicConversation(Conversation):
    provider = "anthropic"

    def __init__(self, mcfg: dict, system: str):
        super().__init__(mcfg, system)
        import anthropic

        self.client = anthropic.Anthropic()
        self.messages: list[dict] = []

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results, text=None):
        content = [{"type": "tool_result", "tool_use_id": cid, "content": out} for cid, _, out in results]
        if text:
            content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})

    def step(self, tools, tool_choice="auto"):
        request = {
            "model": self.model_id,
            "max_tokens": self.mcfg["max_output_tokens"],
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.mcfg["reasoning"]["effort"]},
            "cache_control": {"type": "ephemeral"},
        }
        params = {**request, "system": [{"type": "text", "text": self.system}],
                  "messages": self.messages}
        if tools:
            params["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
            if tool_choice == "none":
                params["tool_choice"] = request["tool_choice"] = {"type": "none"}
        t0 = time.monotonic()

        def call():
            with self.client.messages.stream(**params) as stream:
                return stream.get_final_message()

        msg, attempts = _retry(call)
        latency = (time.monotonic() - t0) * 1000
        content = [b.model_dump(exclude_none=True) for b in msg.content]
        self.messages.append({"role": "assistant", "content": content})
        text = "".join(b.text for b in msg.content if b.type == "text")
        summary = "\n".join(b.thinking for b in msg.content if b.type == "thinking" and b.thinking) or None
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in msg.content if b.type == "tool_use"]
        u = msg.usage
        usage = {"input": u.input_tokens, "cached_input": u.cache_read_input_tokens or 0,
                 "cache_write": u.cache_creation_input_tokens or 0, "output": u.output_tokens,
                 "reasoning": None}
        return Response(text, calls, summary, usage, compute_cost(self.mcfg["price"], usage), latency,
                        msg.stop_reason, msg.model, {**request, "attempts": attempts})

    def native_history(self):
        return self.messages

    def _store(self):
        return self.messages


# --------------------------------------------------------------------------- #
# OpenAI (Responses API)
# --------------------------------------------------------------------------- #

class OpenAIConversation(Conversation):
    provider = "openai"

    def __init__(self, mcfg: dict, system: str):
        super().__init__(mcfg, system)
        import openai

        self.client = openai.OpenAI()
        self.items: list[dict] = []
        self.cache_key = hashlib.sha256(f"{self.model_id}\n{system}".encode()).hexdigest()[:32]

    def add_user(self, text):
        self.items.append({"role": "user", "content": text})

    def add_tool_results(self, results, text=None):
        for cid, _, out in results:
            self.items.append({"type": "function_call_output", "call_id": cid, "output": out})
        if text:
            self.items.append({"role": "user", "content": text})

    def step(self, tools, tool_choice="auto"):
        request = {
            "model": self.model_id,
            "reasoning": {"effort": self.mcfg["reasoning"]["effort"], "summary": "auto"},
            "max_output_tokens": self.mcfg["max_output_tokens"],
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self.cache_key,
        }
        params = {**request, "instructions": self.system, "input": self.items}
        if tools:
            params["tools"] = [{"type": "function", "name": t["name"], "description": t["description"],
                                "parameters": t["parameters"], "strict": False} for t in tools]
            if tool_choice == "none":
                params["tool_choice"] = request["tool_choice"] = "none"
        t0 = time.monotonic()
        resp, attempts = _retry(lambda: self.client.responses.create(**params))
        latency = (time.monotonic() - t0) * 1000
        summaries, calls = [], []
        for item in resp.output:
            self.items.append(item.model_dump(exclude_none=True))
            if item.type == "reasoning":
                summaries += [s.text for s in (item.summary or [])]
            elif item.type == "function_call":
                calls.append(ToolCall(item.call_id, item.name, json.loads(item.arguments or "{}")))
        u = resp.usage
        cached = (u.input_tokens_details.cached_tokens if u.input_tokens_details else 0) or 0
        usage = {"input": u.input_tokens - cached, "cached_input": cached, "cache_write": 0,
                 "output": u.output_tokens,
                 "reasoning": u.output_tokens_details.reasoning_tokens if u.output_tokens_details else None}
        return Response(resp.output_text or "", calls, "\n".join(summaries) or None, usage,
                        compute_cost(self.mcfg["price"], usage), latency,
                        getattr(resp, "status", None), resp.model, {**request, "attempts": attempts})

    def native_history(self):
        return self.items

    def _store(self):
        return self.items


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #

class GeminiConversation(Conversation):
    provider = "gemini"

    def __init__(self, mcfg: dict, system: str):
        super().__init__(mcfg, system)
        from google import genai

        self.client = genai.Client()
        self.contents: list = []

    def add_user(self, text):
        from google.genai import types

        self.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    def add_tool_results(self, results, text=None):
        from google.genai import types

        parts = [types.Part(function_response=types.FunctionResponse(id=cid or None, name=name,
                                                                     response={"result": out}))
                 for cid, name, out in results]
        if text:
            parts.append(types.Part(text=text))
        self.contents.append(types.Content(role="user", parts=parts))

    def step(self, tools, tool_choice="auto"):
        from google.genai import types

        request = {
            "model": self.model_id,
            "thinking_config": {"thinking_level": self.mcfg["reasoning"]["thinking_level"],
                                "include_thoughts": True},
            "max_output_tokens": self.mcfg["max_output_tokens"],
        }
        config = types.GenerateContentConfig(
            system_instruction=self.system,
            thinking_config=types.ThinkingConfig(**request["thinking_config"]),
            max_output_tokens=request["max_output_tokens"],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["parameters"]) for t in tools])]
            if tools else None,
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))
            if tools and tool_choice == "none" else None,
        )
        if tools and tool_choice == "none":
            request["tool_choice"] = "none"
        t0 = time.monotonic()
        resp, attempts = _retry(lambda: self.client.models.generate_content(
            model=self.model_id, contents=self.contents, config=config))
        latency = (time.monotonic() - t0) * 1000
        cand = resp.candidates[0]
        self.contents.append(cand.content)
        parts = cand.content.parts or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        summary = "\n".join(p.text for p in parts if p.text and p.thought) or None
        calls = [ToolCall(p.function_call.id or f"call_{i}", p.function_call.name, dict(p.function_call.args or {}))
                 for i, p in enumerate(parts) if p.function_call]
        m = resp.usage_metadata
        cached = m.cached_content_token_count or 0
        thoughts = m.thoughts_token_count or 0
        usage = {"input": (m.prompt_token_count or 0) - cached, "cached_input": cached, "cache_write": 0,
                 "output": (m.candidates_token_count or 0) + thoughts, "reasoning": thoughts}
        return Response(text, calls, summary, usage, compute_cost(self.mcfg["price"], usage), latency,
                        str(cand.finish_reason) if cand.finish_reason else None,
                        getattr(resp, "model_version", None) or self.model_id, {**request, "attempts": attempts})

    def native_history(self):
        return [c.model_dump(exclude_none=True, mode="json") for c in self.contents]

    def _store(self):
        return self.contents


# --------------------------------------------------------------------------- #
# Mock (tests and development; never calls an API)
# --------------------------------------------------------------------------- #

class MockConversation(Conversation):
    """Replays mcfg['script']: a list of steps, each with "tool_calls": [[name, args], ...] and/or "text".

    Optional per step: "stop_reason", "served_model", "output_tokens", "reasoning_summary".
    A step whose tool calls arrive while tool_choice="none" has its calls dropped, as a real API would.
    """
    provider = "mock"

    def __init__(self, mcfg: dict, system: str):
        super().__init__(mcfg, system)
        self.script = list(mcfg.get("script", [{"text": "Explanation: mock\nExact Answer: mock\nConfidence: 50%"}]))
        self.history: list[dict] = [{"role": "system", "content": system}]
        self.n = 0

    def add_user(self, text):
        self.history.append({"role": "user", "content": text})

    def add_tool_results(self, results, text=None):
        self.history.append({"role": "tool", "content": [{"id": c, "name": n, "output": o} for c, n, o in results]})
        if text:
            self.history.append({"role": "user", "content": text})

    def step(self, tools, tool_choice="auto"):
        step = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        calls = [] if tool_choice == "none" else [
            ToolCall(f"mock_{self.n}_{i}", name, args) for i, (name, args) in enumerate(step.get("tool_calls", []))]
        text = step.get("text", "")
        self.history.append({"role": "assistant", "content": text, "tool_calls": [c.__dict__ for c in calls]})
        usage = {"input": 100, "cached_input": 0, "cache_write": 0, "output": step.get("output_tokens", 20),
                 "reasoning": None}
        stop = step.get("stop_reason", "tool_use" if calls else "end_turn")
        return Response(text, calls, step.get("reasoning_summary"), usage, 0.0, 0.0, stop,
                        step.get("served_model", self.model_id), {"model": self.model_id, "tool_choice": tool_choice})

    def native_history(self):
        return self.history

    def _store(self):
        return self.history


PROVIDERS = {"anthropic": AnthropicConversation, "openai": OpenAIConversation,
             "gemini": GeminiConversation, "mock": MockConversation}


def open_conversation(mcfg: dict, system: str) -> Conversation:
    return PROVIDERS[mcfg["provider"]](mcfg, system)


# --------------------------------------------------------------------------- #
# One-call check per model (spec 01, item 16)
# --------------------------------------------------------------------------- #

CHECK_TOOL = {"name": "search", "description": "Search a document collection. Returns the top results.",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                             "required": ["query"]}}


def served_matches(requested: str, served: str | None) -> bool:
    """True if the provider served the requested model (allowing dated snapshots like `-2026-08-01`).

    Providers can silently route retired IDs to other models (gemini-3.7-flash -> gemini-3.8-flash).
    """
    if not served:
        return False
    served = served.removeprefix("models/")
    return served == requested or served.startswith(requested + "-")


def check_model(key: str) -> dict:
    """Two short calls: one that should request the tool, one with a tool result. About $0.01."""
    mcfg = load_model(key)
    mcfg = {**mcfg, "max_output_tokens": 4000}
    conv = open_conversation(mcfg, system="Use the search tool before answering.")
    conv.add_user("Which river flows through Paris? Search first, then answer in one word.")
    r1 = conv.step([CHECK_TOOL])
    out: dict[str, Any] = {"model": key, "model_id_returned": r1.model_id,
                           "served_matches_requested": served_matches(mcfg["model_id"], r1.model_id),
                           "request": r1.request,
                           "first_call": {"tool_calls": [c.__dict__ for c in r1.tool_calls],
                                          "usage": r1.usage, "has_reasoning_summary": bool(r1.reasoning_summary),
                                          "stop_reason": r1.stop_reason}}
    if r1.tool_calls:
        conv.add_tool_results([(c.id, c.name, "[doc1] The Seine flows through Paris.") for c in r1.tool_calls])
        r2 = conv.step([CHECK_TOOL])
        out["second_call"] = {"text": r2.text[:200], "usage": r2.usage}
        out["cost_usd"] = round(r1.cost_usd + r2.cost_usd, 5)
    else:
        out["cost_usd"] = round(r1.cost_usd, 5)
    return out


def main(argv: list[str] | None = None) -> None:
    from src.benchmark import load_env

    load_env()
    ap = argparse.ArgumentParser(prog="python -m src.models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--model", action="append", help="model key; repeatable; default all six")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        for key in args.model or all_model_keys():
            try:
                print(json.dumps(check_model(key), indent=2, default=str))
            except Exception as e:
                print(json.dumps({"model": key, "error": f"{type(e).__name__}: {e}"}))


if __name__ == "__main__":
    main()
