"""Model clients. `MockClient` lets the whole pipeline run without an API key.

API facts this file relies on (platform.claude.com docs, checked 2026-11-03):
- `temperature` is rejected by the current Messages API.
- Thinking depth on the 5.5 models is controlled by `output_config.effort`
  (low/medium/high/xhigh/max) with adaptive thinking; `budget_tokens` returns 400.
- Turning thinking OFF: Haiku 5.5 -> thinking.type "disabled" (only at effort <= high);
  Sonnet 5.5 -> "between_tools" (no tools => no thinking; effort <= high);
  Opus 5.5 -> impossible (adaptive thinking always on).
- Thinking text is only returned with display="summarized", and it is a SUMMARY written
  by a different model, not the raw chain of thought.
- Thinking blocks must be passed back unchanged in later turns.
- With caching, usage.input_tokens excludes cached tokens:
  total prompt = input_tokens + cache_read_input_tokens + cache_creation_input_tokens.
"""
import os
import random
import time

# thinking level -> request params, per model family. None = combination not possible.
THINKING = {
    "haiku": {
        "default": {"thinking": {"type": "adaptive", "display": "summarized"}},  # model's default effort
        "off":  {"thinking": {"type": "disabled"}, "effort": "high"},
        "low":  {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "low"},
        "high": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "high"},
    },
    "sonnet": {
        "default": {"thinking": {"type": "adaptive", "display": "summarized"}},
        "off":  {"thinking": {"type": "between_tools"}, "effort": "high"},
        "low":  {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "low"},
        "high": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "high"},
    },
    "fable": {  # Fable 5.1: adaptive thinking always on (like Opus 5.5)
        "default": {"thinking": {"type": "adaptive", "display": "summarized"}},
        "off":  None,
        "low":  {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "low"},
        "high": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "high"},
    },
    "opus": {
        "default": {"thinking": {"type": "adaptive", "display": "summarized"}},
        "off":  None,
        "low":  {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "low"},
        "high": {"thinking": {"type": "adaptive", "display": "summarized"}, "effort": "high"},
    },
}


def family(model: str) -> str:
    for f in ("haiku", "sonnet", "opus", "fable"):
        if f in model:
            return f
    return "mock"


def thinking_params(model: str, level: str | None):
    if level is None:
        return {}
    if model.startswith("mock"):
        return {"level": level}
    return THINKING[family(model)][level]


def _blocks_to_dicts(content):
    out = []
    for b in content:
        d = b.model_dump(exclude_none=True)
        d.pop("citations", None)
        out.append(d)
    return out


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 64000, cache: bool = True, **_):
        import anthropic
        from dotenv import load_dotenv

        load_dotenv(".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is empty - put your Console key in .env")
        self.c = anthropic.Anthropic()
        self.model, self.max_tokens, self.cache = model, max_tokens, cache

    def chat(self, system, messages, params=None):
        """Returns dict with assistant `content` (raw blocks, to append to history),
        `text`, `thinking` (summarized text), `stop_reason`, `usage`."""
        import anthropic
        params = params or {}
        kw = dict(model=self.model, max_tokens=self.max_tokens, messages=messages)
        if system:
            kw["system"] = system
        if self.cache:
            kw["cache_control"] = {"type": "ephemeral"}  # automatic, moves with the conversation
        if "thinking" in params:
            kw["thinking"] = params["thinking"]
        if "effort" in params:
            kw["output_config"] = {"effort": params["effort"]}
        for k in ("tools", "tool_choice"):
            if k in params:
                kw[k] = params[k]
        for attempt in range(6):
            try:
                with self.c.messages.stream(**kw) as s:  # streaming: large max_tokens
                    m = s.get_final_message()
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt + random.random())
        u = m.usage
        details = getattr(u, "output_tokens_details", None)
        usage = {
            "input_tokens": u.input_tokens,
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
            "output_tokens": u.output_tokens,
            "thinking_tokens": getattr(details, "thinking_tokens", None) if details else None,
        }
        usage["total_input_tokens"] = (usage["input_tokens"] + usage["cache_read_input_tokens"]
                                       + usage["cache_creation_input_tokens"])
        return {
            "content": _blocks_to_dicts(m.content),
            "text": "".join(b.text for b in m.content if b.type == "text"),
            "thinking": "\n".join(b.thinking for b in m.content if b.type == "thinking"),
            "block_types": [b.type for b in m.content],
            "stop_reason": m.stop_reason,
            "usage": usage,
        }


class MockClient:
    """Answers in Chinese, then drifts to English with a hazard that grows with context.
    Only for testing the pipeline - its numbers mean nothing."""

    def __init__(self, model: str = "mock", seed: int = 0, **_):
        self.model, self.rng = model, random.Random(seed)

    def chat(self, system, messages, params=None):
        ctx = sum(len(str(m["content"])) for m in messages) + len(system or "")
        level = (params or {}).get("level")
        hazard = min(0.9, ctx / 40000 * (2 if level == "high" else 1))
        drift = self.rng.random() < hazard
        text = "Sure - here is the answer in English." if drift else "好的，下面是我的回答。"
        thinking = "" if level in (None, "off") else (
            "The user wrote in English. I'll answer." if drift and self.rng.random() < .5
            else "Rule says answer in Chinese. The user wrote in English.")
        content = ([{"type": "thinking", "thinking": thinking, "signature": "x"}] if thinking else []) \
            + [{"type": "text", "text": text}]
        stop = "end_turn"
        p = params or {}
        if "tools" in p and p.get("tool_choice", {}).get("type") != "none" and self.rng.random() < .6:
            content.append({"type": "tool_use", "id": f"toolu_mock_{self.rng.randrange(10**6)}",
                            "name": "bash", "input": {"command": "pytest -q"}})
            stop = "tool_use"
        n = ctx // 3
        return {"content": content, "text": text, "thinking": thinking,
                "block_types": [c["type"] for c in content], "stop_reason": stop,
                "usage": {"input_tokens": 50, "cache_read_input_tokens": n - 50,
                          "cache_creation_input_tokens": 0, "output_tokens": 100,
                          "thinking_tokens": 60 if thinking else 0, "total_input_tokens": n}}


def make_client(model: str, seed: int = 0, max_tokens: int = 64000, cache: bool = True, **_):
    if model.startswith("mock"):
        return MockClient(model, seed=seed)
    return AnthropicClient(model, max_tokens=max_tokens, cache=cache)
