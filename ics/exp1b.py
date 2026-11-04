"""Experiment 1b - language drift in a realistic project setting.

Condition = model x rule_placement x thinking x task x seed.

rule_placement
  system : system prompt holds ONLY the rule; the project md (with the rule line removed)
           is still sent in the first user message, so both arms see the same context and
           differ only in where the rule lives.
  md     : no system prompt; the rule is one line inside the project md, which is sent in
           the first user message (as Claude Code does with CLAUDE.md).
thinking   off / low / high  (see ics/client.py THINKING for the exact API params)
task       qa (short questions) / coding (debug, design, refactor on the fake project)

Per turn we log the full reply, the summarized thinking, stop_reason and token usage,
and classify violations:
  knew_but_failed : thinking mentions the language rule, output is not Chinese
  forgot          : thinking exists but never mentions the rule
  no_thinking     : output violated and there is no visible thinking to inspect
"""
import itertools
import json
import random
import re
from pathlib import Path

from .client import make_client, thinking_params
from .detect import CHECKS, language_profile

RULE_MENTION = re.compile(r"中文|汉语|普通话|Chinese|Mandarin", re.I)
LANG_MENTION = re.compile(r"\blanguage\b|语言", re.I)
RULE_MARKER = "{{RULE}}"


def load_tasks(path: str) -> list[str]:
    """Tasks separated by a line containing only '====='. '#' lines before a task are comments."""
    raw = Path(path).read_text(encoding="utf-8")
    if not re.search(r"^=====\s*$", raw, flags=re.M):  # one task per line (QA files)
        return [l.strip() for l in raw.splitlines() if l.strip() and not l.startswith("#")]
    tasks = []
    for chunk in re.split(r"^=====\s*$", raw, flags=re.M):
        lines = [l for l in chunk.strip("\n").splitlines() if not l.startswith("#")]
        t = "\n".join(lines).strip()
        if t:
            tasks.append(t)
    return tasks


def build_md(cfg: dict, placement: str) -> str:
    md = Path(cfg["project_md"]).read_text(encoding="utf-8")
    # strip HTML comments: they hold notes for the experimenter, never for the model
    md = re.sub(r"<!--.*?-->\s*", "", md, flags=re.S)
    assert RULE_MARKER in md, f"{cfg['project_md']} must contain {RULE_MARKER} where the rule goes"
    if placement == "md":
        return md.replace(RULE_MARKER, cfg["rule"]["md_line"])
    # drop the whole line holding the marker
    return "\n".join(l for l in md.splitlines() if RULE_MARKER not in l)


def first_user_message(md: str, task: str) -> str:
    return ("Below are the project instructions from CLAUDE.md at the repository root.\n\n"
            f"<project_instructions>\n{md}\n</project_instructions>\n\n{task}")


def classify(compliant, thinking: str):
    if compliant is not False:
        return None
    if not thinking.strip():
        return "no_thinking"
    return "knew_but_failed" if RULE_MENTION.search(thinking) else "forgot"


def run_one(cfg, model, placement, think, task_kind, seed, out):
    params = thinking_params(model, think)
    if params is None:
        print(f"[skip] {model} cannot run thinking={think}")
        return
    check = CHECKS[cfg["rule"]["check"]]
    tasks = load_tasks(cfg["tasks"][task_kind])
    rng = random.Random(seed)
    if cfg.get("shuffle_tasks", True):
        rng.shuffle(tasks)
    tasks = tasks[: cfg["n_turns"]]

    md = build_md(cfg, placement)
    system = cfg["rule"]["system_text"] if placement == "system" else None
    client = make_client(model, seed=seed, max_tokens=cfg.get("max_tokens", 64000),
                         cache=cfg.get("cache", True))
    run_id = f"{model}|{placement}|{think}|{task_kind}|s{seed}"
    messages = []
    for i, task in enumerate(tasks, start=1):
        user = first_user_message(md, task) if i == 1 else task
        messages.append({"role": "user", "content": user})
        r = client.chat(system, messages, params)
        # pass the assistant turn back unchanged (thinking blocks + signature included)
        messages.append({"role": "assistant", "content": r["content"] or [{"type": "text", "text": ""}]})
        compliant = check(r["text"])
        rec = {
            "run_id": run_id, "model": model, "placement": placement, "thinking_level": think,
            "task_kind": task_kind, "seed": seed, "turn": i,
            **r["usage"], "stop_reason": r["stop_reason"], "block_types": r["block_types"],
            "compliant": compliant,
            "violation_type": classify(compliant, r["thinking"]),
            "lang": language_profile(r["text"]),
            "thinking_lang": language_profile(r["thinking"]) if r["thinking"] else None,
            "thinking_mentions_rule": bool(RULE_MENTION.search(r["thinking"])) if r["thinking"] else None,
            "thinking_mentions_language": bool(LANG_MENTION.search(r["thinking"])) if r["thinking"] else None,
            "user": task, "reply": r["text"], "thinking": r["thinking"],
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        tag = {True: ".", False: "X", None: "?"}[compliant]
        print(f"  t{i:02d} {tag} ctx={rec['total_input_tokens']:>7} think={rec['thinking_tokens']} "
              f"out={rec['output_tokens']} {rec['stop_reason']}", flush=True)
        if cfg.get("stop_at_first_violation", False) and compliant is False:
            break


def run_experiment(cfg: dict, out_path: str, done: set | None = None):
    done = done or set()
    levels = ["off" if l is False else str(l) for l in cfg["thinking_levels"]]  # YAML: off -> False
    grid = itertools.product(cfg["models"], cfg["rule_placements"], levels,
                             cfg["task_kinds"], range(cfg["seeds"]))
    with open(out_path, "a", encoding="utf-8") as out:
        for model, placement, think, task_kind, seed in grid:
            rid = f"{model}|{placement}|{think}|{task_kind}|s{seed}"
            if rid in done:
                continue
            print(f"[run] {rid}", flush=True)
            run_one(cfg, model, placement, think, task_kind, seed, out)
