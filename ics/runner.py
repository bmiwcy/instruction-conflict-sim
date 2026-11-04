"""Experiment 1 runner: one rule stated once, then N scripted user turns.

Every turn is logged to JSONL so analysis never needs to re-call the API.
A run's condition = (model, rule_placement, user_lang, seed).
"""
import itertools
import json
import random
from pathlib import Path

from .client import make_client
from .detect import CHECKS, language_profile


def load_turns(path: str) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def run_one(cfg: dict, model: str, placement: str, user_lang: str, seed: int, out):
    rule = cfg["rule"]["text"]
    check = CHECKS[cfg["rule"]["check"]]
    turns = load_turns(cfg["user_turns"][user_lang])
    rng = random.Random(seed)
    if cfg.get("shuffle_turns", True):
        rng.shuffle(turns)
    turns = turns[: cfg["n_turns"]]

    # exp1a pilot settings: no thinking params (=> model default: adaptive thinking, display
    # omitted), no caching, only the text is kept in history. Kept as-is for reproducibility.
    client = make_client(model, seed=seed, max_tokens=cfg.get("max_tokens", 600), cache=False)
    system = rule if placement == "system" else cfg.get("base_system")
    messages = []
    run_id = f"{model}|{placement}|{user_lang}|s{seed}"

    for i, user_text in enumerate(turns, start=1):
        if placement == "first_user" and i == 1:
            user_text = f"{rule}\n\n{user_text}"
        messages.append({"role": "user", "content": user_text})
        r = client.chat(system, messages)
        reply, usage = r["text"], {**r["usage"], "stop_reason": r["stop_reason"],
                                   "block_types": r["block_types"]}
        messages.append({"role": "assistant", "content": reply})
        rec = {
            "run_id": run_id, "model": model, "placement": placement,
            "user_lang": user_lang, "seed": seed, "turn": i,
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "stop_reason": usage.get("stop_reason"), "block_types": usage.get("block_types"),
            "compliant": check(reply), "lang": language_profile(reply),
            "user": user_text, "reply": reply,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        if cfg.get("stop_at_first_violation", False) and rec["compliant"] is False:
            break


def run_experiment(cfg: dict, out_path: str, done: set | None = None):
    done = done or set()
    grid = itertools.product(cfg["models"], cfg["rule_placements"],
                             cfg["user_langs"], range(cfg["seeds"]))
    with open(out_path, "a", encoding="utf-8") as out:
        for model, placement, lang, seed in grid:
            rid = f"{model}|{placement}|{lang}|s{seed}"
            if rid in done:
                continue
            print(f"[run] {rid}")
            run_one(cfg, model, placement, lang, seed, out)
