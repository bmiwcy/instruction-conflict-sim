"""Experiment 1c - does a run of tool calls pull the reply language to English?

The user writes only in Chinese; the earlier assistant replies are Chinese (fixed history).
After the user's "好的！请优化！", N scripted tool steps (assistant tool_use + user tool_result,
realistic English output) are inserted. Then:

  call 0  the model responds. Its first text block is the "first text".
          If it ends its turn without tools, that response is also the final report.
  call 1  otherwise its tool calls get one normal, non-contradicting result each, and the final
          report is requested with tool_choice=none (no extra words are added, so nothing nudges
          the language). An empty report is retried up to `empty_retries` times.

Condition = model x rule x audience x n_tools x seed.
  rule      none   : project md present, rule line removed (no language rule anywhere)
            md     : rule is one line in the project md (sent in the first user message)
            system : rule appended to the agent system prompt (md present, rule line removed)
            no_md  : no project md at all and no rule (closest to the real case)
            md_in_system : same project md (rule line removed) but placed in the system prompt
                     instead of the first user message; no rule. Control for `none`.
  audience  no / yes : whether the agent system prompt says that text between tool calls is
            summarized for the user rather than shown verbatim
The English agent system prompt is present in every arm.
"""
import itertools
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from .client import make_client, thinking_params
from .detect import CHECKS, language_profile
from .exp1b import RULE_MARKER, RULE_MENTION

TOOLS = [
    {"name": "bash", "description": "Run a shell command in the repository root and return its output.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read a file from the repository.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                      "required": ["path"]}},
    {"name": "edit_file", "description": "Replace an exact string in a file.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
         "required": ["path", "old", "new"]}},
]


def load_scenario(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _aud(a) -> str:
    return "yes" if a in (True, "yes") else "no"  # YAML turns yes/no into booleans


def build_context(sc: dict, rule_cond: str, rule: dict, audience="no"):
    system = sc["agent_system"].strip()
    if _aud(audience) == "yes":
        system += "\n\n" + sc["audience_sentence"].strip()
    if rule_cond == "system":
        system += "\n\n" + rule["system_text"]
    hist = [dict(m) for m in sc["history"]]
    if rule_cond == "md_in_system":
        md = "\n".join(l for l in sc["project_md"].splitlines() if RULE_MARKER not in l)
        system += ("\n\n<project_instructions file=\"CLAUDE.md\">\n" + md.strip()
                   + "\n</project_instructions>")
    elif rule_cond != "no_md":
        md = sc["project_md"]
        if rule_cond == "md":
            md = md.replace(RULE_MARKER, rule["md_line"])
        else:
            md = "\n".join(l for l in md.splitlines() if RULE_MARKER not in l)
        hist[0]["content"] = ("<project_instructions file=\"CLAUDE.md\">\n" + md.strip()
                              + "\n</project_instructions>\n\n" + hist[0]["content"])
    return system, hist


def scripted_steps(sc: dict, n: int):
    msgs = []
    for i, st in enumerate(sc["steps"][:n]):
        tid = f"toolu_sim_{i:03d}"
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": st["tool"], "input": st["input"]}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": st["output"]}]})
    return msgs


def normal_result(sc: dict, use: dict) -> str:
    no = sc["normal_outputs"]
    if use["name"] == "read_file":
        path = str(use["input"].get("path", "")).lstrip("./")
        for st in sc["steps"]:
            if st["tool"] == "read_file" and st["input"].get("path") == path:
                return st["output"]
        return no["read_file_unknown"]
    return no.get(use["name"], no["bash"])


def _text_entries(content, call, check):
    return [{"call": call, "text": b["text"], "compliant": check(b["text"]),
             "lang": language_profile(b["text"])}
            for b in content if b["type"] == "text" and b["text"].strip()]


def run_one(cfg, sc, model, rule_cond, audience, n, seed):
    params = thinking_params(model, cfg.get("thinking_level"))
    check = CHECKS[cfg["rule"]["check"]]
    system, messages = build_context(sc, rule_cond, cfg["rule"], audience)
    messages += scripted_steps(sc, n)
    client = make_client(model, seed=seed, max_tokens=cfg.get("max_tokens", 32000),
                         cache=cfg.get("cache", True))
    usage_sum, thinking_all = {}, []

    def call(msgs, extra):
        r = client.chat(system, msgs, {**params, **extra})
        for k, v in r["usage"].items():
            usage_sum[k] = usage_sum.get(k, 0) + (v or 0)
        if r["thinking"]:
            thinking_all.append(r["thinking"])
        return r

    r0 = call(messages, {"tools": TOOLS})
    texts = _text_entries(r0["content"], 0, check)
    uses = [b for b in r0["content"] if b["type"] == "tool_use"]
    report, report_blocks, retries = r0, r0["block_types"], 0
    if r0["stop_reason"] == "tool_use" and uses:
        msgs = messages + [
            {"role": "assistant", "content": r0["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"],
                                          "content": normal_result(sc, u)} for u in uses]}]
        for retries in range(cfg.get("empty_retries", 2) + 1):
            report = call(msgs, {"tools": TOOLS, "tool_choice": {"type": "none"}})
            report_blocks = report["block_types"]
            if report["text"].strip():
                break
        texts += _text_entries(report["content"], 1, check)

    first = texts[0] if texts else None
    final_text = report["text"]
    thinking = "\n".join(thinking_all)
    return {
        "run_id": f"{model}|{rule_cond}|aud-{_aud(audience)}|n{n}|s{seed}",
        "model": model, "rule_cond": rule_cond, "audience": _aud(audience), "n_tools": n, "seed": seed,
        "first_text": first["text"] if first else "",
        "first_compliant": first["compliant"] if first else None,
        "first_at_call": first["call"] if first else None,
        "first_response_tool_calls": [{"name": u["name"], "input": u["input"]} for u in uses],
        "final_text": final_text, "final_compliant": check(final_text),
        "final_stop_reason": report["stop_reason"], "final_block_types": report_blocks,
        "report_requested": report is not r0, "empty_retries_used": retries,
        "texts": texts, "thinking": thinking,
        "thinking_mentions_rule": bool(RULE_MENTION.search(thinking)) if thinking else None,
        "thinking_lang": language_profile(thinking) if thinking else None,
        **{f"sum_{k}": v for k, v in usage_sum.items()},
    }


def plan(cfg):
    seeds_by_n = {int(k): v for k, v in cfg["seeds_by_n"].items()}
    for model, rc, aud, n in itertools.product(cfg["models"], cfg["rule_conds"],
                                               cfg["audience"], cfg["n_tools"]):
        for seed in range(seeds_by_n[n]):
            yield model, rc, _aud(aud), n, seed


def run_experiment(cfg, out_path, done=None):
    done = done or set()
    sc = load_scenario(cfg["scenario"])
    jobs = [j for j in plan(cfg) if f"{j[0]}|{j[1]}|aud-{j[2]}|n{j[3]}|s{j[4]}" not in done]
    print(f"{len(jobs)} runs to do", flush=True)
    lock = threading.Lock()
    tag = {True: "zh", False: "EN", None: "--"}
    with open(out_path, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(cfg.get("workers", 4)) as ex:
        futs = {ex.submit(run_one, cfg, sc, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                rec = f.result()
            except Exception as e:  # keep going; the run is retried on the next invocation
                print(f"[{i}/{len(jobs)}] ERROR {j}: {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            print(f"[{i}/{len(jobs)}] {rec['run_id']} first={tag[rec['first_compliant']]} "
                  f"final={tag[rec['final_compliant']]}", flush=True)


# ---------- analysis ----------

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def table(recs, keys=("model", "rule_cond", "audience", "n_tools")):
    from collections import defaultdict
    g = defaultdict(list)
    for r in recs:
        g[tuple(r[k] for k in keys)].append(r)
    rows = []
    for k, rs in sorted(g.items()):
        valid_first = [r for r in rs if r["first_compliant"] is not None]
        valid_final = [r for r in rs if r["final_compliant"] is not None]
        kf = sum(r["first_compliant"] is False for r in valid_first)
        kr = sum(r["final_compliant"] is False for r in valid_final)
        p, lo, hi = wilson(kf, len(valid_first))
        rows.append({**dict(zip(keys, k)), "runs": len(rs),
                     "first_drift": f"{kf}/{len(valid_first)}", "first_rate": round(p, 2),
                     "ci95": [round(lo, 2), round(hi, 2)],
                     "final_drift": f"{kr}/{len(valid_final)}",
                     "empty_final": len(rs) - len(valid_final),
                     "first_drift_thinking_mentions_rule":
                         sum(r["first_compliant"] is False and bool(r["thinking_mentions_rule"]) for r in rs)})
    return rows


def cost(recs):
    from .cost import record_cost
    return sum(record_cost({"model": r["model"],
                            "input_tokens": r.get("sum_input_tokens", 0),
                            "cache_creation_input_tokens": r.get("sum_cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": r.get("sum_cache_read_input_tokens", 0),
                            "output_tokens": r.get("sum_output_tokens", 0),
                            "total_input_tokens": 0}) for r in recs)


# validated default categorical palette (dataviz skill, light surface), fixed order
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
RULE_STYLE = {  # color follows the entity; marker is the secondary encoding
    "no_md":  ("#2a78d6", "o", "no md, no rule"),
    "none":   ("#eb6834", "s", "md, no rule"),
    "md":     ("#1baf7a", "^", "rule in md"),
    "system": ("#eda100", "D", "rule in system prompt"),
    "md_in_system": ("#e87ba4", "v", "md in system prompt, no rule"),
}
AUD_STYLE = {"no": ("#2a78d6", "o", "without audience sentence"),
             "yes": ("#eb6834", "s", "with audience sentence")}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.03, 1.03)


def _rate_series(recs, ns):
    xs, ps, los, his, labels = [], [], [], [], []
    for n in ns:
        rs = [r for r in recs if r["n_tools"] == n and r["first_compliant"] is not None]
        if not rs:
            continue
        k = sum(r["first_compliant"] is False for r in rs)
        p, lo, hi = wilson(k, len(rs))
        xs.append(n); ps.append(p); los.append(p - lo); his.append(hi - p); labels.append(f"{k}/{len(rs)}")
    return xs, ps, los, his, labels


def plot(recs, out_dir, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [r for r in recs if r["model"] == model]
    ns = sorted({r["n_tools"] for r in recs})
    pos = {n: i for i, n in enumerate(ns)}  # categorical x so 0/5/10/25 are evenly spaced
    paths = []

    # Figure 1: drift rate by rule condition (audience = no)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    _style_axes(ax)
    rules = [r for r in RULE_STYLE if any(x["rule_cond"] == r for x in recs)]
    off = {r: (i - (len(rules) - 1) / 2) * 0.07 for i, r in enumerate(rules)}
    for rc in rules:
        color, marker, label = RULE_STYLE[rc]
        xs, ps, lo, hi, lab = _rate_series([r for r in recs if r["rule_cond"] == rc and r["audience"] == "no"], ns)
        if not xs:
            continue
        x = [pos[v] + off[rc] for v in xs]
        ax.errorbar(x, ps, yerr=[lo, hi], color=color, marker=marker, markersize=7, linewidth=2,
                    capsize=3, elinewidth=1.2, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlim(-0.4, len(ns) - 0.6)
    ax.set_xlabel("scripted tool calls before the model speaks", color=INK, fontsize=10)
    ax.set_ylabel("share of runs whose first text is not Chinese", color=INK, fontsize=10)
    ax.set_title(f"Language drift after tool calls ({model}, no audience sentence, 95% Wilson CI)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    p1 = Path(out_dir) / "exp1c_drift_by_rule.png"
    fig.savefig(p1, dpi=200, facecolor=SURFACE)
    paths.append(p1)

    # Figure 2: audience sentence yes/no, one panel per rule condition
    fig, axs = plt.subplots(1, len(rules), figsize=(3.0 * len(rules), 3.6), sharey=True, facecolor=SURFACE)
    axs = [axs] if len(rules) == 1 else axs
    for ax, rc in zip(axs, rules):
        _style_axes(ax)
        for j, aud in enumerate(("no", "yes")):
            color, marker, label = AUD_STYLE[aud]
            xs, ps, lo, hi, lab = _rate_series([r for r in recs if r["rule_cond"] == rc and r["audience"] == aud], ns)
            if not xs:
                continue
            x = [pos[v] + (j - 0.5) * 0.12 for v in xs]
            ax.errorbar(x, ps, yerr=[lo, hi], color=color, marker=marker, markersize=6, linewidth=2,
                        capsize=3, elinewidth=1.1, markeredgecolor=SURFACE, markeredgewidth=1.3, label=label)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_title(RULE_STYLE[rc][2], color=INK, fontsize=9.5)
        ax.set_xlabel("tool calls", color=INK, fontsize=9)
    axs[0].set_ylabel("first text not Chinese", color=INK, fontsize=9.5)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.suptitle(f"Does saying 'in-between text is summarized' change the language? ({model})",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    p2 = Path(out_dir) / "exp1c_audience.png"
    fig.savefig(p2, dpi=200, facecolor=SURFACE)
    paths.append(p2)
    return paths


MODEL_ORDER = [("claude-haiku-5-5", "Haiku 5.5"), ("claude-sonnet-5-5", "Sonnet 5.5"),
               ("claude-opus-5-5", "Opus 5.5"), ("claude-fable-5-1", "Fable 5.1")]
MODEL_CELLS = [  # (rule_cond, label, color) - categorical slots 2, 5, 3 match the other figure
    ("none", "md in user message, no rule", "#eb6834"),
    ("md_in_system", "md in system prompt, no rule", "#e87ba4"),
    ("md", "rule in md (md in user message)", "#1baf7a"),
]


def plot_models(recs, out_dir, n_tools=25, audience="no"):
    """Grouped bars: x = model, one bar per condition, 95% Wilson interval, k/n above each bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [(m, lab) for m, lab in MODEL_ORDER if any(r["model"] == m for r in recs)]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), facecolor=SURFACE)
    _style_axes(ax)
    w = 0.26
    for j, (rc, label, color) in enumerate(MODEL_CELLS):
        for i, (m, _) in enumerate(models):
            all_rs = [r for r in recs if r["model"] == m and r["rule_cond"] == rc and r["n_tools"] == n_tools
                      and r["audience"] == audience]
            rs = [r for r in all_rs if r["first_compliant"] is not None]
            x = i + (j - 1) * (w + 0.02)
            if not all_rs:
                ax.text(x, 0.02, "not run", rotation=90, ha="center", va="bottom", fontsize=7.5, color=MUTED)
                continue
            no_text = len(all_rs) - len(rs)
            if not rs:
                ax.plot([x - w / 2, x + w / 2], [0.004, 0.004], color=color, linewidth=3, solid_capstyle="butt")
                ax.text(x, 0.03, f"no text\n{no_text}/{len(all_rs)}", ha="center", va="bottom", fontsize=7.5, color=INK)
                continue
            k = sum(r["first_compliant"] is False for r in rs)
            p, lo, hi = wilson(k, len(rs))
            if p > 0:
                ax.bar(x, p, width=w, color=color, edgecolor=SURFACE, linewidth=2)
            else:  # a zero still shows which condition it is
                ax.plot([x - w / 2, x + w / 2], [0.004, 0.004], color=color, linewidth=3, solid_capstyle="butt")
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], color=INK, capsize=3, elinewidth=1, linewidth=0)
            txt = f"{k}/{len(rs)}" + (f"\n(+{no_text} no text)" if no_text else "")
            ax.text(x, hi + 0.02, txt, ha="center", va="bottom", fontsize=7.5, color=INK, linespacing=1.1)
    ax.set_xticks(range(len(models)), [lab for _, lab in models])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("share of runs whose first text is not Chinese", color=INK, fontsize=10)
    ax.set_title(f"Drift after {n_tools} tool steps, by model (95% Wilson CI)", color=INK, fontsize=11, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in MODEL_CELLS], frameon=False, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, labelcolor=INK)
    fig.tight_layout()
    path = Path(out_dir) / "exp1c_by_model.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    return path
