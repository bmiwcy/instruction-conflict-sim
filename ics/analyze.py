"""Survival analysis: P(still compliant after x) per condition.

"Event" = first non-compliant turn. Runs that never violate are right-censored
at their last turn, so a 30-turn run with no violation is not counted as failing.
"""
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    runs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            runs[r["run_id"]].append(r)
    for v in runs.values():
        v.sort(key=lambda r: r["turn"])
    return runs


def first_violation(recs, axis):
    for r in recs:
        if r["compliant"] is False:  # None (no prose) is not a violation
            return r[axis], True
    return recs[-1][axis], False  # censored


def kaplan_meier(samples):
    """samples: list of (time, event). Returns step xs, ys."""
    times = sorted({t for t, e in samples if e})
    n_at_risk = len(samples)
    s, xs, ys = 1.0, [0], [1.0]
    for t in times:
        at_risk = sum(1 for ti, _ in samples if ti >= t)
        d = sum(1 for ti, e in samples if ti == t and e)
        s *= 1 - d / at_risk
        xs.append(t); ys.append(s)
    xs.append(max(t for t, _ in samples)); ys.append(s)
    return xs, ys


def summarize(path, group_by=("model", "placement", "user_lang"), axis="turn", out_png=None):
    runs = load(path)
    groups = defaultdict(list)
    for recs in runs.values():
        key = " / ".join(str(recs[0][g]) for g in group_by)
        groups[key].append(first_violation(recs, axis))

    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, samples in sorted(groups.items()):
        xs, ys = kaplan_meier(samples)
        ax.step(xs, ys, where="post", label=f"{key} (n={len(samples)})")
        events = [t for t, e in samples if e]
        rows.append({"condition": key, "runs": len(samples),
                     "violated": len(events),
                     "median_first_violation": sorted(events)[len(events) // 2] if events else None,
                     "survival_at_end": round(ys[-1], 3)})
    ax.set_xlabel("turn" if axis == "turn" else "context size (input tokens)")
    ax.set_ylabel("fraction of runs still compliant")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Instruction survival")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return rows
