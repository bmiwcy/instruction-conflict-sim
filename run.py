"""CLI.
  python run.py run     configs/<exp>.yaml
  python run.py analyze configs/<exp>.yaml [--axis turn|total_input_tokens]
Runs are resumable: finished run_ids already in the results file are skipped.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from ics.analyze import load, summarize
from ics.cost import record_cost

GROUP_BY = {
    "exp1a": ("model", "placement", "user_lang"),
    "exp1b": ("model", "placement", "thinking_level", "task_kind"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "analyze"])
    p.add_argument("config")
    p.add_argument("--axis", default=None, choices=["turn", "input_tokens", "total_input_tokens"])
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    runner = cfg.get("runner", "exp1a")
    out = Path("results") / f"{cfg['name']}.jsonl"
    out.parent.mkdir(exist_ok=True)

    if runner == "exp1c":
        from ics import exp1c
        if a.cmd == "run":
            done = set()
            if out.exists():
                done = {json.loads(l)["run_id"] for l in open(out, encoding="utf-8")}
            exp1c.run_experiment(cfg, str(out), done)
        else:
            recs = exp1c.load(out)
            for extra in cfg.get("analyze_with", []):  # e.g. a control run stored in another file
                recs += exp1c.load(extra)
            for r in exp1c.table(recs):
                print(json.dumps(r, ensure_ascii=False))
            print(f"runs={len(recs)} cost=${exp1c.cost(recs):.3f}")
            Path("figures").mkdir(exist_ok=True)
            # the per-condition figures are Haiku-only (larger models ran only the 25-step cells)
            for m in ["claude-haiku-5-5"] if any(r["model"] == "claude-haiku-5-5" for r in recs) else []:
                for pth in exp1c.plot(recs, "figures", m):
                    print("->", pth)
            if len({r["model"] for r in recs}) > 1:
                print("->", exp1c.plot_models(recs, "figures"))
        return

    if a.cmd == "run":
        if runner == "exp1b":
            from ics.exp1b import run_experiment
        else:
            from ics.runner import run_experiment
        done = set()
        if out.exists():
            runs = load(out)
            done = {rid for rid, recs in runs.items()
                    if len(recs) >= cfg["n_turns"] or recs[-1]["compliant"] is False}
        run_experiment(cfg, str(out), done)
        return

    axis = a.axis or ("total_input_tokens" if runner == "exp1b" else "turn")
    Path("figures").mkdir(exist_ok=True)
    png = Path("figures") / f"{cfg['name']}_{axis}.png"
    rows = summarize(out, group_by=GROUP_BY[runner], axis=axis, out_png=png)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))

    recs = [r for v in load(out).values() for r in v]
    cost = sum(record_cost(r) for r in recs)
    print(f"records={len(recs)} cost=${cost:.3f}")
    if runner == "exp1b":
        print("violation types:", dict(Counter(r["violation_type"] for r in recs if r["violation_type"])))
        print("no-evidence (empty) replies:", sum(r["compliant"] is None for r in recs))
        print("stop_reasons:", dict(Counter(r["stop_reason"] for r in recs)))
    print(f"-> {png}")


if __name__ == "__main__":
    main()
