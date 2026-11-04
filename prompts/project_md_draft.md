# SynthLab — CLAUDE.md

<!-- DRAFT for review. {{RULE}} marks the line where the language rule is inserted
     (md condition) or removed (system condition). Keep it on its own line. -->

SynthLab generates synthetic training images in Unreal Engine 5 and trains a semantic
segmentation model on them. The goal is to detect thin road markings (lane lines,
stop lines, crosswalk stripes) in street scenes where real labelled data is scarce.

## Repository layout

```
synthlab/
├── ue5/                      # UE5 project (C++ + Blueprints)
│   ├── Source/SynthLab/
│   │   ├── CaptureActor.cpp      # multi-camera capture, writes RGB + label passes
│   │   ├── SceneRandomizer.cpp   # lighting, weather, asset placement
│   │   └── LabelPass.cpp         # stencil -> class-id mapping
│   └── Config/Randomization.json
├── pipeline/                 # Python, runs outside UE5
│   ├── export/               # converts raw captures to the dataset format
│   │   ├── converter.py
│   │   └── manifest.py
│   ├── dataset/
│   │   ├── synth_dataset.py      # torch Dataset, reads manifest.jsonl
│   │   └── transforms.py
│   ├── train/
│   │   ├── train.py              # entry point (hydra config)
│   │   ├── model.py              # mask-transformer segmentation model
│   │   └── losses.py
│   └── eval/
│       ├── metrics.py            # mIoU, boundary F1, thin-structure recall
│       └── compare_real.py       # sim-vs-real gap report
├── configs/                  # hydra configs
├── tools/                    # one-off scripts, not imported by the pipeline
└── tests/
```

## How to run

- Capture: open `ue5/SynthLab.uproject`, load `Maps/Capture_Main`, run the `BP_CaptureDirector`
  in PIE. Output goes to `/data/synth_out/<date>/raw/`.
- Export: `python -m pipeline.export.converter --raw /data/synth_out/<date>/raw --out data/<date>`
- Train: `python -m pipeline.train.train dataset=<date> model=masktf_small`
- Eval: `python -m pipeline.eval.compare_real ckpt=<path> real=data/real_v3`
- Tests: `pytest -q tests/` (must pass before any commit)

## Coding conventions

- Python 3.11, type hints on all public functions, `ruff` + `black` (line length 100).
- No new dependencies without asking first. Allowed: torch, numpy, opencv, hydra, pydantic.
- Paths: always `pathlib.Path`, never string concatenation. Never hard-code absolute paths in library code.
- Logging via `logging.getLogger(__name__)`; no `print` in `pipeline/`.
- Class ids are defined ONLY in `pipeline/dataset/classes.py`; UE5 stencil values must match it.
- Keep functions under ~60 lines; split instead of adding flags.
{{RULE}}
- When changing the dataset format, bump `MANIFEST_VERSION` in `manifest.py` and add a
  migration note to `docs/dataset_changes.md`.
- C++: Unreal naming conventions (F/U/A prefixes), no raw `new`, prefer `TObjectPtr`.

## Dataset format

Each sample is one line in `manifest.jsonl`:

```json
{"id": "000123", "rgb": "rgb/000123.png", "label": "label/000123.png",
 "camera": "cam_02", "weather": "rain", "sun_elevation": 23.5, "version": 4}
```

Label PNGs are single-channel uint8 class ids. 255 = ignore. Images are 1920x1080; training
uses random 1024x1024 crops.

## Known issues

- Rain particles sometimes leak into the label pass as class 0 (background) instead of ignore.
- `compare_real.py` is slow (> 20 min) because it recomputes features for the real set each run.
- Lane lines thinner than 2 px at 1080p are frequently lost after the 0.5x eval resize.
- Windows path length limits break export when the date folder name is long.

## Workflow rules

- Small, reviewable changes. Explain the reason for a change before showing the diff.
- If a request conflicts with something in this file, point out the conflict before acting.
- Don't touch `ue5/Config/Randomization.json` without asking; it is shared with another team.
- When unsure about the expected behaviour, ask instead of guessing.
