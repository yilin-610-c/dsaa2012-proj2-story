# Claude read this — repository overview and next steps

This file is a **handoff summary** for coding agents (and humans): what this repo does today, where the important code lives, and what to do next. For workflow rules see [`AGENTS.md`](AGENTS.md); for collaboration see [`COLLAB.md`](COLLAB.md).

---

## 1. What this repository does

### 1.1 Core story image pipeline (`storygen`)

- **Parse**: Custom story text — `[SEP]`, `[SCENE-n]`, angle-bracket entities `<Name>` — into `Story` / `Scene` ([`src/storygen/parser.py`](src/storygen/parser.py)).
- **Prompts**: Mostly **rule-based** ([`src/storygen/prompt_builder.py`](src/storygen/prompt_builder.py)); optional **LLM-assisted** path ([`src/storygen/prompt_pipelines.py`](src/storygen/prompt_pipelines.py), `llm_assisted` / `api`).
- **Generate**: SDXL + diffusers, profiles in [`configs/base.yaml`](configs/base.yaml); IP-Adapter, Anchor Bank, img2img routing, CLIP reranking, etc.
- **Outputs**: Under `outputs/<run_name>/` — resolved config, manifest, per-scene `prompt.json`, `selected.png`, logs.

### 1.2 Auto-routing scripts

- **[`scripts/run_auto_story_pipeline.py`](scripts/run_auto_story_pipeline.py)**  
  Classifies story by **count of distinct `<Name>`** across scenes: **single** → `storygen.cli` (default conda `ipadapter`; default `--set` includes `generation.identity_conditioning.scale=0.3`); **double** → StoryDiffusion probe via `run_test_set.py` + `run_probe.py`.

- **[`scripts/run_auto_story_pipeline_modular.py`](scripts/run_auto_story_pipeline_modular.py)**  
  Same routing; **single** path forces modular rule prompt + SDXL backend via `--set`; **double** path passes `--prompt-builder modular --prompt-modular-backend storydiffusion` to `run_test_set.py`.

### 1.3 StoryDiffusion Gradio probe (upstream comparison)

- **[`storydiffusion_gradio_probe/run_probe.py`](storydiffusion_gradio_probe/run_probe.py)** — Calls upstream StoryDiffusion without launching Gradio.
- **[`storydiffusion_gradio_probe/run_test_set.py`](storydiffusion_gradio_probe/run_test_set.py)** — Batch-converts `test_set/*.txt` to probe YAML; supports **`--prompt-builder legacy|modular`**, **`--prompt-modular-backend sdxl|storydiffusion`**, **`--prompt-template-pack`**.

### 1.4 Modular prompt stack (opt-in)

- **Package**: [`src/storygen/prompt_stack/`](src/storygen/prompt_stack/) — `ModularPromptBuilder`, `build_rule_prompt_builder`, template pack merge, StoryDiffusion **post-filter** (strip SDXL-only spatial hacks for probe).
- **Config**: `prompt.builder: legacy | modular`, `prompt.modular.backend`, `prompt.modular.template_pack`; template YAML under [`configs/prompt_templates/`](configs/prompt_templates/) (e.g. `default.yaml`).
- **Wiring**: [`src/storygen/prompt_pipelines.py`](src/storygen/prompt_pipelines.py) `RuleBasedPromptPipeline` uses `build_rule_prompt_builder()` — **default remains legacy**.

### 1.5 Recent prompt / quality fixes (rule-based)

- **Vehicle + door/window**: Avoid treating phrases like “at the door” as a **new scene setting** when `vehicle_context` applies; align with bus/train door semantics where relevant.
- **Truncation**: Higher default budgets for `generation_prompt` / scene-consistency snippet in config; **`generation_trim_truncation_artifacts`** trims broken tails (e.g. trailing `same vehicle context:`) after shortening.
- **Optional “cinematography” clause**: [`configs/base.yaml`](configs/base.yaml) → `prompt.cinematography` (`enabled: false` by default) — short lighting / shot phrases appended to `generation_prompt` when keywords match.
- **Optional dynamic negative**: `prompt.dynamic_negative` (`enabled: false` by default) — appends extra negative tokens from simple heuristics (night / single-subject).
- **Bugfix**: Removed duplicate `_infer_vehicle_context` in `prompt_builder.py`.

### 1.6 DiT (optional, not part of main story CLI)

- **Submodule**: [`third_party/facebookresearch-DiT`](third_party/facebookresearch-DiT) (class-conditional ImageNet; CC-BY-NC).
- **Docs**: [`docs/dit_smoke.md`](docs/dit_smoke.md), [`docs/dit_report_framing.md`](docs/dit_report_framing.md).
- **Script**: [`scripts/dit_smoke.sh`](scripts/dit_smoke.sh) — isolated conda + `sample.py`; not wired to `storygen`.

### 1.7 House rules and docs

- **[`AGENTS.md`](AGENTS.md)** — Checkpoints, scope, docs when behavior changes.
- **[`COLLAB.md`](COLLAB.md)** — Code map, interfaces, modular prompt notes.
- **[`README.md`](README.md)** — Setup, run commands, probe + modular pointers.

---

## 2. Quick commands (cheat sheet)

```bash
# Rule-based smoke (legacy builder unless profile sets modular)
cd /path/to/dsaa2012-proj2-story
PYTHONPATH=src python3 -m storygen.cli --profile smoke_test --input test_set/01.txt

# Auto-router (legacy prompt on single path unless you override)
python3 scripts/run_auto_story_pipeline.py --input test_set/07.txt --run-name my_run

# Auto-router + modular stack (single: modular SDXL; double: modular + storydiffusion sanitizer)
python3 scripts/run_auto_story_pipeline_modular.py --input test_set/07.txt --run-name my_run_modular

# Probe batch with modular StoryDiffusion-friendly prompts
conda activate storydiffusion  # or your env name
python storydiffusion_gradio_probe/run_test_set.py \
  --prompt-builder modular --prompt-modular-backend storydiffusion \
  --input-dir test_set --glob '*.txt' --limit 1 --run
```

Enable optional cinema / dynamic negative in YAML or via `--set prompt.cinematography.enabled=true` etc. (see `configs/base.yaml`).

---

## 3. Suggested next steps

1. **Run a documented A/B**  
   Same stories (`test_set` / `test_setA`): `legacy` vs `modular` (SDXL + StoryDiffusion paths); save `prompt.json` diffs and a short qualitative note (report or `docs/` experiment log).

2. **Tune `cinematography` / `dynamic_negative`**  
   Turn `enabled: true` on one profile; refine keyword lists so they do not fight `style_prompt` or duplicate “cinematic” phrases.

3. **Refactor phase 2 for `prompt_stack`**  
   Move cinema / dynamic-negative / vehicle semantics into named **modifiers** + thin renderer; keep `PromptBuilder` as the single rule engine until parity is proven, or delegate scene-by-scene from modifiers only where safe.

4. **Layout / regional conditioning (separate project)**  
   If needed: define a **layout JSON** in `PromptSpec.metadata` or parallel artifact, then wire **Regional/GLIGEN/ControlNet** in `generators/` — do not overload text-only prompt hacks for this.

5. **CI**  
   Add `pytest` (or document `python -c` smoke) for [`tests/test_prompt_builder.py`](tests/test_prompt_builder.py) and [`tests/test_prompt_stack.py`](tests/test_prompt_stack.py); clarify whether collaborators must `git submodule update --init` for DiT.

---

## 4. Files to read first when touching prompts

| Area | File(s) |
|------|---------|
| Rule prompts | [`src/storygen/prompt_builder.py`](src/storygen/prompt_builder.py) |
| Pipeline selection | [`src/storygen/prompt_pipelines.py`](src/storygen/prompt_pipelines.py) |
| Modular + factory | [`src/storygen/prompt_stack/facade.py`](src/storygen/prompt_stack/facade.py), [`factory.py`](src/storygen/prompt_stack/factory.py) |
| StoryDiffusion sanitizer | [`src/storygen/prompt_stack/renderers/storydiffusion.py`](src/storygen/prompt_stack/renderers/storydiffusion.py) |
| Defaults / toggles | [`configs/base.yaml`](configs/base.yaml), [`configs/prompt_templates/default.yaml`](configs/prompt_templates/default.yaml) |
| Probe CLI | [`storydiffusion_gradio_probe/run_test_set.py`](storydiffusion_gradio_probe/run_test_set.py) |

---

*Last updated to reflect modular prompt stack, prompt fixes (vehicle/door, truncation, cinema/dynamic negative), auto-router scripts, and DiT optional submodule. Update this file when you land major behavior or routing changes.*
