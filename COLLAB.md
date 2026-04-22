# Collaboration Guide

This repo is built on top of the current `storygen` baseline. The goal is to evolve the existing pipeline into a stronger, modular, reproducible system for multi-panel story image generation, not to replace it with a separate codebase.

## Current Objective

The current project goal is to improve the baseline along:
- per-panel correctness
- cross-panel consistency
- narrative continuity
- style consistency
- overall image quality

Engineering goals:
- keep the pipeline fixed and automatic
- keep runs reproducible and inspectable
- preserve modular boundaries so collaborators can work in parallel
- support quick controlled experiments on dev/Test-A cases
- preserve a clean baseline path for comparison and reporting

Non-goals for the current repo stage:
- full repo rewrite
- large custom training stack first
- paper-faithful reimplementation at the cost of engineering stability
- large academic ablation infrastructure

## Team Split

### Collaborator A: orchestration / prompting / evaluation
- pipeline orchestration
- parsing and prompt pipeline upgrades
- future `SceneSpec` design
- scoring / reranking / evaluation
- dev subset workflow and controlled comparisons
- run scripts, output organization, reproducibility docs

### Collaborator B: generation / consistency mechanisms
- generation-module improvements
- reference-conditioned generation
- identity and style consistency mechanisms
- anchor and reference integration
- img2img and continuity-oriented generation routes
- StoryDiffusion-like generation experiments when they fit the repo

### Shared ownership
- config shape
- output naming and logging conventions
- shared dataclass and interface fields
- experiment naming and comparison setup
- final integration and submission path

Rules:
- Do not silently rename shared fields, config keys, CLI flags, or output files.
- Do not remove an existing runnable path without replacing and documenting it.
- Any shared-interface change must be reflected in code and in this document.

## Current Code Map

Current working layout:
- `src/storygen/parser.py`: scene text parsing
- `src/storygen/types.py`: active internal contracts
- `src/storygen/prompt_builder.py`: rule-based prompt construction
- `src/storygen/prompt_pipelines.py`: prompt pipeline selector and placeholders
- `src/storygen/generators/`: generation backends and wrappers
- `src/storygen/scoring/`: scoring and reranking backends
- `src/storygen/pipeline.py`: end-to-end orchestration
- `src/storygen/cli.py`: stable CLI entrypoint
- `configs/base.yaml`: main config and runtime profiles
- `scripts/run_demo.py`: demo entrypoint
- `outputs/<run_name>/...`: run artifacts

Current active internal contracts are still `Scene`, `Story`, and `PromptSpec`. The repo now also has minimal extensibility contracts for `PromptBundle`, `StoryPromptSpec`, `StoryGenerationRequest`, and `StoryGenerationResult`, but the live baseline still runs through scene-level `PromptSpec` generation.

## Active Interfaces

### `Scene` and `Story`

Current source of truth: `src/storygen/types.py`

Current `Scene` fields:
- `scene_id`
- `index`
- `raw_text`
- `clean_text`
- `entities`

Current `Story` fields:
- `source_path`
- `raw_text`
- `scenes`
- `all_entities`
- `recurring_entities`
- `entity_to_scene_ids`

These are already shared pipeline interfaces. Do not rename or remove fields casually.

### `PromptSpec`

Current source of truth: `src/storygen/types.py`

Current `PromptSpec` fields:
- `style_prompt`
- `character_prompt`
- `global_context_prompt`
- `local_prompt`
- `action_prompt`
- `generation_prompt`
- `scoring_prompt`
- `full_prompt`
- `negative_prompt`

Intended meaning:
- `full_prompt`: rich human-inspectable prompt for debugging and metadata
- `generation_prompt`: shorter prompt actually passed into the text-to-image generator
- `scoring_prompt`: short prompt used for CLIP text-image scoring
- `action_prompt`: short action-only prompt used for action-sensitive scoring

Any change to these semantics must be coordinated because prompting, generation, scoring, and output inspection all depend on them.

### Generator and scorer contracts

Current source of truth:
- `src/storygen/generators/base.py`
- `src/storygen/scoring/base.py`

Scene-level generator contract:
- consumes `GenerationRequest`
- returns `GenerationCandidate`

Story-level generator contract:
- consumes `StoryGenerationRequest`
- returns `StoryGenerationResult`

Implemented generation backends:
- `diffusers_text2img`: scene-level baseline backend
- `storydiffusion_direct`: story-level placeholder only; it raises `NotImplementedError`

Do not implement StoryDiffusion by changing `diffusers_text2img` or by treating StoryDiffusion as only another `model_id`. StoryDiffusion should be implemented behind `storydiffusion_direct` because its execution unit is the whole story, not one scene candidate.

Scorer contract:
- consumes `Story`, `Scene`, `PromptSpec`, `GenerationCandidate`, and previous selected results
- returns structured `CandidateScore`
- selects one `SceneSelectionResult`

This contract should stay stable while generation/scoring implementations change underneath.

### Prompt pipeline contracts

Current source of truth: `src/storygen/prompt_pipelines.py`

Implemented prompt pipelines:
- `rule_based`: wraps the existing `PromptBuilder` and preserves baseline behavior
- `llm_assisted`: OpenAI-backed structured prompt planning with validation, local cache, shareable artifacts, and rule-based fallback
- `api`: compatibility alias for `llm_assisted`

`prompt.pipeline` is the config switch. LLM-assisted prompting must keep the downstream `PromptSpec` contract unchanged and must not replace the rule-based baseline.

## Planned Stable Interfaces

These are collaboration targets, not fully implemented repo-wide contracts yet.

### `SceneSpec` (planned)

Purpose:
- structured scene representation before generation

Intended minimum fields:
- `scene_id`
- `raw_text`
- `characters`
- `objects`
- `setting`
- `action`
- `style`
- `continuity_tags`

Possible extensions:
- `camera` or `shot_type`
- `mood`
- `scene_transition_flags`
- `recommended_generation_mode`

Current mapping:
- parts of this are currently spread across `Scene`, `Story`, and prompt-building logic

### `AnchorSpec` (planned)

Purpose:
- cross-panel consistency anchors for character/object/style/background identity

Intended minimum fields:
- `anchor_id`
- `type`
- `reference_paths`
- `must_keep`
- `optional_keep`

Possible extensions:
- `adapter_scale`
- `scope`
- `linked_scene_ids`

Current mapping:
- not yet an active dataclass
- future generation-side work should introduce it without breaking the baseline interfaces

### `ScoringSpec` (planned)

Purpose:
- declarative scoring and reranking definition

Intended minimum fields:
- `scene_id`
- scoring weights
- must-have checks
- consistency targets
- selected metrics

Possible dimensions:
- text alignment
- character consistency
- object continuity
- background or style consistency
- image quality
- narrative continuity

Current mapping:
- parts of this currently live in `configs/base.yaml`, `PromptSpec`, and scorer implementations

## Run Expectations

Stable entrypoints already in the repo:

```bash
PYTHONPATH=src python3 -m storygen --config configs/base.yaml --profile smoke_test --input test_set/01.txt
PYTHONPATH=src python3 -m storygen --config configs/base.yaml --profile demo_run --input test_set/01.txt
PYTHONPATH=src python3 scripts/run_demo.py --profile demo_run --input test_set/01.txt
```

Profiles are dynamic and validated from `configs/base.yaml`; the CLI does not hard-code the profile list.

Current base profiles:
- `smoke_test`: cheap scene-level baseline
- `demo_run`: fuller scene-level baseline
- `cloud_strong_backbone`: scene-level diffusers profile reserved for stronger cloud backbones
- `cloud_storydiffusion`: story-level StoryDiffusion placeholder; currently not runnable for real generation
- `llm_prompt_smoke`: optional LLM-assisted prompt planning profile

Expected comparison workflow:
- baseline path
- prompt-improved path
- anchor or reference-conditioned path
- stronger combined path
- quick dev subset path

Rule:
- major experiment choices should be controlled through config or CLI, not by editing code for each run

Current config pattern already supports this style. Prefer adding switches under existing YAML sections, for example:
- `prompt.rewriter.type`
- future `generation.mode`
- future `anchor.enabled`
- future `reference_conditioning.enabled`
- future `scoring.type`

## Output and Logging Conventions

Current source of truth: `src/storygen/pipeline.py` and `src/storygen/io/results.py`

Current run layout:

```text
outputs/<run_name>/
  config_resolved.yaml
  manifest.json
  run_summary.json
  logs/
    prompt_pipeline.json
    generation_backend.json
    events.jsonl
  scenes/
    scene_001/
      prompt.json
      scene_result.json
      selected.png
      candidates/
        cand_000_seed_....png
```

Required collaboration rule:
- do not silently change output naming or placement
- if output structure must change, keep backward compatibility where feasible and update this document

Each run should continue to preserve:
- resolved config
- prompt pipeline metadata
- generation backend metadata
- event log
- prompt artifacts
- candidate images
- selected images
- scores and selection metadata
- summary metadata for reproducibility

## Current Extensibility Status

Implemented:
- dynamic runtime profiles from YAML
- `prompt.pipeline=rule_based`
- `prompt.pipeline=llm_assisted`
- `prompt.pipeline=api` alias for `llm_assisted`
- `model.backend=diffusers_text2img` with `model.granularity=scene`
- `model.backend=storydiffusion_direct` placeholder with `model.granularity=story`
- minimal run logs under `outputs/<run_name>/logs/`
- local prompt cache under `.cache/prompt_builder/`
- shareable prompt artifacts under `prompt_artifacts/`

Not implemented:
- real StoryDiffusion generation
- training-free attention/reference/latent components

The current baseline behavior should remain unchanged for `smoke_test`, `demo_run`, and any profile using `diffusers_text2img + scene`.

### LLM Prompt Sharing Rules

- `.cache/prompt_builder/` is local acceleration state and should not be committed.
- `prompt_artifacts/` is the collaboration surface for reviewed LLM prompt outputs.
- To share generated prompts with teammates, export an artifact and commit or send that JSON.
- To reuse teammate prompts, set `prompt.artifact.path`; this bypasses API calls.
- API keys must come from environment variables and must never be written into configs, cache records, artifacts, or docs.
- LLM-assisted prompting is for structured extraction and short prompt rewriting only; it does not own image generation or scoring.

## Controlled Experiment Expectations

We do not need a large formal ablation framework right now.

We do need a few clean, repeatable comparisons such as:
- baseline
- baseline + prompt improvements
- baseline + anchor or reference conditioning
- baseline + anchor/reference + rerank

That means:
- baseline behavior must remain runnable
- improved variants should be toggled through config where possible
- collaborators should avoid hard-coded one-off experiment branches inside shared modules

## Shared Code Rules

When modifying shared code:
- prefer additive and modular changes
- preserve working baseline behavior unless intentionally upgraded
- avoid broad repo refactors unless they remove a concrete blocker
- keep interface contracts explicit
- update docs whenever shared behavior changes
- preserve the current config-driven execution model

When changing shared interfaces:
- document the change here
- keep backward compatibility when practical
- make downstream modules still work before merging related changes

## Coding-Agent Rules

When using coding agents on this repo:
- prioritize modular, documented changes
- prefer config-driven toggles over hard-coded branches
- preserve runnable baseline paths
- avoid silent interface drift
- avoid introducing external API dependencies
- avoid agent-based runtime logic
- keep the pipeline automatic and reproducible

If changing a shared contract:
- update code
- update docs
- verify affected entrypoints still run

## Resource Tracking

This project may use public pretrained models, datasets, and other permissible resources, but they must be trackable for later reporting.

At minimum, collaborators should keep track of:
- pretrained model IDs used in configs
- scoring models used in configs
- any future reference datasets or curated dev resources
- any new external resources that affect training, generation, or evaluation

## Immediate Next Collaboration Goals

Near-term repo goals:
1. keep `COLLAB.md` current as the shared contract
2. preserve stable run entrypoints
3. stabilize the future `SceneSpec` / `AnchorSpec` / `ScoringSpec` direction
4. keep major modules switchable through config
5. maintain a quick dev subset workflow
6. keep outputs and metadata easy to inspect
7. preserve a clean baseline comparison path
