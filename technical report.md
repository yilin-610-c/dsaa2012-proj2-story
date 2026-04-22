## Introduction / Task Understanding

写清楚你们理解的任务难点：

每格要忠实于 panel
跨格要一致
叙事要接得上
不能手工修图，必须是自动、可复现 pipeline

这些都和作业要求完全对齐。

## System Overview

一张总图 + 一段总述：

输入 story panels
prompt generation
generation backbone
reference/anchor conditioning
scoring/reranking
output packaging

## Prompt Generation Module

Implemented comparison points:

- Baseline prompt path: rule-based `PromptBuilder`.
- LLM-assisted prompt path: `prompt.pipeline=llm_assisted`.
- The LLM path uses structured JSON extraction, not freeform long prompt writing.
- The final downstream contract remains `PromptSpec`.

Important implementation details to report:

- `style_prompt` and `negative_prompt` remain local config fields.
- LLM output is constrained to shared identity/setting and short per-scene fields.
- `generation_prompt`, `scoring_prompt`, and `action_prompt` are short fields used by generation/scoring.
- Local code validates scene ids, trims lengths, removes command/question framing, and assembles `full_prompt`.
- In the LLM-assisted path, stable character identity is merged into the final `generation_prompt` so the image model sees both "who" and "what happens" in the actual generation input.
- For human main characters, the generated `negative_prompt` is augmented with animal/pet suppression terms to reduce human-to-animal substitutions.
- Local cache is under `.cache/prompt_builder/`; shareable prompt artifacts are under `prompt_artifacts/`.

## Consistency / Generation Module

Implemented Phase 1 route:

- Baseline: text2img for every scene, multi-candidate generation, CLIP-based reranking.
- New optional route: conservative img2img continuity for scene 2+.
- New optional guided route: LLM-assisted route metadata with small/medium/large execution levels.
- Scene 1 always uses text2img so the first frame establishes the story composition.
- For small-change scenes, img2img uses the previous selected image as init image plus the current scene prompt.
- For medium-change scenes, guided routing can use higher-strength img2img.
- For large-change scenes, routing falls back to text2img.

Design principle:

- Character portraits are not treated as scene generation.
- Identity anchors are planned as future conditioning inputs, not as scene 1 init images.
- Current goal is generation-time continuity injection without heavy adapter/control redesign.

Future roadmap:

1. Add `CharacterSpec` extraction for stable identity attributes only.
2. Add a lightweight anchor bank with portrait / half-body / optional full-body assets.
3. Add one identity-focused IP-Adapter.
4. Support hybrid routes: text2img, text2img + identity conditioning, img2img + identity conditioning.
5. Only later consider ControlNet, refiner, shared attention, or multi-adapter setups.

## Controlled Comparisons on Test-A

Core ablations:

- Rule prompt + text2img baseline:
  `PYTHONPATH=src python3 -m storygen.cli --profile smoke_test --input test_set/01.txt`
- LLM prompt + text2img:
  `PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_text2img --input test_set/01.txt`
- Rule prompt + img2img routing:
  `PYTHONPATH=src python3 -m storygen.cli --profile rule_prompt_img2img --input test_set/01.txt`
- LLM prompt + img2img routing:
  `PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_img2img --input test_set/01.txt`
- LLM prompt + guided multi-level img2img routing:
  `PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_img2img_guided --input test_set/01.txt`

What each comparison isolates:

- LLM prompt effect without generation-route changes.
- Img2img continuity effect without LLM prompt changes.
- Combined prompt + generation-time continuity effect.
- Guided route planning and small/medium/large img2img execution effect.

## Qualitative Results

挑 8–12 组最好、最能说明问题的例子。
这也是作业明确要求的。

## Failure Analysis

Observed and expected failure categories:

- API prompt rewriting can be too weak or too generic. Example: an early LLM prompt for the window scene used `Lily gazing out the window` without enough stable human identity cues, and SDXL generated a cat looking out the window.
- Wrong identity can propagate through img2img. In the same run, scene 3 used img2img from the selected scene 2 image, so the cat identity persisted into the next panel.
- Pronoun-only scenes can confuse routing and prompting. If the parser does not attach `Lily` to a scene like "She looks out the window quietly", conservative routing may miss the connection to the previous scene.
- CLIP reranking can overvalue image-to-image similarity. A visually coherent but semantically wrong frame may be selected if previous-image consistency is high.
- Scene changes can break continuity when text2img is needed, while img2img can over-preserve layout when the new scene should change composition.
- Img2img under-change failure: scene2 to scene3 may be routed to img2img but use too little strength, so the previous composition is preserved and the required new action, posture, or layout does not happen.
- Common object priors can override weak character prompts. Window-gazing images often bias toward pets or animals unless the prompt explicitly says human/person and negative prompts suppress animal substitutions.

Mitigations added after the cat failure:

- LLM prompt instructions now require visual, reusable identity cues and explicitly forbid substituting animals for human characters.
- The local prompt assembler merges stable identity into the actual `generation_prompt`.
- Human-character scenes append `cat, dog, animal, pet, non-human subject` to the `negative_prompt`.
- The LLM builder version was bumped to `llm_assisted_v3`, so old cached prompts that caused the failure are not reused.

Mitigations added after the img2img under-change failure:

- LLM-assisted prompt planning now records route metadata: `continuity_subject_ids`, `continuity_route_hint`, `route_change_level`, and `route_reason`.
- Routing can use LLM-resolved continuity subjects instead of relying only on parser entities, which helps pronoun-only scenes such as "She looks out the window."
- Guided routing separates execution by change level: `small` uses low-strength img2img, `medium` uses higher-strength img2img, and `large` falls back to text2img.
- The LLM builder version was bumped to `llm_assisted_v4`, so old cached prompts without route metadata are not reused.

Mitigations added after route-level underestimation and over-similarity reranking:

- The LLM route rubric was tightened: `small` requires nearly unchanged subject, setting, framing, body state, and action; same subject plus same setting is no longer sufficient.
- LLM-assisted prompt planning now records structured `route_factors`, including body-state change, primary-action change, new key objects, and composition-change need.
- Local code performs only generic self-consistency correction from the LLM's own route factors; no story-specific keyword route overrides are added.
- Route-aware scoring reduces previous-image consistency weight for `medium` and `large` changes, and applies an over-similarity penalty when the candidate remains too close to the previous panel.
- The LLM builder version was bumped to `llm_assisted_v5`, so old v4 cached route judgments are not reused.

## Data / External Resources / Compliance

这部分必须认真写：

用了哪些模型
用了哪些 API
数据怎么来的
有没有做筛选/清洗
如何保证自动化、无 manual per-case editing、无 hard-coding

Current external resources to report:

- Image generation backbone configured through diffusers model id.
- CLIP scorer configured through `openai/clip-vit-base-patch32`.
- Optional OpenAI API usage for LLM-assisted prompt planning.
- No manual per-case image editing.
- Experiment routes are selected by YAML profiles / CLI, not by editing code per case.
