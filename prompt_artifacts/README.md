# Prompt Artifacts

This directory is for shareable, reviewed prompt outputs from the `llm_assisted` prompt pipeline.

Files here are different from `.cache/prompt_builder/`:

- `.cache/prompt_builder/` is a local acceleration cache and should not be committed.
- `prompt_artifacts/` is a collaboration surface for prompt JSON files that teammates can reuse without calling the LLM API again.

To reuse an artifact, set:

```yaml
prompt:
  pipeline: llm_assisted
  artifact:
    path: prompt_artifacts/llm_assisted_v3/example.json
```
