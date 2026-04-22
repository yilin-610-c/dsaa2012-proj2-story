# AGENTS.md

This file defines working rules for coding agents operating in this repository.

## 1. General principle

Preserve the current working baseline unless the task explicitly requires changing its behavior.

Prefer small, traceable, reversible changes over broad refactors.

Do not silently change core contracts, routing behavior, logging structure, or output layout without updating docs.

## 2. Before making code changes

Before any non-trivial code change:

1. Inspect the current repo status:
   - `git status`
   - `git branch --show-current`
   - `git rev-parse --short HEAD`

2. If the working tree is in a safe state, create a git checkpoint before the change:
   - checkpoint commit message format:
     - `checkpoint: before <short-task-name>`

3. If a checkpoint commit is not appropriate, record the current commit SHA and planned change in the relevant docs or task notes.

Do not begin large edits without first creating a checkpoint or equivalent traceable record.

## 3. Scope control

Do not mix unrelated changes in one pass.

If the task is about routing, do not also refactor prompt generation, scoring, or unrelated config structure unless required.

If the task is about prompt generation, do not also redesign generators unless required.

Keep PR-sized changes modular.

## 4. Code modification rules

When editing code:

- preserve backwards compatibility when possible
- avoid renaming public/shared fields unless necessary
- do not silently break existing profiles
- prefer config-gated behavior over hard-coded behavior
- prefer additive changes over destructive rewrites

If changing shared interfaces, update all dependent code and docs in the same change.

## 5. Experiment and run logging

If a change affects behavior, add or update the relevant experiment command in project documentation.

At minimum, document:
- profile name
- command used
- purpose of the run
- important config toggles

Prefer updating a dedicated experiment log file if present.
If none exists, update README or the agreed experiment notes file.

## 6. Required documentation updates

When changing any of the following, update documentation in the same task:

- config/profile behavior
- prompt pipeline behavior
- routing policy
- generation backend behavior
- output/logging structure
- experiment commands

At minimum, reflect the change in one of:
- `README.md`
- `COLLAB.md`
- `docs/`
- experiment notes / technical report notes

## 7. Testing expectations

For every non-trivial change:

1. Run the most relevant targeted tests if available.
2. If tests do not exist, add focused tests when reasonable.
3. If runtime behavior changed, run at least one minimal executable command for sanity check.
4. Record what was run.

Do not claim a change is working unless it was actually tested or the lack of testing is explicitly stated.

## 8. Output and artifact discipline

Do not change output file naming or output directory structure unless required.

If output structure changes, update:
- README
- manifest/logging docs
- any code that depends on saved artifacts

## 9. Git discipline

Do not commit directly to main unless explicitly requested.

Prefer working on a feature branch.

Checkpoint commit message:
- `checkpoint: before <task>`

Normal commit message examples:
- `feat(routing): add llm-guided conservative route policy`
- `feat(generator): add img2img route support`
- `fix(prompt): use llm identity metadata in routing`
- `docs(experiments): add llm-guided img2img run commands`

## 10. When uncertain

If implementation details are unclear:
- preserve current behavior
- make the smallest safe change
- leave TODOs only when necessary
- document assumptions clearly

Do not invent hidden behavior that is not reflected in config, logs, or docs.