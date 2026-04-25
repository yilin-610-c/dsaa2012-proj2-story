# Remote Experiments

This project supports local code editing and managed remote GPU experiment runs. The intended workflow is:

1. Edit and test locally.
2. Commit and push to GitHub.
3. Pull the latest code on the remote GPU server.
4. Run a sequential experiment matrix with `scripts/run_experiment_matrix.py`.
5. Inspect `outputs_remote/<experiment_id>/` records and per-run outputs.

The remote server has 2 x RTX 2080 Ti GPUs with about 11GB VRAM each. Another experiment may be running lightly, so use conservative settings and run sequentially. Use `CUDA_VISIBLE_DEVICES=1` by default for story generation experiments. Do not assume 1024 x 1024 or heavy parallel jobs are safe.

## Setup

On the remote server:

```bash
git pull
conda activate storygen
export OPENAI_API_KEY=...
export CUDA_VISIBLE_DEVICES=1
```

Run a quick dry run before launching generation:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --experiment-id remote_dry_run_check \
  --profiles llm_prompt_text2img,llm_prompt_two_character_text2img \
  --stories test_set/01.txt,test_set/06.txt \
  --candidate-count 1 \
  --width 512 \
  --height 512 \
  --base-seed 2026 \
  --extra-set model.num_inference_steps=4 \
  --dry-run
```

## Smoke Run

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --experiment-id remote_smoke_v1 \
  --profiles llm_prompt_text2img,llm_prompt_two_character_text2img \
  --stories "test_set/*.txt" \
  --candidate-count 1 \
  --width 512 \
  --height 512 \
  --base-seed 2026 \
  --resume \
  --continue-on-error
```

## Main Probe

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python scripts/run_experiment_matrix.py \
  --experiment-id final_probe_v1 \
  --profiles llm_prompt_text2img,llm_prompt_ip_adapter_text2img,llm_prompt_two_character_text2img \
  --stories "test_set/*.txt" \
  --candidate-count 3 \
  --width 768 \
  --height 768 \
  --base-seed 2026 \
  --resume \
  --continue-on-error
```

Use `--extra-set` for remote-specific overrides. Each value is forwarded to the existing CLI as `--set KEY=VALUE`:

```bash
--extra-set model.enable_attention_slicing=true
--extra-set model.num_inference_steps=4
```

## Resume Failed Runs

Rerun the same command with:

```bash
--resume --continue-on-error
```

`--resume` skips a job only when its `run_summary.json` already exists. Skipped jobs are recorded with `status=skipped`. Dry runs are recorded with `status=dry_run`.

## Inspect Outputs

Experiment-level records live under:

```text
outputs_remote/<experiment_id>/
```

Important files:

- `manifest.jsonl`: append-only execution history. Repeated invocations append new records.
- `summary.csv`: latest batch table for quick inspection. This file is rewritten on each invocation.
- `README.md`: generated experiment notes and inspection commands.
- `batch_logs/*.out`: captured child CLI stdout.
- `batch_logs/*.err`: captured child CLI stderr.

Per-run files remain under:

```text
outputs_remote/<experiment_id>/<run_name>/
```

Useful inspection commands:

```bash
column -s, -t < outputs_remote/<experiment_id>/summary.csv | less -S
tail -n 20 outputs_remote/<experiment_id>/manifest.jsonl
ls -lt outputs_remote/<experiment_id>/batch_logs
find outputs_remote/<experiment_id> -path '*/scenes/*/prompt.json' | sort
find outputs_remote/<experiment_id> -path '*/logs/events.jsonl' | sort
```

For prompt quality checks, inspect:

```bash
cat outputs_remote/<experiment_id>/<run_name>/scenes/scene_001/prompt.json
```

For route and identity-conditioning behavior, inspect:

```bash
cat outputs_remote/<experiment_id>/<run_name>/logs/events.jsonl
cat outputs_remote/<experiment_id>/<run_name>/logs/prompt_bundle.json
```
