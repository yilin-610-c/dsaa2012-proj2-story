from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECORD_FIELDS = [
    "experiment_id",
    "run_name",
    "profile",
    "input_story_path",
    "command",
    "argv",
    "status",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "git_commit_hash",
    "repo_dirty",
    "candidate_count",
    "width",
    "height",
    "base_seed",
    "cuda_visible_devices",
    "output_directory",
]


@dataclass(frozen=True)
class MatrixJob:
    experiment_id: str
    profile: str
    story_path: Path
    run_name: str
    output_root: Path
    output_directory: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_profiles(value: str) -> list[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    if not profiles:
        raise argparse.ArgumentTypeError("--profiles must include at least one profile")
    return profiles


def validate_name(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError(f"{field_name} cannot be empty")
    if "/" in normalized or "\\" in normalized:
        raise argparse.ArgumentTypeError(f"{field_name} cannot contain path separators: {value}")
    return normalized


def resolve_story_paths(stories: str) -> list[Path]:
    if "," in stories:
        paths = [Path(item.strip()) for item in stories.split(",") if item.strip()]
    else:
        paths = [Path(item) for item in glob.glob(stories)]
    paths = sorted(dict.fromkeys(paths), key=lambda path: str(path))
    if not paths:
        raise FileNotFoundError(f"No stories matched: {stories}")
    return paths


def git_commit_hash(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def repo_is_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def build_jobs(experiment_id: str, profiles: list[str], story_paths: list[Path], output_root: Path) -> list[MatrixJob]:
    jobs = []
    for profile in profiles:
        clean_profile = validate_name(profile, field_name="profile")
        for story_path in story_paths:
            story_stem = validate_name(story_path.stem, field_name="story stem")
            run_name = f"{experiment_id}__{clean_profile}__{story_stem}"
            jobs.append(
                MatrixJob(
                    experiment_id=experiment_id,
                    profile=clean_profile,
                    story_path=story_path,
                    run_name=run_name,
                    output_root=output_root,
                    output_directory=output_root / run_name,
                )
            )
    return jobs


def build_child_argv(job: MatrixJob, args: argparse.Namespace) -> list[str]:
    argv = [
        "python3",
        "-m",
        "storygen.cli",
        "--config",
        args.config,
        "--profile",
        job.profile,
        "--input",
        str(job.story_path),
        "--run-name",
        job.run_name,
        "--set",
        f"runtime.output_root={job.output_root}",
    ]
    if args.candidate_count is not None:
        argv.extend(["--num-candidates", str(args.candidate_count)])
    if args.base_seed is not None:
        argv.extend(["--base-seed", str(args.base_seed)])
    if args.width is not None:
        argv.extend(["--set", f"model.width={args.width}"])
    if args.height is not None:
        argv.extend(["--set", f"model.height={args.height}"])
    for override in args.extra_set:
        argv.extend(["--set", override])
    return argv


def command_string(argv: list[str], *, cuda_visible_devices: str) -> str:
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} PYTHONPATH=src {shlex.join(argv)}"


def record_for_job(
    job: MatrixJob,
    args: argparse.Namespace,
    argv: list[str],
    *,
    status: str,
    start_time: str,
    end_time: str,
    elapsed_seconds: float,
    git_hash: str | None,
    dirty: bool | None,
) -> dict[str, Any]:
    return {
        "experiment_id": job.experiment_id,
        "run_name": job.run_name,
        "profile": job.profile,
        "input_story_path": str(job.story_path),
        "command": command_string(argv, cuda_visible_devices=args.cuda_visible_devices),
        "argv": argv,
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "git_commit_hash": git_hash,
        "repo_dirty": dirty,
        "candidate_count": args.candidate_count,
        "width": args.width,
        "height": args.height,
        "base_seed": args.base_seed,
        "cuda_visible_devices": args.cuda_visible_devices,
        "output_directory": str(job.output_directory),
    }


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["argv"] = json.dumps(row["argv"], ensure_ascii=True)
            writer.writerow(row)


def write_readme(path: Path, *, experiment_id: str, output_root: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    status_lines = "\n".join(f"- `{status}`: {count}" for status, count in sorted(status_counts.items())) or "- no jobs recorded"
    content = f"""# Remote Experiment: {experiment_id}

Output root: `{output_root}`

This directory was produced by `scripts/run_experiment_matrix.py`.

## Records

- `manifest.jsonl` is append-only execution history. Repeated invocations append more records.
- `summary.csv` is rewritten by each batch invocation for quick inspection of the latest batch results.
- `batch_logs/*.out` and `batch_logs/*.err` contain captured child CLI stdout and stderr.

## Latest Batch

Profiles: `{','.join(args.profiles)}`
Stories: `{args.stories}`
CUDA_VISIBLE_DEVICES: `{args.cuda_visible_devices}`

Statuses:

{status_lines}

## Inspect

```bash
column -s, -t < {output_root}/summary.csv | less -S
tail -n 20 {output_root}/manifest.jsonl
ls -lt {output_root}/batch_logs
find {output_root} -path '*/scenes/*/prompt.json' | sort
```

## Resume

Rerun the same command with `--resume --continue-on-error`. Jobs with an existing `run_summary.json` are recorded as `skipped`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_job(job: MatrixJob, args: argparse.Namespace, *, git_hash: str | None, dirty: bool | None, batch_logs_dir: Path) -> dict[str, Any]:
    argv = build_child_argv(job, args)
    out_path = batch_logs_dir / f"{job.run_name}.out"
    err_path = batch_logs_dir / f"{job.run_name}.err"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start_monotonic = time.monotonic()
    start_time = utc_now()

    if args.dry_run:
        out_path.write_text(command_string(argv, cuda_visible_devices=args.cuda_visible_devices) + "\n", encoding="utf-8")
        err_path.write_text("", encoding="utf-8")
        status = "dry_run"
    elif args.resume and (job.output_directory / "run_summary.json").exists():
        out_path.write_text(f"Skipped existing run: {job.output_directory / 'run_summary.json'}\n", encoding="utf-8")
        err_path.write_text("", encoding="utf-8")
        status = "skipped"
    else:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr, env=env)
        status = "success" if result.returncode == 0 else "failed"

    end_time = utc_now()
    elapsed_seconds = time.monotonic() - start_monotonic
    return record_for_job(
        job,
        args,
        argv,
        status=status,
        start_time=start_time,
        end_time=end_time,
        elapsed_seconds=elapsed_seconds,
        git_hash=git_hash,
        dirty=dirty,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sequential remote experiment matrix through storygen.cli")
    parser.add_argument("--experiment-id", required=True, type=lambda value: validate_name(value, field_name="experiment-id"))
    parser.add_argument("--profiles", required=True, type=parse_profiles)
    parser.add_argument("--stories", default="test_set/*.txt")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--extra-set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.cuda_visible_devices = str(args.cuda_visible_devices)

    experiment_id = args.experiment_id
    output_root = Path("outputs_remote") / experiment_id
    batch_logs_dir = output_root / "batch_logs"
    manifest_path = output_root / "manifest.jsonl"
    summary_path = output_root / "summary.csv"
    readme_path = output_root / "README.md"
    story_paths = resolve_story_paths(args.stories)
    jobs = build_jobs(experiment_id, args.profiles, story_paths, output_root)
    repo_root = Path(".").resolve()
    git_hash = git_commit_hash(repo_root)
    dirty = repo_is_dirty(repo_root)

    latest_records: list[dict[str, Any]] = []
    for job in jobs:
        record = run_job(job, args, git_hash=git_hash, dirty=dirty, batch_logs_dir=batch_logs_dir)
        append_manifest(manifest_path, record)
        latest_records.append(record)
        if record["status"] == "failed" and not args.continue_on_error:
            break

    write_summary(summary_path, latest_records)
    write_readme(readme_path, experiment_id=experiment_id, output_root=output_root, records=latest_records, args=args)
    print(f"Experiment complete: {output_root}")
    print(f"Summary: {summary_path}")
    return 1 if any(record["status"] == "failed" for record in latest_records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
