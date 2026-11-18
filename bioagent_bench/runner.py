"""Run generation, agents, and sealed grading."""

import json
import os
import shutil
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .agent import run_agent
from .gateway import DEFAULT_MODEL
from .taskbank import TASK_IDS, generate_bank, grade_task, validate_bank


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_bank(bank: Path, seed: int = None) -> Dict[str, Any]:
    """Ensure a usable bank exists and return its metadata.

    A missing bank is generated.  An existing bank must carry compatible
    ``bank.json`` metadata; a stale v0.1 bank or a seed mismatch is rejected
    with an explicit regeneration command instead of being silently reused.
    ``seed=None`` validates only the format version (used by read-only listing).
    """
    bank = Path(bank)
    if not (bank / "tasks").is_dir():
        generate_bank(bank, seed=2026 if seed is None else seed)
    return validate_bank(bank, expected_seed=seed)


def _zero_result(task_id: str, reason: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "score": 0.0,
        "passed_gate": False,
        "reason": reason,
        "deterministic": False,
        "batch_invariant": False,
        "metrics": {},
        "attempts": [],
    }


def run_suite(
    bank: Path,
    output_root: Path,
    model: str = DEFAULT_MODEL,
    task_ids: List[str] = None,
    replicates: int = 1,
    max_steps: int = 60,
    agent_time_limit: int = 1800,
    grade_time_limit: int = 300,
    max_output_tokens: int = 8192,
    run_id: str = None,
    seed: int = None,
) -> Dict[str, Any]:
    tasks = task_ids or list(TASK_IDS)
    unknown = set(tasks) - set(TASK_IDS)
    if unknown:
        raise ValueError("Unknown task ids: " + ", ".join(sorted(unknown)))
    if len(set(tasks)) != len(tasks):
        raise ValueError("task_ids must not contain duplicates")
    if replicates < 1:
        raise ValueError("replicates must be at least 1")
    bank_metadata = validate_bank(bank, expected_seed=seed)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "created_at": _now(),
        "model": model,
        "task_ids": tasks,
        "replicates": replicates,
        "max_steps": max_steps,
        "agent_time_limit_seconds": agent_time_limit,
        "grade_time_limit_seconds": grade_time_limit,
        "max_output_tokens": max_output_tokens,
        "bank": str(bank.resolve()),
        "bank_version": bank_metadata["version"],
        "bank_seed": bank_metadata["seed"],
        "bank_metadata": bank_metadata,
    }
    _dump(run_dir / "config.json", config)
    records = []

    for task_id in tasks:
        task_dir = bank / "tasks" / task_id
        task_meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        for replicate in range(1, replicates + 1):
            label = "%s replicate %d/%d" % (task_id, replicate, replicates)
            print("\n=== %s ===" % label, flush=True)
            replicate_dir = run_dir / task_id / ("replicate_%d" % replicate)
            workspace = replicate_dir / "workspace"
            workspace.mkdir(parents=True)
            shutil.copy2(task_dir / "brief.md", workspace / "BRIEF.md")
            shutil.copytree(task_dir / "public", workspace / "data")
            started = time.monotonic()
            agent_result = run_agent(
                workspace=workspace,
                log_dir=replicate_dir,
                task_title="%s — %s" % (task_id, task_meta["title"]),
                model=model,
                max_steps=max_steps,
                time_limit_seconds=agent_time_limit,
                max_output_tokens=max_output_tokens,
            )
            if agent_result.get("pipeline_present"):
                try:
                    grading = grade_task(
                        task_dir=task_dir,
                        submission_dir=workspace,
                        grade_dir=replicate_dir / "grading",
                        timeout_seconds=grade_time_limit,
                    )
                except Exception as exc:
                    grading = _zero_result(
                        task_id,
                        "grader exception: %s: %s" % (type(exc).__name__, exc),
                    )
            else:
                grading = _zero_result(task_id, "agent did not create pipeline/predict.sh")
            _dump(replicate_dir / "grading_result.json", grading)
            record = {
                "task_id": task_id,
                "replicate": replicate,
                "score": grading["score"],
                "passed_gate": grading["passed_gate"],
                "grading_reason": grading["reason"],
                "agent_status": agent_result["status"],
                "agent_error": agent_result.get("error"),
                "agent_seconds": agent_result["elapsed_seconds"],
                "total_seconds": round(time.monotonic() - started, 3),
                "steps": agent_result["steps"],
                "usage": agent_result["usage"],
                "metrics": grading.get("metrics", {}),
                "path": str(replicate_dir.relative_to(run_dir)),
            }
            records.append(record)
            _dump(run_dir / "records.json", records)
            print("Result: %.3f/100 | agent=%s | gate=%s | %s" % (
                record["score"], record["agent_status"], record["passed_gate"], record["grading_reason"]
            ), flush=True)
            _dump(run_dir / "summary.json", summarize(config, records))
    summary = summarize(config, records)
    summary["finished_at"] = _now()
    _dump(run_dir / "summary.json", summary)
    return summary


def summarize(config: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    task_scores = {task_id: statistics.mean(item["score"] for item in items) for task_id, items in sorted(by_task.items())}
    expected = len(config["task_ids"]) * config["replicates"]
    expected_pairs = Counter(
        (task_id, replicate)
        for task_id in config["task_ids"]
        for replicate in range(1, config["replicates"] + 1)
    )
    actual_pairs = Counter((record["task_id"], record["replicate"]) for record in records)
    complete = actual_pairs == expected_pairs
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "upstream_inference_cost_usd": 0.0,
        "market_cost_usd": 0.0,
    }
    for record in records:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(record.get("usage", {}).get(key, 0) or 0)
        for key in ("upstream_inference_cost_usd", "market_cost_usd"):
            usage[key] += float(record.get("usage", {}).get(key, 0) or 0)
    return {
        "run_id": config["run_id"],
        "model": config["model"],
        "score": statistics.mean(task_scores.values()) if task_scores and complete else 0.0,
        "complete": complete,
        "task_scores": task_scores,
        "completed_replicates": len(records),
        "expected_replicates": expected,
        "passed_gate": sum(bool(record["passed_gate"]) for record in records),
        "agent_statuses": {status: sum(r["agent_status"] == status for r in records) for status in sorted({r["agent_status"] for r in records})},
        "usage": usage,
    }
