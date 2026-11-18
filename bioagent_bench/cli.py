"""Command-line interface."""

import argparse
import json
from pathlib import Path

from .gateway import DEFAULT_MODEL
from .runner import ensure_bank, run_suite
from .taskbank import TASK_IDS, generate_bank


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="bioagent-bench", description="Independent synthetic computational-biology agent benchmark")
    subparsers = result.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate deterministic public and sealed synthetic task data")
    generate.add_argument("--bank", type=Path, default=Path(".bench"))
    generate.add_argument("--seed", type=int, default=2026)
    generate.add_argument("--force", action="store_true", help="Replace the bank directory if it exists")

    run = subparsers.add_parser("run", help="Run an AI agent and grade its submissions")
    run.add_argument("--bank", type=Path, default=Path(".bench"))
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--tasks", default=",".join(TASK_IDS), help="Comma-separated task ids")
    run.add_argument("--replicates", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=60)
    run.add_argument("--agent-time-limit", type=int, default=1800, help="Seconds per task replicate")
    run.add_argument("--grade-time-limit", type=int, default=300, help="Seconds per prediction execution")
    run.add_argument("--max-output-tokens", type=int, default=8192)
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--run-id")

    tasks = subparsers.add_parser("tasks", help="List generated tasks")
    tasks.add_argument("--bank", type=Path, default=Path(".bench"))
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "generate":
        if args.force and args.bank.exists():
            import shutil
            shutil.rmtree(args.bank)
        generate_bank(args.bank, seed=args.seed)
        print("Generated %d tasks in %s" % (len(TASK_IDS), args.bank))
        return
    if args.command == "tasks":
        ensure_bank(args.bank)
        for task_id in TASK_IDS:
            metadata = json.loads((args.bank / "tasks" / task_id / "task.json").read_text(encoding="utf-8"))
            print("%s  %-30s  %s" % (task_id, metadata["title"], metadata["category"]))
        return
    if args.command == "run":
        ensure_bank(args.bank, seed=args.seed)
        tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
        summary = run_suite(
            bank=args.bank,
            output_root=args.output,
            model=args.model,
            task_ids=tasks,
            replicates=args.replicates,
            max_steps=args.max_steps,
            agent_time_limit=args.agent_time_limit,
            grade_time_limit=args.grade_time_limit,
            max_output_tokens=args.max_output_tokens,
            run_id=args.run_id,
        )
        print("\n" + json.dumps(summary, indent=2, sort_keys=True))
        return


if __name__ == "__main__":
    main()
