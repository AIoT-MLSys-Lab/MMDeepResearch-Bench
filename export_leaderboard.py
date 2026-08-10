from __future__ import annotations

import argparse
from pathlib import Path

from detail.leaderboard_export import export_leaderboard_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical MMDR leaderboard and task-level scores.")
    parser.add_argument("--results", required=True, type=Path, help="Pipeline results JSONL.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Defaults to the run summary directory.")
    parser.add_argument("--model_name", default="", help="Leaderboard display name; inferred when omitted.")
    parser.add_argument("--split", choices=("all", "research", "daily"), default="all")
    parser.add_argument("--expected_tasks", type=int, default=140)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    outputs = export_leaderboard_outputs(
        results_jsonl=args.results,
        output_dir=args.output_dir,
        model_name=args.model_name,
        split=args.split,
        expected_tasks=args.expected_tasks,
        require_complete=args.require_complete,
    )
    for name, path in outputs.items():
        try:
            display = path.relative_to(Path.cwd())
        except ValueError:
            display = path.name
        print(f"{name}: {display}")


if __name__ == "__main__":
    main()
