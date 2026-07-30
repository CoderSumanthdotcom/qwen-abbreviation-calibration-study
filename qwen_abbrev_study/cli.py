"""Command-line interface with an explicit live-request safety gate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .config import ConfigError, load_config
from .datasets import DatasetError
from .runner import StudyExecutionError, run_study
from .storage import StorageError


def _comma_separated(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("must contain at least one value")
    return items


def _prompt_ids_file(value: str) -> list[str]:
    path = Path(value).expanduser()
    try:
        items = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as error:
        raise argparse.ArgumentTypeError(f"cannot read prompt ID file: {error}") from error
    if not items:
        raise argparse.ArgumentTypeError("prompt ID file must contain at least one ID")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("prompt ID file contains duplicate IDs")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-abbreviation-study",
        description=(
            "Run prepared Qwen3 abbreviation prompts. Defaults to a network-free dry run."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to the study JSON config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize live provider requests",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only; this is the default",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing matching experiment output directory",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="With --resume, rerun records that already succeeded",
    )
    parser.add_argument(
        "--models",
        type=_comma_separated,
        help="Comma-separated model keys",
    )
    parser.add_argument(
        "--strategies",
        type=_comma_separated,
        help="Comma-separated strategy IDs",
    )
    parser.add_argument(
        "--prompt-ids-file",
        type=_prompt_ids_file,
        help="Run only prompt IDs listed one per line in this file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum prompt rows per selected model/strategy",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.rerun and not args.resume:
        parser.error("--rerun requires --resume")
    try:
        config = load_config(args.config)
        summary = run_study(
            config,
            execute=args.execute,
            resume=args.resume,
            rerun=args.rerun,
            model_keys=args.models,
            strategy_ids=args.strategies,
            prompt_ids=args.prompt_ids_file,
            limit=args.limit,
        )
    except (ConfigError, DatasetError, StorageError, StudyExecutionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1 if summary.completed_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
