"""Phase-scoped RouteTune command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .baseline import BaselineError, load_result, prediction_errors, run_validation_baseline
from .challenge import EXPECTED_CHALLENGE_FINGERPRINT, challenge_fingerprint
from .config import DEFAULT_CONFIG, EXPECTED_SELECTION_FINGERPRINT
from .data import DataError, OFFICIAL_SOURCES, build_splits, fetch_source, selection_fingerprint
from .evaluation import EvaluationError, parse_strict_label
from .inference import InferenceUnavailable, build_messages, run_quantized_qwen_smoke


def _load_locked_splits(cache_dir: Path, offline: bool):
    sources = {
        spec.name: fetch_source(spec, cache_dir, offline=offline)
        for spec in OFFICIAL_SOURCES
    }
    splits = build_splits(sources["official_train"], sources["official_test"])
    fingerprint = selection_fingerprint(splits)
    if fingerprint != EXPECTED_SELECTION_FINGERPRINT:
        raise DataError(
            "dataset selection fingerprint changed: "
            f"expected {EXPECTED_SELECTION_FINGERPRINT}, got {fingerprint}"
        )
    return splits, fingerprint


def _run_baseline(args: argparse.Namespace) -> dict[str, object]:
    splits, fingerprint = _load_locked_splits(args.cache_dir, args.offline)
    artifact = run_validation_baseline(
        training_records=splits.train,
        validation_records=splits.validation,
        labels=DEFAULT_CONFIG.labels,
        seed=DEFAULT_CONFIG.seed,
        dataset_fingerprint=fingerprint,
        output_dir=args.output_dir,
    )
    result = load_result(artifact)
    return {
        "artifact": str(artifact.resolve()),
        "metrics": result["metrics"],
        "official_test_evaluated": False,
    }


def _show_metrics(args: argparse.Namespace) -> dict[str, object]:
    result = load_result(args.artifact)
    return {
        "algorithm": result["algorithm"],
        "evaluation_split": result["evaluation_split"],
        "dataset_selection_fingerprint": result["dataset_selection_fingerprint"],
        "metrics": result["metrics"],
    }


def _show_errors(args: argparse.Namespace) -> dict[str, object]:
    result = load_result(args.artifact)
    errors = prediction_errors(result, args.limit)
    return {"shown": len(errors), "limit": args.limit, "errors": errors}


def _prompt_diagnostics(_args: argparse.Namespace) -> dict[str, object]:
    message = "My card purchase is still pending."
    raw_examples = (
        "pending_card_payment",
        "pending_card_payment\n",
        "Label: pending_card_payment",
        "Pending_Card_Payment",
        "pending_card_payment because it is processing",
        "",
    )
    return {
        "messages": build_messages(message),
        "parser_examples": [
            {
                "raw": raw,
                "parsed": parse_strict_label(raw, DEFAULT_CONFIG.labels),
                "valid": parse_strict_label(raw, DEFAULT_CONFIG.labels) is not None,
            }
            for raw in raw_examples
        ],
        "challenge_fingerprint": challenge_fingerprint(),
        "challenge_fingerprint_matches_frozen": (
            challenge_fingerprint() == EXPECTED_CHALLENGE_FINGERPRINT
        ),
    }


def _qwen_smoke(args: argparse.Namespace) -> dict[str, object]:
    return run_quantized_qwen_smoke(args.message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RouteTune Phase 2 tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline", help="fit training data and evaluate validation data"
    )
    baseline.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    baseline.add_argument("--output-dir", type=Path, required=True)
    baseline.add_argument("--offline", action="store_true")
    baseline.set_defaults(handler=_run_baseline)

    metrics = subparsers.add_parser("show-metrics", help="inspect saved metrics")
    metrics.add_argument("artifact", type=Path)
    metrics.set_defaults(handler=_show_metrics)

    errors = subparsers.add_parser("show-errors", help="inspect prediction errors")
    errors.add_argument("artifact", type=Path)
    errors.add_argument("--limit", type=int, default=10)
    errors.set_defaults(handler=_show_errors)

    diagnostics = subparsers.add_parser(
        "prompt-diagnostics", help="run CPU-only prompt and parser checks"
    )
    diagnostics.set_defaults(handler=_prompt_diagnostics)

    smoke = subparsers.add_parser(
        "qwen-smoke", help="optionally run the real local quantized Qwen model"
    )
    smoke.add_argument(
        "--message", default="My replacement card has not arrived yet."
    )
    smoke.set_defaults(handler=_qwen_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (
        BaselineError,
        DataError,
        EvaluationError,
        InferenceUnavailable,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

