"""
CSIT5520 NLI Evaluation Pipeline — Global Orchestration.

This script executes the full NLI evaluation pipeline, supporting both
zero-shot prompting and fine-tuning paradigms on MultiNLI matched and
mismatched splits.

Usage::

    # Evaluate only the prompting paradigm on both splits
    python main.py --mode prompting

    # Fine‑tune RoBERTa and evaluate on the mismatched split only
    python main.py --mode finetuning --split mismatched

    # Run everything (prompting + finetuning) on all splits
    python main.py --mode all --split both
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the Python path regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import logging
from dataclasses import asdict
from typing import List, Optional, Sequence

from sklearn.metrics import accuracy_score

from config import RUNTIME_CONFIG
from data_handler import NLIDataHandler
from evaluator_finetuning import RobertaFinetuneEvaluator
from evaluator_prompting import FlanT5PromptEvaluator
from utils import set_seed, setup_logger

logger = setup_logger(__name__)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command‑line arguments.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument list; if ``None``, uses ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes ``mode`` and ``split``.
    """
    parser = argparse.ArgumentParser(
        description="CSIT5520 NLI Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["prompting", "finetuning", "all"],
        default="all",
        help="Evaluation paradigm (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        choices=["matched", "mismatched", "both"],
        default="both",
        help="Data split(s) to evaluate (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _resolve_splits(split_arg: str) -> List[str]:
    """Convert the CLI split argument to a list of split names."""
    if split_arg == "both":
        return ["matched", "mismatched"]
    return [split_arg]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_prompting(splits: List[str]) -> None:
    """Evaluate with zero‑shot FLAN‑T5 prompting.

    Parameters
    ----------
    splits : list of str
        Names of the splits to evaluate (e.g. ``["matched", "mismatched"]``).
    """
    logger.info("=== Paradigm A: Zero-shot Prompting (FLAN‑T5) ===")
    handler = NLIDataHandler()
    evaluator = FlanT5PromptEvaluator()

    for split in splits:
        y_true = evaluator.get_ground_truth(split)
        y_pred = evaluator.evaluate(split)
        acc = accuracy_score(y_true, y_pred)
        logger.info(
            "[Prompting] Split: %-10s | Accuracy: %.4f",
            split,
            acc,
        )


def run_finetuning(splits: List[str]) -> None:
    """Fine‑tune RoBERTa and evaluate on the requested splits.

    Parameters
    ----------
    splits : list of str
        Names of the splits to evaluate after training.
    """
    logger.info("=== Paradigm B: Fine‑tuning (RoBERTa) ===")
    handler = NLIDataHandler()
    evaluator = RobertaFinetuneEvaluator(handler)

    # Train (matched → train, mismatched → eval during training)
    evaluator.train()

    # Evaluate on each requested split
    for split in splits:
        y_true = evaluator.get_ground_truth(split)
        y_pred = evaluator.evaluate(split)
        acc = accuracy_score(y_true, y_pred)
        logger.info(
            "[Fine‑tuning] Split: %-10s | Accuracy: %.4f",
            split,
            acc,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    """Orchestrate the NLI evaluation pipeline.

    Parameters
    ----------
    argv : sequence of str, optional
        Command‑line arguments; default is ``sys.argv``.
    """
    args = _parse_args(argv)

    # Global determinism
    set_seed(RUNTIME_CONFIG.seed)

    splits = _resolve_splits(args.split)
    logger.info("Selected splits: %s", splits)

    if args.mode in ("prompting", "all"):
        run_prompting(splits)

    if args.mode in ("finetuning", "all"):
        run_finetuning(splits)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()