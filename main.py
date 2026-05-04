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
import json
import os
from datetime import datetime
from typing import List, Optional, Sequence

from sklearn.metrics import accuracy_score

from config import RESULTDIRS, RUNTIME_CONFIG
from data_handler import NLIDataHandler, load_crows_pairs
from error_analysis import NLIErrorAnalyzer
from evaluator_finetuning import RobertaFinetuneEvaluator
from evaluator_masked import BiasEvaluator
from evaluator_prompting import FlanT5PromptEvaluator
from hallucination_evaluator import HallucinationEvaluator
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
        Parsed arguments with attributes ``task``, ``mode``, ``split``,
        and ``skip_train``.
    """
    parser = argparse.ArgumentParser(
        description="CSIT5520 NLI Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["nli", "hallucination", "bias"],
        default="nli",
        help="High‑level task to run: NLI evaluation, Hallucination "
        "detection, or Bias evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=["prompting", "finetuning", "all"],
        default="all",
        help="Evaluation paradigm (default: %(default)s). Only used when --task nli.",
    )
    parser.add_argument(
        "--split",
        choices=["matched", "mismatched", "both"],
        default="both",
        help="Data split(s) to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        default=False,
        help="Skip fine‑tuning and use the existing checkpoint. "
        "Ignored for prompting mode.",
    )
    parser.add_argument(
        "--bias_n_samples",
        type=int,
        default=80,
        help="Number of CrowS-Pairs samples for bias evaluation (default: 80).",
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

        # ------------------------------------------------------------------
        # Qualitative error analysis — export misclassified examples
        # ------------------------------------------------------------------
        analyzer = NLIErrorAnalyzer()
        ds = handler.get_dataset(split)
        error_path = RESULTDIRS.root / "errors" / f"prompting_{split}_errors.csv"
        errors_df = analyzer.extract_and_export_errors(ds, y_pred, error_path)
        logger.info(
            "[Prompting] Split: %-10s | Accuracy: %.4f | %d errors → %s",
            split,
            acc,
            len(errors_df),
            error_path,
        )


def run_finetuning(splits: List[str], skip_train: bool = False) -> None:
    """Fine‑tune RoBERTa and evaluate on the requested splits.

    Parameters
    ----------
    splits : list of str
        Names of the splits to evaluate after training.
    skip_train : bool, optional
        If ``True``, skip the training phase (default: ``False``).
    """
    logger.info("=== Paradigm B: Fine‑tuning (RoBERTa) ===")
    handler = NLIDataHandler()
    evaluator = RobertaFinetuneEvaluator(handler)

    # Train only when explicitly requested (matched → train, mismatched → eval)
    if not skip_train:
        evaluator.train()

    # Evaluate on each requested split
    for split in splits:
        y_true = evaluator.get_ground_truth(split)
        y_pred = evaluator.evaluate(split)
        acc = accuracy_score(y_true, y_pred)

        # ------------------------------------------------------------------
        # Qualitative error analysis — export misclassified examples
        # ------------------------------------------------------------------
        analyzer = NLIErrorAnalyzer()
        ds = handler.get_dataset(split)
        error_path = RESULTDIRS.root / "errors" / f"finetuning_{split}_errors.csv"
        errors_df = analyzer.extract_and_export_errors(ds, y_pred, error_path)
        logger.info(
            "[Fine‑tuning] Split: %-10s | Accuracy: %.4f | %d errors → %s",
            split,
            acc,
            len(errors_df),
            error_path,
        )


# ---------------------------------------------------------------------------
# Bias evaluation task
# ---------------------------------------------------------------------------


def run_bias_evaluation(n_samples: int = 80) -> None:
    """Evaluate social bias in Masked LMs using CrowS-Pairs & PLL.

    Loads the CrowS-Pairs dataset, then for each model in the evaluation
    set, computes the Pseudo-Log-Likelihood bias score, logs summary
    statistics, and cleans up GPU/MPS memory before loading the next model.

    Parameters
    ----------
    n_samples : int, optional
        Number of CrowS-Pairs examples to sample (default: 80).
    """
    logger.info("=== Task C: Bias Evaluation (CrowS-Pairs + PLL) ===")

    # ------------------------------------------------------------------
    # 1. Load the CrowS-Pairs dataset
    # ------------------------------------------------------------------
    logger.info("Loading CrowS-Pairs dataset (n_samples=%d) ...", n_samples)
    pairs = load_crows_pairs(
        bias_type="socioeconomic status/occupation",
        n_samples=n_samples,
        seed=RUNTIME_CONFIG.seed,
    )
    logger.info("Loaded %d CrowS-Pairs examples.", len(pairs))

    stereotypic_sentences = [p.sentence_stereo for p in pairs]
    anti_stereotypic_sentences = [p.sentence_antistereo for p in pairs]

    # ------------------------------------------------------------------
    # 2. Models to evaluate
    # ------------------------------------------------------------------
    model_names = ["bert-base-uncased", "roberta-base", "microsoft/deberta-base"]

    all_results: dict[str, dict[str, object]] = {}

    for model_name in model_names:
        logger.info("--- Evaluating %s ---", model_name)
        try:
            evaluator = BiasEvaluator(
                model_name=model_name,
                max_batch_size=32,
            )
            bias_score, detailed_results = evaluator.evaluate(
                stereotypical_sentences=stereotypic_sentences,
                anti_stereotypical_sentences=anti_stereotypic_sentences,
            )
            evaluator.cleanup()
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", model_name, exc, exc_info=True)
            bias_score = None
            detailed_results = []

        # Log summary
        if bias_score is not None:
            n_stereo_preferred = sum(1 for r in detailed_results if r["stereo_higher"])
            logger.info(
                "[Bias] %-30s | Bias Score: %.2f%% (%d / %d pairs preferred stereotype)",
                model_name,
                bias_score,
                n_stereo_preferred,
                len(detailed_results),
            )
        else:
            logger.warning("[Bias] %-30s | Evaluation FAILED", model_name)

        # Store results for JSON export
        all_results[model_name] = {
            "bias_score": bias_score,
            "n_pairs_evaluated": len(detailed_results),
            "n_stereo_preferred": (
                sum(1 for r in detailed_results if r["stereo_higher"])
                if detailed_results
                else None
            ),
            "detailed_results": detailed_results,
        }

    # ------------------------------------------------------------------
    # 3. Aggregate to JSON
    # ------------------------------------------------------------------
    os.makedirs(str(RESULTDIRS.root / "bias"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTDIRS.root / "bias" / f"bias_results_{timestamp}.json"

    # Prepare serializable summary (exclude full detailed_results for brevity)
    summary = {}
    for model_name, results in all_results.items():
        summary[model_name] = {
            "bias_score": results["bias_score"],
            "n_pairs_evaluated": results["n_pairs_evaluated"],
            "n_stereo_preferred": results["n_stereo_preferred"],
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Bias evaluation results saved to %s", json_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Orchestrate the evaluation pipeline (NLI or Hallucination detection).

    Parameters
    ----------
    argv : sequence of str, optional
        Command‑line arguments; default is ``sys.argv``.
    """
    args = _parse_args(argv)

    # Global determinism
    set_seed(RUNTIME_CONFIG.seed)

    # ------------------------------------------------------------------
    # Task routing
    # ------------------------------------------------------------------
    if args.task == "nli":
        logger.info("Task: NLI evaluation")
        splits = _resolve_splits(args.split)
        logger.info("Selected splits: %s", splits)

        if args.mode in ("prompting", "all"):
            run_prompting(splits)

        if args.mode in ("finetuning", "all"):
            run_finetuning(splits, skip_train=args.skip_train)

    elif args.task == "bias":
        logger.info("Task: Bias evaluation")
        run_bias_evaluation(n_samples=args.bias_n_samples)

    elif args.task == "hallucination":
        logger.info("Task: Hallucination detection")
        handler = NLIDataHandler()
        evaluator = HallucinationEvaluator(data_handler=handler)
        evaluator.run_evaluation()

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
