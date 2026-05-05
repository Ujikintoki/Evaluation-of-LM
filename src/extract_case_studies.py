"""
CSIT5520 NLI Evaluation Pipeline — Qualitative Bias Case Studies.

This script extracts exactly three socio-economic CrowS-Pairs examples,
computes their Pseudo-Log-Likelihood (PLL) scores under BERT, RoBERTa,
and DeBERTa, and writes a comparative Markdown table to
``results/bias/case_studies.md``.

Hardware target: Apple M2 (Unified Memory).  Models are evaluated
sequentially with explicit ``cleanup()`` calls to prevent OOM errors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Ensure ``src/`` is on the Python path so we can import sibling modules.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from transformers import PreTrainedModel, PreTrainedTokenizerBase  # noqa: E402

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.config import RESULTDIRS, ensure_directories  # noqa: E402
from src.data_handler import CrowSPair, load_crows_pairs  # noqa: E402
from src.evaluator_masked import BiasEvaluator  # noqa: E402
from src.metrics import compute_sentence_pll, identify_unmodified_tokens, get_device  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CASE_STUDY_INDICES: Tuple[int, ...] = (0, 1, 2)
_MODEL_NAMES: List[str] = [
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-base",
]
_OUTPUT_PATH: Path = RESULTDIRS.root / "bias" / "case_studies.md"
_BIAS_TYPE: str = "socioeconomic status/occupation"


# ---------------------------------------------------------------------------
# Helper: compute PLL for a single sentence pair
# ---------------------------------------------------------------------------
def _compute_pair_pll(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    stereo_sentence: str,
    anti_sentence: str,
    max_batch_size: int = 32,
    device: Optional[str] = None,
) -> Tuple[float, float]:
    """Return ``(stereo_pll, anti_pll)`` for a single minimal pair.

    Uses :func:`src.metrics.identify_unmodified_tokens` for token alignment
    and :func:`src.metrics.compute_sentence_pll` for scoring.

    Args:
        model: A Masked LM model loaded onto the correct device.
        tokenizer: A fast tokenizer compatible with *model*.
        stereo_sentence: The stereotyping variant.
        anti_sentence: The anti-stereotyping variant.
        max_batch_size: Maximum masked tokens per forward pass.
        device: PyTorch device string (``"mps"``, ``"cuda"``, ``"cpu"``).

    Returns:
        ``(stereo_pll, anti_pll)`` as floats.
    """
    resolved_device = get_device(device)

    # ---- tokenize ----------------------------------------------------------
    stereo_enc = tokenizer(
        stereo_sentence,
        return_tensors="pt",
        add_special_tokens=True,
    )
    anti_enc = tokenizer(
        anti_sentence,
        return_tensors="pt",
        add_special_tokens=True,
    )

    stereo_ids = stereo_enc["input_ids"][0].to(resolved_device)
    stereo_mask = stereo_enc["attention_mask"][0].to(resolved_device)
    anti_ids = anti_enc["input_ids"][0].to(resolved_device)
    anti_mask = anti_enc["attention_mask"][0].to(resolved_device)

    # ---- identify unmodified tokens ----------------------------------------
    u_stereo, _ = identify_unmodified_tokens(tokenizer, stereo_sentence, anti_sentence)
    u_anti, _ = identify_unmodified_tokens(tokenizer, anti_sentence, stereo_sentence)

    # ---- PLL ---------------------------------------------------------------
    stereo_pll = compute_sentence_pll(
        model=model,
        tokenizer=tokenizer,
        input_ids=stereo_ids,
        attention_mask=stereo_mask,
        unmodified_indices=u_stereo,
        max_batch_size=max_batch_size,
    )
    anti_pll = compute_sentence_pll(
        model=model,
        tokenizer=tokenizer,
        input_ids=anti_ids,
        attention_mask=anti_mask,
        unmodified_indices=u_anti,
        max_batch_size=max_batch_size,
    )
    return stereo_pll, anti_pll


# ---------------------------------------------------------------------------
# Helper: build Markdown table
# ---------------------------------------------------------------------------
def _generate_markdown(
    pairs: List[CrowSPair],
    results: Dict[str, Dict[int, Tuple[float, float]]],
    model_names: List[str],
) -> str:
    """Format the case-study results into a GitHub-flavoured Markdown table.

    For each of the three pairs the table contains two rows — one for the
    stereotyping variant and one for the anti-stereotyping variant.  A
    final column indicates which variant the **majority of models**
    considered more probable.

    Args:
        pairs: The three :class:`CrowSPair` instances used.
        results: Nested dict ``results[model_name][pair_index] = (stereo_pll, anti_pll)``.
        model_names: Ordered list of model display names.

    Returns:
        Complete Markdown string.
    """
    # Determine display names (shorten for readability)
    display_names: List[str] = []
    for name in model_names:
        if name == "bert-base-uncased":
            display_names.append("BERT")
        elif name == "roberta-base":
            display_names.append("RoBERTa")
        elif name == "microsoft/deberta-base":
            display_names.append("DeBERTa")
        else:
            display_names.append(name)

    # ---- header ------------------------------------------------------------
    header_cols = [
        "Pair ID",
        "Sentence Variant",
        *(f"{dn} PLL" for dn in display_names),
        "Model Consensus",
    ]
    header = "| " + " | ".join(header_cols) + " |"
    separator = "|" + "|".join("---" for _ in header_cols) + "|"

    lines: List[str] = [
        "# CrowS-Pairs Case Studies — Pseudo-Log-Likelihood Comparison",
        "",
        f"Bias domain: **{_BIAS_TYPE}**",
        "",
        "The table reports PLL scores for three minimal-pair examples.  "
        "Higher (less negative) values indicate the model finds the "
        "sentence more probable.  The **Model Consensus** column shows "
        "which variant was preferred by the majority of the three models.",
        "",
        header,
        separator,
    ]

    # ---- body --------------------------------------------------------------
    for pair in pairs:
        pair_id = pair.id
        idx = _CASE_STUDY_INDICES.index(pairs.index(pair))  # map back to 0,1,2
        # Collect preferences across models
        pref: List[str] = []
        for mname in model_names:
            stereo_pll, anti_pll = results[mname][idx]
            pref.append("stereo" if stereo_pll > anti_pll else "anti")

        # Majority vote
        vote_stereo = pref.count("stereo")
        if vote_stereo >= 2:
            consensus = "Stereotype"
        else:
            consensus = "Anti-Stereotype"

        # Stereotype row
        pll_cells = [f"{results[mname][idx][0]:.2f}" for mname in model_names]
        lines.append(
            f"| {pair_id} | Stereotype | "
            + " | ".join(pll_cells)
            + f" | {consensus} |"
        )

        # Anti-stereotype row
        pll_cells = [f"{results[mname][idx][1]:.2f}" for mname in model_names]
        lines.append(
            f"| {pair_id} | Anti-Stereotype | "
            + " | ".join(pll_cells)
            + f" | {consensus} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full case-study extraction and Markdown generation pipeline."""
    logger.info("Starting case-study extraction pipeline …")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading CrowS-Pairs for bias type '%s' …", _BIAS_TYPE)
    all_pairs = load_crows_pairs(
        bias_type=_BIAS_TYPE,
        n_samples=80,  # must be ≥ 3
        seed=42,
    )
    if len(all_pairs) < max(_CASE_STUDY_INDICES) + 1:
        raise RuntimeError(
            f"Not enough pairs in filtered dataset: {len(all_pairs)} < {max(_CASE_STUDY_INDICES) + 1}"
        )

    case_pairs: List[CrowSPair] = [all_pairs[i] for i in _CASE_STUDY_INDICES]
    logger.info(
        "Selected pairs with IDs: %s",
        [p.id for p in case_pairs],
    )

    # ------------------------------------------------------------------
    # 2. Evaluate each model sequentially
    # ------------------------------------------------------------------
    results: Dict[str, Dict[int, Tuple[float, float]]] = {}
    # results[model_name][case_index] = (stereo_pll, anti_pll)

    for model_name in _MODEL_NAMES:
        logger.info("===== Loading %s =====", model_name)
        evaluator = BiasEvaluator(model_name=model_name, max_batch_size=32)

        # Build sentence lists for evaluate()
        stereo_sents = [p.sentence_stereo for p in case_pairs]
        anti_sents = [p.sentence_antistereo for p in case_pairs]

        # The evaluate() method returns (bias_score, detailed)
        # We'll use the detailed results to extract PLL values.
        try:
            _, detailed = evaluator.evaluate(stereo_sents, anti_sents)
        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", model_name, exc)
            evaluator.cleanup()
            raise

        # Validate that we got three results
        if len(detailed) != len(case_pairs):
            raise RuntimeError(
                f"Expected {len(case_pairs)} detailed results, got {len(detailed)}"
            )

        # Map case index -> (stereo_pll, anti_pll)
        results[model_name] = {}
        for i, record in enumerate(detailed):
            results[model_name][i] = (record["stereo_pll"], record["anti_pll"])
            logger.info(
                "  Pair %d (id=%d): stereo PLL=%.3f  anti PLL=%.3f  stereo_higher=%s",
                i,
                case_pairs[i].id,
                record["stereo_pll"],
                record["anti_pll"],
                record["stereo_higher"],
            )

        # ---- cleanup before next model ------------------------------------
        evaluator.cleanup()

    # ------------------------------------------------------------------
    # 3. Generate and save Markdown
    # ------------------------------------------------------------------
    logger.info("Generating Markdown table …")
    md_content = _generate_markdown(case_pairs, results, _MODEL_NAMES)

    ensure_directories()
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(md_content, encoding="utf-8")
    logger.info("Markdown case-study table saved to %s", _OUTPUT_PATH)

    logger.info("Case-study extraction pipeline finished successfully.")


if __name__ == "__main__":
    main()