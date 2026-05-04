"""
"Multi-Model Bias Evaluator for Masked Language Models.

Provides a :class:`BiasEvaluator` wrapper that dynamically loads Hugging Face
Masked LM models, applies the Pseudo-Log-Likelihood (PLL) metric from
:mod:`src.metrics`, and cleans up GPU / MPS memory between model runs to
respect Apple M2 unified-memory constraints.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from metrics import evaluate_bias_with_pll

logger = logging.getLogger(__name__)

__all__ = ["BiasEvaluator"]


class BiasEvaluator:
    """Load a Masked LM and evaluate stereotype bias via the PLL metric.

    This class wraps the model loading, PLL computation, and resource
    cleanup into a single lifecycle so that multiple models can be evaluated
    sequentially without exhausting shared memory on Apple Silicon devices.

    Parameters:
        model_name: Hugging Face model identifier (e.g. ``"bert-base-uncased"``).
        max_batch_size: Maximum number of masked token variants to run in a
            single forward pass during PLL computation.
        device: Device override; if ``None`` the metric auto-detects the
            best available backend (``mps`` > ``cuda`` > ``cpu``).
    """

    def __init__(
        self,
        model_name: str,
        max_batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        """Initialise the evaluator by loading the model and tokenizer.

        Args:
            model_name: Hugging Face model identifier.
            max_batch_size: Batch size for the PLL masked forward passes.
            device: Device override (``"mps"``, ``"cuda"``, ``"cpu"``, or
                ``None`` for auto-detection).

        Raises:
            RuntimeError: If the model or tokenizer cannot be loaded.
        """
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self._device_arg = device

        self.model = None
        self.tokenizer = None

        logger.info("Loading Masked LM: '%s' …", model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load tokenizer for '{model_name}'.") from exc

        try:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model for '{model_name}'.") from exc

        logger.info(
            "Successfully loaded '%s' (tokenizer fast=%s, model params=%.1fM).",
            model_name,
            self.tokenizer.is_fast,
            sum(p.numel() for p in self.model.parameters()) / 1e6,
        )

        # Re-raise if the loaded tokenizer is not fast (required by downstream code).
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"Tokenizer for '{model_name}' was not loaded as a fast tokenizer. "
                "Ensure use_fast=True and that a fast implementation is available."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        stereotypical_sentences: List[str],
        anti_stereotypical_sentences: List[str],
    ) -> Tuple[float, List[Dict[str, Union[str, float, bool]]]]:
        """Run the PLL-based bias evaluation on a set of minimal-pair sentences.

        Delegates to :func:`metrics.evaluate_bias_with_pll` using the
        loaded model and tokenizer.

        Args:
            stereotypical_sentences: List of *N* stereotyping sentences.
            anti_stereotypical_sentences: List of *N* anti-stereotyping
                sentences, aligned one-to-one with the first list.

        Returns:
            Tuple ``(bias_score, detailed_results)`` — see
            :func:`metrics.evaluate_bias_with_pll` for the schema of
            ``detailed_results``.

        Raises:
            RuntimeError: If the model or tokenizer are ``None``.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer must be loaded before calling evaluate()."
            )

        logger.info(
            "Evaluating bias for '%s' on %d sentence pairs …",
            self.model_name,
            len(stereotypical_sentences),
        )

        bias_score, detailed_results = evaluate_bias_with_pll(
            model=self.model,
            tokenizer=self.tokenizer,
            stereotypical_sentences=stereotypical_sentences,
            anti_stereotypical_sentences=anti_stereotypical_sentences,
            max_batch_size=self.max_batch_size,
            device=self._device_arg,
        )

        logger.info(
            "Bias score for '%s': %.2f%% (%d / %d pairs preferred stereotype).",
            self.model_name,
            bias_score,
            sum(1 for r in detailed_results if r["stereo_higher"]),
            len(detailed_results),
        )
        return bias_score, detailed_results

    def cleanup(self) -> None:
        """Release the model and tokenizer and clear GPU / MPS memory.

        This must be called before loading a subsequent model on
        memory-constrained Apple M2 hardware to prevent out-of-memory
        errors caused by lingering tensor allocations in unified memory.
        """
        logger.info("Cleaning up '%s' …", self.model_name)

        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.debug("MPS cache cleared.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared.")
        except Exception:
            logger.warning(
                "Exception while clearing accelerator cache (ignored).",
                exc_info=True,
            )

        # Give the garbage collector a nudge for any circular references
        import gc

        gc.collect()
        logger.info("Cleanup for '%s' complete.", self.model_name)
