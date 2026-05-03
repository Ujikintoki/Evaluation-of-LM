"""
CSIT5520 NLI Evaluation Pipeline — Zero-shot Prompting Evaluator.

Implements ``FlanT5PromptEvaluator``, a deterministic verbalizer-based
inference engine that uses ``google/flan-t5-base`` to score NLI hypotheses
without any gradient updates.

**Verbalizer protocol**:
    1. The prompt asks the model to respond with one of three single-token
       verbalizer words defined in ``PromptingConfig``.
    2. During inference, ``<pad>`` is fed as the sole ``decoder_input_ids``.
    3. The logit vector at the first generated position is sliced to keep
       only the three verbalizer token IDs.
    4. :math:`\hat{y} = \arg\max` over those three logits gives the
       predicted class index.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    NLI_ID2LABEL,
    NLI_LABEL2ID,
    NLI_LABELS,
    PROMPT_CONFIG,
)
from data_handler import NLIDataHandler
from utils import get_device, set_seed, setup_logger

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
_logger = setup_logger(__name__)

# Verbalizer lookup — matches NLI_LABELS ordering
_VERBALIZER = (
    PROMPT_CONFIG.verbalizer_entailment,    # index 0 = entailment
    PROMPT_CONFIG.verbalizer_neutral,       # index 1 = neutral
    PROMPT_CONFIG.verbalizer_contradiction, # index 2 = contradiction
)

# Prompt template that solicits the verbalizer words
_PROMPT_TEMPLATE: str = (
    "Premise: {premise}. Hypothesis: {hypothesis}. "
    "Do these statements agree? Answer Yes (entailment), "
    "Maybe (neutral), or No (contradiction)."
)


class FlanT5PromptEvaluator:
    """Zero-shot NLI evaluator using FLAN-T5 with a deterministic verbalizer.

    The evaluator:
        1. Loads ``google/flan-t5-base`` and its tokenizer.
        2. Moves the model to the device returned by ``get_device()``.
        3. Resolves the vocabulary IDs for the three single-token verbalizer
           words ("Yes", "Maybe", "No").
        4. Runs a batched forward pass using ``<pad>`` as the sole decoder
           input token, extracts logits at position 0 for the three target
           tokens, and applies :math:`\arg\max`.

    Typical usage::

        evaluator = FlanT5PromptEvaluator()
        predictions = evaluator.evaluate("matched")
        # predictions is a list[int] with values in {0, 1, 2}
    """

    def __init__(self) -> None:
        """Initialise model, tokenizer, device, and verbalizer token IDs."""
        self.model_name: str = PROMPT_CONFIG.model_name
        self.batch_size: int = PROMPT_CONFIG.batch_size
        self.max_seq_length: int = PROMPT_CONFIG.max_seq_length

        # Reproducibility
        set_seed()

        # Hardware
        self.device: torch.device = get_device(verbose=True)
        _logger.info("Device: %s", self.device)

        # Tokenizer & model
        _logger.info("Loading tokenizer for %s ...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        _logger.info("Loading model %s ...", self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Resolve verbalizer token IDs — one per NLI label
        self._verbalizer_ids: List[int] = self._resolve_verbalizer_ids()
        _logger.info("Verbalizer IDs: %s  (words: %s)", self._verbalizer_ids, _VERBALIZER)

    # ------------------------------------------------------------------
    # Verbalizer token resolution
    # ------------------------------------------------------------------

    def _resolve_verbalizer_ids(self) -> List[int]:
        """Map each verbalizer word to a single token ID in the vocabulary.

        Raises:
            ValueError: If any verbalizer word is not a single token.

        Returns:
            List[int] of vocabulary IDs in NLI_LABELS order.
        """
        ids: List[int] = []
        for label, word in zip(NLI_LABELS, _VERBALIZER):
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"Verbalizer word '{word}' for label '{label}' "
                    f"tokenizes to {len(token_ids)} tokens ({token_ids}). "
                    f"Must be a single token."
                )
            ids.append(token_ids[0])
        return ids

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _build_prompt(self, example: dict) -> dict:
        """Format a single example into the verbalizer-aware NLI prompt."""
        return {
            "prompt": _PROMPT_TEMPLATE.format(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
            )
        }

    def _build_dataloader(self, split: str) -> DataLoader:
        """Load, prompt, tokenize, and return a DataLoader for *split*.

        Args:
            split: ``"matched"`` or ``"mismatched"``.

        Returns:
            DataLoader yielding batches of (input_ids, attention_mask).
        """
        handler = NLIDataHandler()
        dataset = handler.get_dataset(split).map(self._build_prompt)

        def _collate(batch: list) -> Dict[str, torch.Tensor]:
            tokenized = self.tokenizer(
                [ex["prompt"] for ex in batch],
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            return {
                "input_ids": tokenized["input_ids"].to(self.device),
                "attention_mask": tokenized["attention_mask"].to(self.device),
            }

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, split: str) -> List[int]:
        """Run zero-shot prompting inference on the requested data split.

        Args:
            split: ``"matched"`` or ``"mismatched"``.

        Returns:
            Flat list of integer predictions  (0 = entailment,
            1 = neutral, 2 = contradiction), one per example.
        """
        _logger.info("Starting zero-shot evaluation on '%s' split.", split)
        dataloader = self._build_dataloader(split)

        # <pad> token is the standard decoder start token for T5
        pad_id: int = int(self.tokenizer.pad_token_id)
        target_ids_tensor = torch.tensor(
            self._verbalizer_ids, dtype=torch.long, device=self.device
        )  # shape: (3,)

        all_predictions: List[int] = []
        total_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Evaluating {split}", unit="batch")
        for batch in pbar:
            batch_size_actual = batch["input_ids"].size(0)

            # Decoder input: a single <pad> token per example
            decoder_input_ids = torch.full(
                (batch_size_actual, 1), pad_id,
                dtype=torch.long, device=self.device
            )

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

            # logits shape: (B, 1, vocab_size) — take position 0
            logits = outputs.logits[:, 0, :]  # (B, vocab_size)

            # Extract only the three target verbalizer token logits
            label_logits = logits.index_select(1, target_ids_tensor)  # (B, 3)

            # Argmax → predicted class index (0, 1, or 2)
            batch_preds = label_logits.argmax(dim=1)
            all_predictions.extend(batch_preds.tolist())

        _logger.info(
            "Zero-shot evaluation on '%s' complete. %d predictions.",
            split,
            len(all_predictions),
        )
        return all_predictions

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_ground_truth(self, split: str) -> List[int]:
        """Return the integer-encoded gold labels for a split.

        Args:
            split: ``"matched"`` or ``"mismatched"``.

        Returns:
            List of integer label IDs in the same order as ``evaluate()``.
        """
        handler = NLIDataHandler()
        dataset = handler.get_dataset(split)
        return [NLI_LABEL2ID[ex["label"]] for ex in dataset]
