"""
CSIT5520 NLI Evaluation Pipeline - Data Management.

Provides an ``NLIDataHandler`` class that:
    * Loads the matched / mismatched JSONL evaluation files via Hugging Face
      ``datasets``.
    * Standardises column names (``premise``, ``hypothesis``, ``label``).
    * Filters out rows whose gold label is not in the NLI label set
      (e.g. the MultiNLI sentinel ``-``).
    * Supplies paradigm-specific preprocessing:
        - Prompt construction for zero-shot Seq2Seq evaluation.
        - Dual-sequence tokenization for RoBERTa fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

from config import DATASET_FILES, FINETUNE_CONFIG, NLI_LABEL2ID

# ---------------------------------------------------------------------------
# Prompt template - kept as a module constant for consistency
# ---------------------------------------------------------------------------
_NLI_PROMPT_TEMPLATE: str = (
    "Premise: {premise}. Hypothesis: {hypothesis}. "
    "Entailment, neutral, or contradiction?"
)


class NLIDataHandler:
    """Load, standardise, and preprocess the MultiNLI evaluation subsets.

    The handler reads the two JSONL files whose paths are defined in
    ``src.config.DATASET_FILES`` and exposes methods that return
    ``DatasetDict`` or ``Dataset`` objects tailored for either zero-shot
    prompting or sequence-classification fine-tuning.
    """

    def __init__(
        self,
        matched_path: Union[str, Path] = DATASET_FILES.matched,
        mismatched_path: Union[str, Path] = DATASET_FILES.mismatched,
    ) -> None:
        """Instantiate the handler by loading both JSONL files into memory.

        Args:
            matched_path: Path to the matched JSONL file.
            mismatched_path: Path to the mismatched JSONL file.
        """
        self.matched_path: Path = Path(matched_path)
        self.mismatched_path: Path = Path(mismatched_path)

        # Load via datasets - each split gets up to 2 500 examples.
        raw: DatasetDict = load_dataset(
            "json",
            data_files={
                "matched": str(self.matched_path),
                "mismatched": str(self.mismatched_path),
            },
        )
        # Standardise column names and drop invalid labels.
        self.dataset: DatasetDict = self._standardize(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize(dataset_dict: DatasetDict) -> DatasetDict:
        """Rename raw columns and filter invalid labels.

        Column mapping:
            * ``sentence1``   -> ``premise``
            * ``sentence2``   -> ``hypothesis``
            * ``gold_label``  -> ``label``

        Rows whose label is not one of ``config.NLI_LABELS``
        (e.g. the MultiNLI sentinel ``'-'``) are silently dropped.

        Args:
            dataset_dict: Raw DatasetDict with two splits.

        Returns:
            A new DatasetDict with standardised, filtered splits.
        """
        column_map = {
            "sentence1": "premise",
            "sentence2": "hypothesis",
            "gold_label": "label",
        }
        mapped = DatasetDict()
        for split_name, ds in dataset_dict.items():
            for old, new in column_map.items():
                if old in ds.column_names:
                    ds = ds.rename_column(old, new)
            # Drop examples with invalid gold labels (e.g. '-')
            before = len(ds)
            ds = ds.filter(lambda x: x["label"] in NLI_LABEL2ID)
            after = len(ds)
            if before != after:
                print(
                    f"[NLIDataHandler] Dropped {before - after} "
                    f"examples from '{split_name}' (invalid label)"
                )
            mapped[split_name] = ds
        return mapped

    # ------------------------------------------------------------------
    # Public retrieval
    # ------------------------------------------------------------------

    def get_dataset(self, split: str) -> Dataset:
        """Return a single split by name (``"matched"`` or ``"mismatched"``)."""
        if split not in self.dataset:
            raise KeyError(
                f"Split '{split}' not found. Available: {list(self.dataset.keys())}"
            )
        return self.dataset[split]

    def get_dataset_dict(self) -> DatasetDict:
        """Return the full DatasetDict (matched + mismatched)."""
        return self.dataset

    # ------------------------------------------------------------------
    # Paradigm A - Zero-shot Prompting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(example: dict) -> dict:
        """Format a single example into the standardised NLI prompt.

        Args:
            example: A dict with ``premise``, ``hypothesis`` keys.

        Returns:
            Dict with a new ``prompt`` key containing the formatted string.
        """
        return {
            "prompt": _NLI_PROMPT_TEMPLATE.format(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
            )
        }

    def preprocess_for_prompting(
        self, split: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        """Produce a prompt column for zero-shot evaluation.

        Applies ``_build_prompt`` to every example, adding a ``prompt`` field
        that contains the full textual input for the model.

        Args:
            split: If ``None``, returns a ``DatasetDict`` with both
                ``matched`` and ``mismatched`` splits.  Otherwise, return a
                single ``Dataset`` for the requested split.

        Returns:
            A Dataset or DatasetDict with the additional ``prompt`` column.
        """
        if split is not None:
            return self.get_dataset(split).map(self._build_prompt)

        processed = DatasetDict()
        for s_name in self.dataset.keys():
            processed[s_name] = self.dataset[s_name].map(self._build_prompt)
        return processed

    # ------------------------------------------------------------------
    # Paradigm B - Fine-tuning  (dual-sequence tokenization)
    # ------------------------------------------------------------------

    def preprocess_for_finetuning(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = FINETUNE_CONFIG.max_seq_length,
        split: Optional[str] = None,
    ) -> Union[Dataset, DatasetDict]:
        """Tokenize premise-hypothesis pairs for sequence classification.

        Uses dual-sequence tokenization  (``tokenizer(premise, hypothesis)``)
        with truncation and padding to ``max_length``.  The gold label is
        encoded as an integer via ``NLI_LABEL2ID``.

        Args:
            tokenizer: A Hugging Face ``PreTrainedTokenizer``.
            max_length: Maximum sequence length after tokenization.
            split: If ``None``, return a ``DatasetDict`` with both splits;
                otherwise return the single requested split.

        Returns:
            A Dataset or DatasetDict with ``input_ids``, ``attention_mask``,
            and ``label`` (integer) columns.
        """

        def _tokenize(examples: Dict[str, list]) -> Dict[str, list]:
            tokenized = tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            tokenized["label"] = [
                NLI_LABEL2ID[lbl] for lbl in examples["label"]
            ]
            return tokenized

        # Columns to drop after tokenization
        cols_to_drop = ["premise", "hypothesis", "label"]

        if split is not None:
            ds = self.get_dataset(split)
            return ds.map(
                _tokenize,
                batched=True,
                remove_columns=[c for c in cols_to_drop if c in ds.column_names],
            )

        processed = DatasetDict()
        for s_name, ds in self.dataset.items():
            processed[s_name] = ds.map(
                _tokenize,
                batched=True,
                remove_columns=[c for c in cols_to_drop if c in ds.column_names],
            )
        return processed

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, int]:
        """Return a simple size summary for both splits."""
        return {split: len(ds) for split, ds in self.dataset.items()}
