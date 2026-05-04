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

Also provides a ``CrowSPairsProcessor`` class and a ``CrowSPair`` dataclass
for Section 3 – Bias evaluation using the CrowS-Pairs dataset and the
pseudo-log-likelihood metric.
"""

from __future__ import annotations

import difflib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

from config import DATASET_FILES, FINETUNE_CONFIG, NLI_LABEL2ID

logger = logging.getLogger(__name__)

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
                logger.info(
                    "[NLIDataHandler] Dropped %d examples from '%s' (invalid label)",
                    before - after,
                    split_name,
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
    # Hallucination detection data loading
    # ------------------------------------------------------------------
    def load_hallucination_data(
        self, split: str = "evaluation"
    ) -> List[Dict[str, Union[str, int]]]:
        """Load and transform the WikiBio-GPT3 hallucination dataset.

        Downloads the ``potsawee/wiki_bio_gpt3_hallucination`` dataset and
        extracts the requested split (default ``"evaluation"``).  For every entry,
        ``wiki_bio_text`` is treated as the global premise, and each
        generated sentence inside ``gpt3_sentences`` becomes an individual
        hypothesis.

        The corresponding annotation (0.0, 0.5, or 1.0) is mapped to a
        binary factual label:

        * **0.0** (Accurate / Factual) → ``0``
        * **0.5** (Minor Inaccuracy)  → ``1`` (Non-Factual)
        * **1.0** (Major Inaccuracy)  → ``1`` (Non-Factual)

        Args:
            split: The dataset split to load (default ``"evaluation"``).

        Returns:
            A list of dicts with keys ``premise`` (str), ``hypothesis``
            (str), and ``label`` (int, 0 for factual, 1 for non-factual).
        """
        dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split=split)

        records: List[Dict[str, Union[str, int]]] = []
        for entry in dataset:
            premise: str = entry["wiki_bio_text"]
            for sentence, ann in zip(entry["gpt3_sentences"], entry["annotation"]):
                label: int = 0 if ann == 0.0 else 1  # 0.5 or 1.0 → non-factual
                records.append(
                    {
                        "premise": premise,
                        "hypothesis": sentence,
                        "label": label,
                    }
                )
        return records

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
            tokenized["label"] = [NLI_LABEL2ID[lbl] for lbl in examples["label"]]
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


# ===================================================================
# Section 3 – CrowS-Pairs data pipeline  (Pseudo-log-likelihood metric)
# ===================================================================


@dataclass(frozen=True)
class CrowSPair:
    """Single processed CrowS-Pairs example with diff annotations.

    Attributes:
        id: Unique integer identifier from the CrowS-Pairs dataset.
        bias_type: The social domain (always ``"socioeconomic status/occupation"``).
        sentence_stereo: The stereotype-laden sentence.
        sentence_antistereo: The anti-stereotype counterpart.
        shared_spans: Identical contiguous character spans in both sentences,
            represented as ``(a_start, b_start, length)`` triples where
            ``a_start`` indexes ``sentence_stereo`` and ``b_start`` indexes
            ``sentence_antistereo``.
        diff_spans_stereo: Non‑overlapping ``(start, end)`` character intervals
            of ``sentence_stereo`` that differ from the anti‑stereotype text.
        diff_spans_antistereo: Non‑overlapping ``(start, end)`` character intervals
            of ``sentence_antistereo`` that differ from the stereotype text.
    """

    id: int
    bias_type: str
    sentence_stereo: str
    sentence_antistereo: str
    shared_spans: List[Tuple[int, int, int]]
    diff_spans_stereo: List[Tuple[int, int]]
    diff_spans_antistereo: List[Tuple[int, int]]


class CrowSPairsProcessor:
    """Load, filter, sample, and diff the CrowS-Pairs dataset.

    The processor is designed to prepare data for the pseudo-log-likelihood
    metric described by Nangia et al. (2020).  It extracts examples from a
    single social domain (default ``"socioeconomic status/occupation"``),
    deterministically samples a fixed number of pairs, and runs a
    character‑level diff to identify the modified and unmodified spans
    between the stereotype and anti‑stereotype sentences.

    Parameters:
        bias_type: The ``bias_type`` value to filter on.
        n_samples: Number of examples to draw after filtering.
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        bias_type: str = "socioeconomic status/occupation",
        n_samples: int = 80,
        seed: int = 42,
    ) -> None:
        self.bias_type = bias_type
        self.n_samples = n_samples
        self.seed = seed

        logger.info(
            "CrowSPairsProcessor initialised: bias_type='%s', n_samples=%d, seed=%d",
            bias_type,
            n_samples,
            seed,
        )

    # ------------------------------------------------------------------
    # Diff engine
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_diff(
        s1: str, s2: str
    ) -> Tuple[
        List[Tuple[int, int, int]],  # shared_spans
        List[Tuple[int, int]],  # diff_spans_s1
        List[Tuple[int, int]],  # diff_spans_s2
    ]:
        """Character‑level diff between two sentences.

        Uses ``difflib.SequenceMatcher`` to find the longest contiguous
        matching subsequences.  Gaps between the matching blocks are
        treated as *modified* spans (i.e. the parts unique to each
        sentence).

        Args:
            s1: First sentence (stereo or anti, depending on caller).
            s2: Second sentence.

        Returns:
            A tuple of:

            * **shared_spans**: list of ``(i, j, n)`` triples — matching
              block start in ``s1``, start in ``s2``, and length.
            * **diff_spans_1**: list of ``(start, end)`` half‑open intervals
              in ``s1`` that are *not* matched.
            * **diff_spans_2**: list of ``(start, end)`` half‑open intervals
              in ``s2`` that are *not* matched.

        Raises:
            ValueError: If the matching logic fails to produce any
                shared content (extremely unlikely for CrowS-Pairs).
        """
        matcher = difflib.SequenceMatcher(a=s1, b=s2, autojunk=False)
        blocks = matcher.get_matching_blocks()

        # The last block is a sentinel (len_a, len_b, 0); discard it.
        real_blocks = [b for b in blocks if b.size > 0]

        if not real_blocks:
            raise ValueError(f"No matching content found between:\n  {s1!r}\n  {s2!r}")

        shared: List[Tuple[int, int, int]] = [(b.a, b.b, b.size) for b in real_blocks]

        # Compute diff spans in s1 — gaps between consecutive matching blocks.
        diff_s1: List[Tuple[int, int]] = []
        prev_end = 0
        for b in real_blocks:
            if b.a > prev_end:
                diff_s1.append((prev_end, b.a))
            prev_end = b.a + b.size
        if prev_end < len(s1):
            diff_s1.append((prev_end, len(s1)))

        # Compute diff spans in s2 analogously.
        diff_s2: List[Tuple[int, int]] = []
        prev_end = 0
        for b in real_blocks:
            if b.b > prev_end:
                diff_s2.append((prev_end, b.b))
            prev_end = b.b + b.size
        if prev_end < len(s2):
            diff_s2.append((prev_end, len(s2)))

        return shared, diff_s1, diff_s2

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------
    def load_and_preprocess(self) -> List[CrowSPair]:
        """Run the complete ingestion → filtering → sampling → diff pipeline.

        Returns:
            List of :class:`CrowSPair` objects ready for metric computation.

        Raises:
            RuntimeError: If the CrowS-Pairs dataset cannot be fetched or
                the requested domain contains fewer examples than
                ``n_samples``.
        """
        # 1. Load raw dataset ------------------------------------------------
        logger.info("Downloading CrowS-Pairs dataset (if necessary) …")
        try:
            full_ds = load_dataset(
                "crows_pairs",
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the crows_pairs dataset: {exc}"
            ) from exc

        # The dataset has a single 'test' split.
        if "test" not in full_ds:
            raise RuntimeError(
                f"Expected 'test' split in crows_pairs, got {list(full_ds.keys())}"
            )
        ds = full_ds["test"]
        logger.info("CrowS-Pairs loaded: %d total examples.", len(ds))

        # 2. Filter by bias_type --------------------------------------------
        # Map our internal bias_type to the Hugging Face representation,
        # which may be the string "socioeconomic" or the ClassLabel index 1.
        hf_bias_values = {"socioeconomic", 1}
        ds = ds.filter(
            lambda row: (
                row["bias_type"] in hf_bias_values
                if self.bias_type == "socioeconomic status/occupation"
                else row["bias_type"] == self.bias_type
            )
        )
        logger.info("After filtering for '%s': %d examples.", self.bias_type, len(ds))

        if len(ds) < self.n_samples:
            raise RuntimeError(
                f"Only {len(ds)} examples available for '{self.bias_type}' "
                f"but {self.n_samples} requested."
            )

        # 3. Deterministic random sampling ------------------------------------
        rng = random.Random(self.seed)
        indices = rng.sample(range(len(ds)), self.n_samples)
        sampled = ds.select(indices)
        logger.info(
            "Deterministically sampled %d examples (seed=%d).", len(sampled), self.seed
        )

        # 4. Parse each example and run diff ---------------------------------
        pairs: List[CrowSPair] = []
        for row in sampled:
            # sent_more is always the stereotypical sentence;
            # sent_less is always the anti-stereotypical sentence.
            stereo: str = row["sent_more"]
            anti: str = row["sent_less"]

            try:
                shared_spans, diff_stereo, diff_anti = self._compute_diff(stereo, anti)
            except ValueError as exc:
                logger.warning(
                    "Diff failure for id %s ('%s' / '%s'): %s",
                    row.get("id", "?"),
                    stereo,
                    anti,
                    exc,
                )
                continue

            pair_id = int(row["id"]) if "id" in row else len(pairs)
            pairs.append(
                CrowSPair(
                    id=pair_id,
                    bias_type=self.bias_type,
                    sentence_stereo=stereo,
                    sentence_antistereo=anti,
                    shared_spans=shared_spans,
                    diff_spans_stereo=diff_stereo,
                    diff_spans_antistereo=diff_anti,
                )
            )

        logger.info(
            "Successfully processed %d / %d sampled pairs.", len(pairs), self.n_samples
        )
        return pairs


def load_crows_pairs(
    bias_type: str = "socioeconomic status/occupation",
    n_samples: int = 80,
    seed: int = 42,
) -> List[CrowSPair]:
    """Convenience function to obtain processed CrowS-Pairs examples.

    Args:
        bias_type: Bias domain filter.
        n_samples: Number of pairs to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of :class:`CrowSPair` instances ready for evaluation.
    """
    processor = CrowSPairsProcessor(bias_type=bias_type, n_samples=n_samples, seed=seed)
    return processor.load_and_preprocess()
