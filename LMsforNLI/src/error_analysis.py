"""
CSIT5520 NLI Evaluation Pipeline — Error Analysis Utility.

Provides the ``NLIErrorAnalyzer`` class for extracting and exporting
misclassified samples to CSV for qualitative case studies.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset

from config import NLI_ID2LABEL, NLI_LABEL2ID


class NLIErrorAnalyzer:
    """Extract misclassified NLI samples and export them for qualitative analysis.

    This class is intended to be instantiated (optionally with a custom label
    mapping) and then used via the ``extract_and_export_errors`` method to
    produce a CSV file ready for inclusion in academic reports.
    """

    def __init__(
        self,
        label2id: dict[str, int] = NLI_LABEL2ID,
        id2label: dict[int, str] = NLI_ID2LABEL,
    ) -> None:
        """Initialise the error analyser.

        Args:
            label2id: Mapping from human-readable label strings to integer
                class indices. Defaults to ``NLI_LABEL2ID`` from
                :mod:`config`.
            id2label: Mapping from integer class indices back to
                human-readable label strings. Defaults to ``NLI_ID2LABEL``
                from :mod:`config`.
        """
        self.label2id = label2id
        self.id2label = id2label

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_and_export_errors(
        self,
        dataset: Dataset,
        predictions: List[int],
        output_path: Path,
    ) -> pd.DataFrame:
        """Identify misclassified samples and save them as a CSV file.

        The method compares the *predictions* (list of integer class indices,
        ordered identically to the rows in *dataset*) with the ground-truth
        ``"label"`` column of the Hugging Face ``Dataset``.  Every row where
        the two differ is collected, enriched with human-readable label
        strings, and written to *output_path*.

        Args:
            dataset: A Hugging Face ``Dataset`` containing at least the
                columns ``premise``, ``hypothesis``, and ``label`` (string).
            predictions: Predicted class indices, one per example in
                *dataset*.
            output_path: Filesystem path for the resulting CSV file.

        Returns:
            A ``pandas.DataFrame`` of misclassified samples with columns:
            ``premise``, ``hypothesis``, ``gold_label`` (human-readable
            ground truth), and ``pred_label`` (human-readable prediction).

        Raises:
            ValueError: If the length of *predictions* does not match the
                number of rows in *dataset*.
        """
        if len(predictions) != len(dataset):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) does not match "
                f"dataset size ({len(dataset)})."
            )

        # Ground-truth labels as integer indices
        gold_strs: list  = dataset["label"]
        gold_ids: List[int] = [self.label2id[lbl] for lbl in gold_strs]

        # Locate misclassified indices
        error_indices: List[int] = [
            i for i, (gold, pred) in enumerate(zip(gold_ids, predictions))
            if gold != pred
        ]

        # Build error records
        records: List[dict] = []
        for idx in error_indices:
            records.append(
                {
                    "premise": dataset[idx]["premise"],
                    "hypothesis": dataset[idx]["hypothesis"],
                    "gold_label": self.id2label[gold_ids[idx]],
                    "pred_label": self.id2label[predictions[idx]],
                }
            )

        # Convert to DataFrame and persist
        df = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        return df