#!/usr/bin/env python3
"""
CSIT5520 NLI Evaluation Pipeline – Hallucination Detection (Section 2.2).

Implements a ``HallucinationEvaluator`` that runs two NLI pipelines on the
WikiBio-GPT3 hallucination dataset:

* **Discriminative** : Finetuned RoBERTa (checkpoint-462).
* **Generative** : Zero-shot FLAN-T5 (prompting).

NLI predictions are mapped to binary hallucination labels:

* Entailment → *Factual* (0).
* Neutral / Contradiction → *Non-Factual* (1).

Metrics (Accuracy, Precision, Recall, F1 with ``pos_label=1``) are computed
and printed to the console.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    RobertaForSequenceClassification,
)

# Add the src directory to the path so we can import local modules.
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from config import (
    FINETUNE_CONFIG,
    NLI_ID2LABEL,
    PROMPT_CONFIG,
    RESULTDIRS,
)
from data_handler import NLIDataHandler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from NLI label (string) to hallucination binary label.
# 0 = Factual, 1 = Non-Factual.
_HALLUCINATION_LABEL_MAP: Dict[str, int] = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 1,
}

# Verbalizer strings used by the FLAN-T5 prompting paradigm.
_VERBALIZER_TO_NLI: Dict[str, str] = {
    PROMPT_CONFIG.verbalizer_entailment: "entailment",
    PROMPT_CONFIG.verbalizer_neutral: "neutral",
    PROMPT_CONFIG.verbalizer_contradiction: "contradiction",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _map_nli_to_hallucination(nli_label: str) -> int:
    """Convert a three-way NLI prediction to a binary hallucination label.

    Args:
        nli_label: One of ``"entailment"``, ``"neutral"``, or
            ``"contradiction"``.

    Returns:
        0 for *Factual* (entailment) and 1 for *Non-Factual*
        (neutral / contradiction).
    """
    return _HALLUCINATION_LABEL_MAP[nli_label]


# ---------------------------------------------------------------------------
# HallucinationEvaluator
# ---------------------------------------------------------------------------


class HallucinationEvaluator:
    """Evaluate hallucination detection via NLI pipelines.

    The evaluator loads the hallucination dataset through an existing
    ``NLIDataHandler`` and provides separate routines for the
    discriminative (RoBERTa) and generative (FLAN-T5) models.

    Args:
        data_handler: An initialised ``NLIDataHandler`` instance.
        hallucination_split: Which split of the WikiBio‑GPT3 dataset to
            use (default ``"evaluation"``).
        roberta_checkpoint: Path to the fine-tuned RoBERTa checkpoint
            directory. Defaults to ``results/checkpoints/checkpoint-462``.
        batch_size: Number of examples to process at once for RoBERTa.
        flan_batch_size: Number of examples to process for FLAN‑T5
            (generation is done one‑by‑one for simplicity).
        max_prompt_length: Truncation length for the FLAN‑T5 prompt.
    """

    def __init__(
        self,
        data_handler: NLIDataHandler,
        hallucination_split: str = "evaluation",
        roberta_checkpoint: Optional[Union[str, Path]] = None,
        batch_size: int = FINETUNE_CONFIG.eval_batch_size,
        flan_batch_size: int = PROMPT_CONFIG.batch_size,
        max_prompt_length: int = PROMPT_CONFIG.max_seq_length,
    ) -> None:
        self.data_handler = data_handler

        # Device selection – MPS for Apple Silicon, CPU otherwise.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[HallucinationEvaluator] Using device: {self.device}")

        # Load hallucination data
        self.records: List[Dict[str, Union[str, int]]] = (
            self.data_handler.load_hallucination_data(split=hallucination_split)
        )
        print(
            f"[HallucinationEvaluator] Loaded {len(self.records)} "
            f"hallucination examples from split '{hallucination_split}'."
        )

        # Paths
        if roberta_checkpoint is None:
            roberta_checkpoint = RESULTDIRS.checkpoints / "checkpoint-462"
        self.roberta_checkpoint = Path(roberta_checkpoint)

        self.batch_size = batch_size
        self.flan_batch_size = flan_batch_size
        self.max_prompt_length = max_prompt_length

        # Ground-truth labels (binary: 0 = Factual, 1 = Non-Factual)
        self.y_true: List[int] = [int(rec["label"]) for rec in self.records]  # type: ignore[arg-type]

        # Predictions — populated by evaluate_roberta / evaluate_flan_t5
        self.roberta_preds: Optional[List[int]] = None
        self.flan_preds: Optional[List[int]] = None

    # ------------------------------------------------------------------
    # Discriminative pipeline – RoBERTa
    # ------------------------------------------------------------------

    def evaluate_roberta(self) -> Dict[str, float]:
        """Run hallucination detection with the fine-tuned RoBERTa model.

        Returns:
            Dictionary containing accuracy, precision, recall, and F1.
        """
        print("\n=== RoBERTa (Discriminative) ===")

        # Load model & tokenizer from checkpoint
        model = RobertaForSequenceClassification.from_pretrained(
            str(self.roberta_checkpoint)
        )
        model.to(self.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(self.roberta_checkpoint))

        # Tokenize the whole dataset
        dataset = self._build_tokenized_dataset(tokenizer)

        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        all_preds: List[int] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="RoBERTa inference"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)

        # Map NLI label ids (0=entail, 1=neutral, 2=contradiction) to binary
        y_pred: List[int] = []
        for nli_id in all_preds:
            nli_label = NLI_ID2LABEL[nli_id]
            y_pred.append(_map_nli_to_hallucination(nli_label))

        self.roberta_preds = y_pred
        return self._compute_metrics(y_pred, "RoBERTa")

    def _build_tokenized_dataset(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> Dataset:
        """Convert the loaded records into a tokenized Hugging Face Dataset.

        Args:
            tokenizer: The tokenizer to apply.

        Returns:
            A ``Dataset`` with ``input_ids`` and ``attention_mask`` columns.
        """
        premises = [rec["premise"] for rec in self.records]
        hypotheses = [rec["hypothesis"] for rec in self.records]

        tokenized = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding="max_length",
            max_length=FINETUNE_CONFIG.max_seq_length,
        )
        dataset = Dataset.from_dict(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return dataset

    # ------------------------------------------------------------------
    # Generative pipeline – FLAN-T5
    # ------------------------------------------------------------------

    def evaluate_flan_t5(self) -> Dict[str, float]:
        """Run hallucination detection with the FLAN-T5 prompting model.

        Returns:
            Dictionary containing accuracy, precision, recall, and F1.
        """
        print("\n=== FLAN-T5 (Generative) ===")

        model = AutoModelForSeq2SeqLM.from_pretrained(PROMPT_CONFIG.model_name)
        model.to(self.device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(PROMPT_CONFIG.model_name)

        # Pre-compute the token ids for each verbalizer to speed up
        # classification after generation.
        verbalizer_token_ids: Dict[str, int] = {
            nli_label: tokenizer.convert_tokens_to_ids(verbalizer)
            for verbalizer, nli_label in _VERBALIZER_TO_NLI.items()
        }

        y_pred: List[int] = []

        with torch.no_grad():
            for rec in tqdm(self.records, desc="FLAN-T5 inference"):
                prompt = self._build_prompt(str(rec["premise"]), str(rec["hypothesis"]))

                # Tokenize prompt with truncation.
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_prompt_length,
                ).to(self.device)

                # Generate a single token answer.
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                )
                # The new token is the first one after the input length.
                answer_id = generated_ids[0, -1].item()

                # Map the token id back to an NLI label via verbalizer.
                nli_label = self._decode_verbalizer(answer_id, verbalizer_token_ids)
                y_pred.append(_map_nli_to_hallucination(nli_label))

        self.flan_preds = y_pred
        return self._compute_metrics(y_pred, "FLAN-T5")

    @staticmethod
    def _build_prompt(premise: str, hypothesis: str) -> str:
        """Format the premise-hypothesis pair for FLAN-T5.

        Args:
            premise: The reference text.
            hypothesis: The generated sentence.

        Returns:
            A zero-shot NLI prompt string.
        """
        return (
            f"Premise: {premise}. Hypothesis: {hypothesis}. "
            "Entailment, neutral, or contradiction?"
        )

    @staticmethod
    def _decode_verbalizer(answer_id: int, verbalizer_token_ids: Dict[str, int]) -> str:
        """Map the generated token id to the corresponding NLI label.

        If the generated token does not match any verbalizer exactly, the
        function falls back to the most frequent class in hallucination
        detection (**Non-Factual**, i.e. ``"neutral"``).

        Args:
            answer_id: Token id produced by the model.
            verbalizer_token_ids: Mapping from NLI label string to the
                expected token id of the verbalizer.

        Returns:
            One of ``"entailment"``, ``"neutral"``, or ``"contradiction"``.
        """
        for label, token_id in verbalizer_token_ids.items():
            if token_id == answer_id:
                return label
        # Fallback: treat as non-factual (return neutral)
        return "neutral"

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, y_pred: List[int], model_name: str) -> Dict[str, float]:
        """Compute and report hallucination detection metrics.

        Args:
            y_pred: Binary predictions (0=Factual, 1=Non-Factual).
            model_name: Human-readable name for logging.

        Returns:
            Dictionary with ``accuracy``, ``precision``, ``recall``,
            and ``f1``.
        """
        acc = accuracy_score(self.y_true, y_pred)
        prec = precision_score(self.y_true, y_pred, pos_label=1)
        rec = recall_score(self.y_true, y_pred, pos_label=1)
        f1 = f1_score(self.y_true, y_pred, pos_label=1)

        print(f"\n--- {model_name} Hallucination Detection Metrics ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # ------------------------------------------------------------------
    # Case study export
    # ------------------------------------------------------------------

    def export_case_studies(
        self,
        output_dir: Union[str, Path] = "results/errors",
    ) -> None:
        """Export instance-level predictions for qualitative case study analysis.

        Builds a CSV file (``hallucination_case_studies.csv``) containing
        every evaluated premise-hypothesis pair together with the ground-
        truth label and the predictions from both models.  Rows are sorted so
        that discrepancy cases (where ``roberta_pred != flan_t5_pred`` or
        either prediction disagrees with ``true_label``) appear first.

        Args:
            output_dir: Directory where the CSV file will be written.
                Created if it does not exist.

        Raises:
            RuntimeError: If either ``roberta_preds`` or ``flan_preds``
                has not been populated by calling the evaluation methods.
        """
        if self.roberta_preds is None or self.flan_preds is None:
            raise RuntimeError(
                "Predictions are not available. Call evaluate_roberta() "
                "and evaluate_flan_t5() before exporting case studies."
            )

        output_dir = Path(output_dir)
        os.makedirs(str(output_dir), exist_ok=True)
        output_path = output_dir / "hallucination_case_studies.csv"

        # Build rows
        rows: List[Dict[str, Union[str, int]]] = []
        for i, rec in enumerate(self.records):
            true_label = int(rec["label"])  # type: ignore[arg-type]
            rows.append(
                {
                    "premise": str(rec["premise"]),
                    "hypothesis": str(rec["hypothesis"]),
                    "true_label": true_label,
                    "roberta_pred": self.roberta_preds[i],
                    "flan_t5_pred": self.flan_preds[i],
                }
            )

        # Sort: discrepancy rows first
        def _sort_key(row: Dict[str, Union[str, int]]) -> int:
            roberta_pred = int(row["roberta_pred"])
            flan_pred = int(row["flan_t5_pred"])
            true_label = int(row["true_label"])
            discrepancy = (
                roberta_pred != flan_pred
                or roberta_pred != true_label
                or flan_pred != true_label
            )
            return 0 if discrepancy else 1

        rows.sort(key=_sort_key)

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "premise",
                    "hypothesis",
                    "true_label",
                    "roberta_pred",
                    "flan_t5_pred",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        n_total = len(rows)
        n_disc = sum(
            1
            for r in rows
            if r["roberta_pred"] != r["flan_t5_pred"]
            or r["roberta_pred"] != r["true_label"]
            or r["flan_t5_pred"] != r["true_label"]
        )
        print(
            f"\n[HallucinationEvaluator] Case study export complete: "
            f"{n_total} total rows ({n_disc} with discrepancies) → "
            f"{output_path}"
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_evaluation(self) -> None:
        """Run both pipelines sequentially and display comparative results."""
        roberta_metrics = self.evaluate_roberta()
        flan_metrics = self.evaluate_flan_t5()

        print("\n========== Summary ==========")
        print(f"{'Metric':<12} {'RoBERTa':>10} {'FLAN-T5':>10}")
        for metric in ("accuracy", "precision", "recall", "f1"):
            print(
                f"{metric.capitalize():<12} "
                f"{roberta_metrics[metric]:>10.4f} "
                f"{flan_metrics[metric]:>10.4f}"
            )

        # Export case studies for qualitative analysis
        self.export_case_studies()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute hallucination detection evaluation for both NLI paradigms."""
    # Create a data handler (it will load MultiNLI files, but we only need it
    # for the hallucination data loading capability).
    handler = NLIDataHandler()

    evaluator = HallucinationEvaluator(
        data_handler=handler,
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
