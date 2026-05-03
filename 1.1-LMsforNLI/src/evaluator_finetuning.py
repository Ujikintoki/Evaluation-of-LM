"""
CSIT5520 NLI Evaluation Pipeline — Fine-tuning Evaluator (Paradigm B).

This module provides ``RobertaFinetuneEvaluator``, a class that fine-tunes
``roberta-base`` for 3-way NLI sequence classification on MultiNLI data and
performs evaluation on held‑out splits.

The class leverages the Hugging Face ``Trainer`` API and the shared
configuration, data handling, and utilities defined in ``src/``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from config import (
    FINETUNE_CONFIG,
    NLI_LABEL2ID,
    NLI_ID2LABEL,
    RESULTDIRS,
    RUNTIME_CONFIG,
    ensure_directories,
)
from data_handler import NLIDataHandler
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from utils import set_seed, setup_logger

logger = setup_logger(__name__)


class RobertaFinetuneEvaluator:
    """Fine-tune and evaluate a RoBERTa NLI classifier.

    The class handles:
        * Model and tokenizer instantiation, including label mappings.
        * Dataset preparation via ``NLIDataHandler``.
        * Hugging Face ``Trainer`` integration with ``TrainingArguments``
          derived from ``FinetuningConfig``.
        * Metric computation (accuracy, precision, recall, F1 – macro).
        * Automatic MPS device usage on Apple Silicon.

    Parameters
    ----------
    data_handler : NLIDataHandler
        Pre‑instantiated data handler that provides raw splits.
    model_name : str, optional
        Hugging Face model identifier.  Defaults to ``FinetuneConfig.model_name``.
    train_split : str, optional
        Name of the split to use for training (default ``"matched"``).
    eval_split : str, optional
        Name of the split to use for evaluation (default ``"mismatched"``).
    """

    def __init__(
        self,
        data_handler: NLIDataHandler,
        model_name: Optional[str] = None,
        train_split: str = "matched",
        eval_split: str = "mismatched",
    ) -> None:
        self.data_handler = data_handler
        self.train_split = train_split
        self.eval_split = eval_split

        # Ensure output directories exist
        ensure_directories()

        # --- Reproducibility ---
        set_seed(RUNTIME_CONFIG.seed)

        # --- Model name ---
        model_name = model_name or FINETUNE_CONFIG.model_name

        logger.info("Loading tokenizer for %s", model_name)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

        # RoBERTa uses a byte‑level BPE without a default padding token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token (%s)", self.tokenizer.pad_token)

        logger.info("Loading model for sequence classification (%d labels)", 3)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            label2id=NLI_LABEL2ID,
            id2label=NLI_ID2LABEL,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # --- Datasets (lazily cached after first call) ---
        self._train_dataset: Optional[Dataset] = None
        self._eval_dataset: Optional[Dataset] = None

        # --- Trainer internal state ---
        self._trainer: Optional[Trainer] = None

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    def _prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Tokenize training and evaluation splits (cached)."""
        if self._train_dataset is not None and self._eval_dataset is not None:
            return self._train_dataset, self._eval_dataset

        logger.info(
            "Tokenizing splits: train='%s', eval='%s'",
            self.train_split,
            self.eval_split,
        )
        train_ds = self.data_handler.preprocess_for_finetuning(
            self.tokenizer,
            max_length=FINETUNE_CONFIG.max_seq_length,
            split=self.train_split,
        )
        eval_ds = self.data_handler.preprocess_for_finetuning(
            self.tokenizer,
            max_length=FINETUNE_CONFIG.max_seq_length,
            split=self.eval_split,
        )

        self._train_dataset = train_ds
        self._eval_dataset = eval_ds
        return train_ds, eval_ds

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @staticmethod
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute accuracy, precision, recall, and F1 (macro average).

        Parameters
        ----------
        eval_pred : tuple
            Tuple of ``(logits, labels)`` as returned by ``Trainer.predict``.

        Returns
        -------
        dict
            Dictionary with keys ``accuracy``, ``precision_macro``,
            ``recall_macro``, ``f1_macro``.
        """
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        return {
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }

    # ------------------------------------------------------------------
    # Trainer setup
    # ------------------------------------------------------------------
    def _build_trainer(self) -> Trainer:
        """Instantiate a ``Trainer`` with the fine‑tuning configuration."""
        train_ds, eval_ds = self._prepare_datasets()

        training_args = TrainingArguments(
            output_dir=str(RESULTDIRS.checkpoints),
            logging_dir=str(RESULTDIRS.logs),
            run_name="roberta-nli",
            # --- Learning rate & optimisation ---
            learning_rate=FINETUNE_CONFIG.learning_rate,
            weight_decay=FINETUNE_CONFIG.weight_decay,
            adam_epsilon=FINETUNE_CONFIG.adam_epsilon,
            warmup_ratio=FINETUNE_CONFIG.warmup_ratio,
            optim="adamw_torch",
            # --- Batch sizes ---
            per_device_train_batch_size=FINETUNE_CONFIG.batch_size,
            per_device_eval_batch_size=FINETUNE_CONFIG.eval_batch_size,
            gradient_accumulation_steps=FINETUNE_CONFIG.gradient_accumulation_steps,
            max_grad_norm=FINETUNE_CONFIG.max_grad_norm,
            # --- Epochs & evaluation ---
            num_train_epochs=FINETUNE_CONFIG.num_epochs,
            eval_strategy="steps",
            eval_steps=FINETUNE_CONFIG.eval_steps,
            save_steps=FINETUNE_CONFIG.save_steps,
            logging_steps=FINETUNE_CONFIG.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=FINETUNE_CONFIG.save_total_limit,
            # --- Reproducibility & hardware ---
            seed=RUNTIME_CONFIG.seed,
            data_seed=RUNTIME_CONFIG.seed,
            use_mps_device=True,          # explicit Apple Silicon support
            # --- Misc ---
            disable_tqdm=False,
            report_to=["none"],  # no external logging (wandb, etc.)
        )

        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the full fine‑tuning loop.

        The trainer automatically restores the best checkpoint at the end
        (``load_best_model_at_end=True``).
        """
        logger.info("Starting fine‑tuning")
        self._trainer = self._build_trainer()
        self._trainer.train()

        # Log final evaluation metrics on the eval split
        final_metrics = self._trainer.evaluate()
        logger.info("Final eval metrics: %s", final_metrics)

    def evaluate(self, split: Optional[str] = None) -> List[int]:
        """Evaluate the fine‑tuned model on the given (or default) split.

        Parameters
        ----------
        split : str, optional
            Name of the split to evaluate.  If ``None``, the default
            ``eval_split`` is used.

        Returns
        -------
        list[int]
            Integer predictions (0, 1, or 2).
        """
        split = split or self.eval_split
        logger.info("Evaluating on split '%s'", split)

        ds = self.data_handler.preprocess_for_finetuning(
            self.tokenizer,
            max_length=FINETUNE_CONFIG.max_seq_length,
            split=split,
        )

        if self._trainer is None:
            # Direct evaluation without preceding training – use a bare
            # trainer instantiated with the current model.
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
            )
        else:
            trainer = self._trainer

        preds_output = trainer.predict(ds)
        pred_indices = np.argmax(preds_output.predictions, axis=-1).tolist()

        # Compute and log metrics if ground‑truth is available
        labels = ds["label"]
        metrics = self.compute_metrics((preds_output.predictions, np.array(labels)))
        logger.info("Evaluation on '%s' results: %s", split, metrics)

        return pred_indices

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_ground_truth(self, split: Optional[str] = None) -> List[int]:
        """Return integer labels for the given split."""
        split = split or self.eval_split
        ds = self.data_handler.get_dataset(split)
        return [NLI_LABEL2ID[label] for label in ds["label"]]
