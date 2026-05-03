"""
CSIT5520 NLI Evaluation Pipeline — Global Configuration.

All hyperparameters, directory paths, and label vocabularies are centralised
through frozen dataclasses so that every module reads from a single source
of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Project root  (points to  CSIT5520_NLI_Evaluation/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataDirs:
    """Absolute paths to data folders."""

    raw: Path = PROJECT_ROOT / "data" / "raw"
    processed: Path = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class ResultDirs:
    """Absolute paths to results folders."""

    root: Path = PROJECT_ROOT / "results"
    checkpoints: Path = PROJECT_ROOT / "results" / "checkpoints"
    logs: Path = PROJECT_ROOT / "results" / "logs"


# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetFiles:
    """Absolute paths to the raw JSONL evaluation files."""

    matched: Path = PROJECT_ROOT / "data" / "raw" / "dev_matched_sampled-1.jsonl"
    mismatched: Path = PROJECT_ROOT / "data" / "raw" / "dev_mismatched_sampled-1.jsonl"


# ---------------------------------------------------------------------------
# NLI label vocabulary
# ---------------------------------------------------------------------------
NLI_LABELS: Tuple[str, ...] = ("entailment", "neutral", "contradiction")
NLI_LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(NLI_LABELS)}
NLI_ID2LABEL: dict[int, str] = {idx: label for label, idx in NLI_LABEL2ID.items()}


# ---------------------------------------------------------------------------
# Paradigm A — Zero-shot Prompting (flan-t5-base)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptingConfig:
    """Hyperparameters for the zero-shot prompting pipeline."""

    model_name: str = "google/flan-t5-base"
    max_seq_length: int = 512
    batch_size: int = 16
    num_workers: int = 2

    # Verbalizer strings — single-token words the model must produce.
    #  "Yes" (entailment), "Maybe" (neutral), "No" (contradiction) are all
    #  single tokens in T5's SentencePiece vocabulary (ids: 2163, 3836, 465).
    #  They are semantically aligned with NLI labels in FLAN instruction
    #  tuning and follow the first-transition scoring protocol.
    verbalizer_entailment: str = "Yes"
    verbalizer_neutral: str = "Maybe"
    verbalizer_contradiction: str = "No"


# ---------------------------------------------------------------------------
# Paradigm B — Fine-tuning (roberta-base)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FinetuningConfig:
    """Hyperparameters for the fine-tuning pipeline."""

    model_name: str = "roberta-base"
    max_seq_length: int = 256
    batch_size: int = 16  # training batch size
    eval_batch_size: int = 32
    num_workers: int = 2

    # Optimiser & scheduler
    learning_rate: float = 2e-5
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06

    # Training loop
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2

    # Checkpointing
    save_total_limit: int = 2
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100


# ---------------------------------------------------------------------------
# Generic runtime flags
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime behaviour that isn't specific to a single paradigm."""

    seed: int = 42
    deterministic: bool = True  # if True, enable PyTorch deterministic mode


# ---------------------------------------------------------------------------
# Convenience singleton-like accessors
# ---------------------------------------------------------------------------
DATADIRS = DataDirs()
RESULTDIRS = ResultDirs()
DATASET_FILES = DatasetFiles()
PROMPT_CONFIG = PromptingConfig()
FINETUNE_CONFIG = FinetuningConfig()
RUNTIME_CONFIG = RuntimeConfig()


def ensure_directories() -> None:
    """Create all required output directories if they do not exist."""
    os.makedirs(str(DATADIRS.processed), exist_ok=True)
    os.makedirs(str(RESULTDIRS.checkpoints), exist_ok=True)
    os.makedirs(str(RESULTDIRS.logs), exist_ok=True)
