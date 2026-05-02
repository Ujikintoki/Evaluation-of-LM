"""
CSIT5520 NLI Evaluation Pipeline — Utility Functions.

Provides:
    * Hardware probing  (MPS > CPU)
    * Deterministic device selection
    * Standardised logging with preconfigured formatters
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from config import RESULTDIRS, RUNTIME_CONFIG


# ---------------------------------------------------------------------------
# Hardware probe — MPS prioritised per project spec
# ---------------------------------------------------------------------------


def get_device(verbose: bool = True) -> torch.device:
    """Return the best available torch device, favouring Apple MPS.

    Priority:  CUDA  >  MPS  >  CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "CUDA"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "MPS (Apple Metal Performance Shaders)"
    else:
        device = torch.device("cpu")
        backend = "CPU"

    if verbose:
        print(f"[Hardware Probe] Using device: {device}  ({backend})")
    return device


def is_mps_available() -> bool:
    """Lightweight check for MPS availability (no side effects)."""
    return torch.backends.mps.is_available()


# ---------------------------------------------------------------------------
# Standardised logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_logger_initialized: bool = False


def setup_logger(
    name: Optional[str] = None,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create / retrieve a logger with the project-wide formatter.

    On first call, a file handler writing to ``results/logs/`` is attached
    automatically  (named with a UTC timestamp)  unless *log_file* is
    explicitly provided.  Subsequent calls reuse the same file handler so a
    single run produces a single log file.

    Args:
        name: Logger name (pass ``__name__`` from the calling module).
        log_file: Explicit path for the log file.  If *None*, a timestamped
            path under ``RESULTDIRS.logs`` is used on first call.
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger`.
    """
    global _logger_initialized

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # --- Console handler (stderr) ---
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

        # --- File handler (once per process) ---
        if not _logger_initialized:
            if log_file is None:
                RESULTDIRS.logs.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                log_file = RESULTDIRS.logs / f"nli_eval_{timestamp}.log"
            else:
                log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(level)
            file_fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)

            logger.info("Log file created at %s", log_file)
            _logger_initialized = True

    return logger


# ---------------------------------------------------------------------------
# Determinism helper
# ---------------------------------------------------------------------------


def set_seed(seed: Optional[int] = None) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility.

    Args:
        seed: Integer seed.  Defaults to ``RUNTIME_CONFIG.seed``.
    """
    import random

    import numpy as np

    _seed = seed if seed is not None else RUNTIME_CONFIG.seed
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

    if RUNTIME_CONFIG.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
