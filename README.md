# CSIT5520 NLP Evaluation Pipeline — *Hallucination Detection via Natural Language Inference*

<div align="center">

**Course** : CSIT5520 Natural Language Processing  
**Module**   : 1 — Task 1.1 & 2.2  
**Date**     : 2025-07-17

</div>

---

## Abstract

This repository provides an end-to-end, hardware-accelerated evaluation
pipeline for Natural Language Inference (NLI) and its downstream application
to **Hallucination Detection**. Two complementary paradigms are implemented:

1. **Zero-shot Prompting** using `google/flan-t5-base` with a deterministic
   verbalizer (first-prediction scoring).
2. **Fine-tuning** a `roberta-base` sequence classifier on MultiNLI data.

Both paradigms are evaluated on the MultiNLI matched/mismatched subsets and
then leveraged to perform binary hallucination detection on the
[WikiBio-GPT3 Hallucination](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination)
dataset via a standard NLI-to-factuality mapping. The pipeline prioritises
**Apple Metal Performance Shaders (MPS)** for tensor computation and is
entirely local (zero external API calls).

---

## Repository Structure

```
Evaluation-of-LM/
├── LMsforNLI/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── dev_matched_sampled-1.jsonl      # MultiNLI matched subset (2 500 ex.)
│   │   │   └── dev_mismatched_sampled-1.jsonl   # MultiNLI mismatched subset (2 500 ex.)
│   │   └── processed/                            # Reserved for preprocessed artefacts
│   ├── results/
│   │   ├── checkpoints/
│   │   │   └── checkpoint-462/                   # Best fine-tuned RoBERTa weights
│   │   ├── errors/                               # Error analysis & case-study CSV exports
│   │   └── logs/                                 # Timestamped per-run log files
│   ├── src/
│   │   ├── config.py                             # Frozen dataclass configuration
│   │   ├── data_handler.py                       # Data loading & preprocessing
│   │   ├── evaluator_prompting.py                # Zero-shot FLAN-T5 evaluator
│   │   ├── evaluator_finetuning.py               # RoBERTa fine-tuning & evaluation
│   │   ├── hallucination_evaluator.py            # Hallucination detection evaluator
│   │   ├── error_analysis.py                     # Misclassification extractor
│   │   └── utils.py                              # Device probing, logging, seed
│   ├── main.py                                   # Global CLI entry point
│   ├── requirements.txt                          # Python dependencies
│   └── README.md                                 # Module-level task description
├── README.md                                     # ← This document
└── .gitignore
```

---

## Installation

### 1. Clone & set up environment

```bash
git clone <repo_url>
cd Evaluation-of-LM
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r LMsforNLI/requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| `torch` | >= 2.0.0 | Tensor backend (MPS / CUDA / CPU) |
| `transformers` | >= 4.36.0 | Model loading, training, tokenization |
| `datasets` | >= 2.16.0 | Hugging Face dataset access |
| `scikit-learn` | >= 1.3.0 | Metrics (Accuracy, Precision, Recall, F1) |
| `tqdm` | >= 4.66.0 | Progress bars |
| `numpy` | >= 1.24.0 | Numerical operations |

### 3. Verify hardware backend

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected output on Apple Silicon: True
```

The pipeline automatically selects the best available device in the order
**CUDA → MPS → CPU**.

---

## Configuration

All hyper-parameters and paths are centralised in
[`src/config.py`](LMsforNLI/src/config.py) through frozen dataclasses to
ensure a single source of truth:

| Dataclass | Responsibility |
|---|---|
| `DataDirs` | Raw / processed data directories |
| `ResultDirs` | Checkpoints, logs, error-exports |
| `DatasetFiles` | MultiNLI JSONL file paths |
| `PromptingConfig` | FLAN-T5 model name, max length, verbalizer strings |
| `FinetuningConfig` | RoBERTa hyper-parameters (lr, epochs, batch size, …) |
| `RuntimeConfig` | Random seed, deterministic mode |

The label vocabulary is also defined at module level:
```python
NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_LABEL2ID = {...}
NLI_ID2LABEL = {...}
```

---

## Module Descriptions

### `src/config.py` — Global Configuration
Frozen dataclasses that every other module imports. Changing a value here
propagates everywhere.

### `src/data_handler.py` — Data Management
`NLIDataHandler` loads MultiNLI JSONL files via Hugging Face `datasets`,
standardises column names (`premise`, `hypothesis`, `label`), filters invalid
labels (e.g. the MultiNLI sentinel `-`), and exposes paradigm-specific
preprocessing:

- `preprocess_for_prompting()` — build prompt strings for zero-shot evaluation.
- `preprocess_for_finetuning()` — dual-sequence tokenization for RoBERTa.
- `load_hallucination_data()` — download and transform the WikiBio-GPT3
  hallucination dataset into binary-labelled premise-hypothesis pairs.

### `src/evaluator_prompting.py` — Paradigm A (Zero-Shot Prompting)
`FlanT5PromptEvaluator` loads `google/flan-t5-base`, constructs verbalizer-aware
prompts, and performs **first-transition scoring**: after feeding `<pad>` as
the sole decoder input, it slices the logit vector to keep only the three
verbalizer token IDs (`Yes` = entailment, `Maybe` = neutral,
`No` = contradiction) and takes `argmax`.

### `src/evaluator_finetuning.py` — Paradigm B (Fine-Tuning)
`RobertaFinetuneEvaluator` fine-tunes `roberta-base` for 3-way sequence
classification using the Hugging Face `Trainer` API. It reports macro-averaged
precision, recall, and F1 alongside accuracy and always loads the best
checkpoint (`checkpoint-462`).

### `src/hallucination_evaluator.py` — Section 2.2 (Hallucination Detection)
`HallucinationEvaluator` runs both the discriminative and generative NLI
models on the WikiBio-GPT3 dataset and maps the three-way NLI output to binary
hallucination labels:

| NLI Prediction | Hallucination Label |
|---|---|
| Entailment | **0** (Factual) |
| Neutral | **1** (Non-Factual) |
| Contradiction | **1** (Non-Factual) |

Metrics are computed with `pos_label=1` (hallucination = positive). The class
also exports `hallucination_case_studies.csv` with discrepancy cases
prioritised for qualitative analysis.

### `src/error_analysis.py` — Qualitative Analysis
`NLIErrorAnalyzer` identifies misclassified examples and writes them to CSV.
Used by `main.py` after each NLI evaluation run.

### `src/utils.py` — Utilities
- `get_device()` — hardware probe (CUDA > MPS > CPU).
- `setup_logger()` — standardised logging with per-run timestamped files.
- `set_seed()` — deterministic mode across `random`, `numpy`, and `torch`.

---

## Usage

The entry point is [`LMsforNLI/main.py`](LMsforNLI/main.py), which exposes a
unified CLI with task routing via the `--task` flag.

### Phase 1 — NLI Evaluation

```bash
# Zero-shot prompting on all splits
python LMsforNLI/main.py --task nli --mode prompting --split both

# Fine-tune RoBERTa and evaluate on mismatched split
python LMsforNLI/main.py --task nli --mode finetuning --split mismatched

# Run both paradigms (default)
python LMsforNLI/main.py --task nli --mode all --split both
```

### Phase 2 — Hallucination Detection

```bash
python LMsforNLI/main.py --task hallucination
```

This executes the discriminative (RoBERTa) and generative (FLAN-T5) pipelines
sequentially, prints metrics, and produces
`results/errors/hallucination_case_studies.csv`.

### Full CLI Reference

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--task` | `nli`, `hallucination` | `nli` | High-level task |
| `--mode` | `prompting`, `finetuning`, `all` | `all` | Paradigm (NLI only) |
| `--split` | `matched`, `mismatched`, `both` | `both` | Data split (NLI only) |
| `--skip_train` | flag | `False` | Use existing checkpoint (finetuning only) |

---

## Results

### Phase 1 — MultiNLI Accuracy

| Model | Matched | Mismatched |
|---|---|---|
| FLAN-T5 (zero-shot) | — | — |
| RoBERTa (fine-tuned) | — | — |

*(Values will be populated after running the pipeline.)*

### Phase 2 — Hallucination Detection (WikiBio-GPT3)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| RoBERTa (discriminative) | 0.9580 | 0.9900 | 0.9914 | 0.9907 |
| FLAN-T5 (generative) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*Metrics computed with `sklearn.metrics` using `pos_label=1` (Non-Factual).*

---

## Hardware Acceleration

The pipeline prioritises **MPS** on Apple Silicon:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

All inference is strictly wrapped in `torch.no_grad()` to prevent memory
leakage during evaluation.

---

## Reproducibility

- Random seed is set globally: `seed = 42`.
- PyTorch `deterministic` mode is activated (`torch.backends.cudnn.deterministic`).
- The fine-tuned checkpoint is versioned as `checkpoint-462`.
- Generation is performed with `do_sample=False` (greedy decoding).

---

## Academic Context

This pipeline was developed as coursework for **CSIT5520 Natural Language
Processing** at The Hong Kong Polytechnic University. It adheres to the
project specification's strict constraint of **zero external API calls** —
all models are instantiated and inferenced locally.

### Hallucination Detection Rationale

Natural Language Inference provides a principled framework for hallucination
detection: when a premise (reference text) *entails* a hypothesis (generated
sentence), the sentence is considered factual; otherwise it is flagged as a
potential hallucination (Honovich et al., 2022). Our mapping follows this
convention by treating *entailment* as *Factual* and *neutral / contradiction*
as *Non-Factual*.

---

## Licence

This project is for academic use only. All pre-trained models are used under
their respective licences (Flan-T5: Apache 2.0; RoBERTa: MIT).
