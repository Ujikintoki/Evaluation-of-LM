# CSIT5520 NLP Evaluation Pipeline — *Hallucination Detection via Natural Language Inference &amp; Social-Bias Measurement*

<div align="center">

**Course** : CSIT5520 Natural Language Processing  
**Module**   : 1 — Task 1.1, 2.2, & 3  
**Date**     : 2026-05-05

</div>

---

## Abstract

This repository provides an end-to-end, hardware-accelerated evaluation pipeline for three complementary tasks in applied NLP:

1. **Natural Language Inference (NLI)** — two paradigms:
   * **Zero-shot Prompting** using `google/flan-t5-base` with a deterministic verbalizer (first-prediction scoring).
   * **Fine-tuning** a `roberta-base` sequence classifier on MultiNLI data.
2. **Hallucination Detection** — leveraging the NLI models to perform binary factual verification on the WikiBio‑GPT3 dataset.
3. **Social‑Bias Evaluation** — quantifying stereotype biases in Masked Language Models (BERT, RoBERTa, DeBERTa) via the Pseudo‑Log‑Likelihood (PLL) metric on the CrowS‑Pairs dataset, complemented by qualitative case studies.

All components are **entirely local** (zero external API calls) and are optimised for the **Apple Metal Performance Shaders (MPS)** backend, with aggressive unified‑memory management to prevent out‑of‑memory exceptions on Apple M2 hardware.

---

## Repository Structure

```
Evaluation-of-LM/
├── data/
│   ├── raw/
│   │   ├── dev_matched_sampled-1.jsonl      # MultiNLI matched subset (2 500 ex.)
│   │   └── dev_mismatched_sampled-1.jsonl   # MultiNLI mismatched subset (2 500 ex.)
│   └── processed/                            # Reserved for preprocessed artefacts
├── results/
│   ├── bias/                                 # Section 3: PLL results &amp; case studies
│   │   ├── bias_results_*.json
│   │   └── case_studies.md
│   ├── checkpoints/
│   │   └── checkpoint-462/                   # Best fine‑tuned RoBERTa weights
│   ├── errors/                               # Error analysis &amp; case‑study CSV exports
│   └── logs/                                 # Timestamped per‑run log files
├── src/
│   ├── config.py                             # Frozen dataclass configuration
│   ├── data_handler.py                       # Data loading, MultiNLI & CrowS‑Pairs
│   ├── evaluator_prompting.py                # Zero‑shot FLAN‑T5 evaluator
│   ├── evaluator_finetuning.py               # RoBERTa fine‑tuning &amp; evaluation
│   ├── evaluator_masked.py                   # Section 3: Masked LM evaluator (PLL)
│   ├── metrics.py                            # PLL computation &amp; token alignment
│   ├── extract_case_studies.py               # Section 3: Qualitative case‑study script
│   ├── hallucination_evaluator.py            # Hallucination detection evaluator
│   ├── error_analysis.py                     # Misclassification extractor
│   └── utils.py                              # Device probing, logging, seed
├── main.py                                   # Global CLI entry point
├── requirements.txt                          # Python dependencies
├── README.md                                 # ← This document
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
pip install -r requirements.txt
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

All hyper‑parameters and paths are centralised in
[`src/config.py`](src/config.py) through frozen dataclasses to
ensure a single source of truth:

| Dataclass | Responsibility |
|---|---|
| `DataDirs` | Raw / processed data directories |
| `ResultDirs` | Checkpoints, logs, error‑exports, and bias results |
| `DatasetFiles` | MultiNLI JSONL file paths |
| `PromptingConfig` | FLAN‑T5 model name, max length, verbalizer strings |
| `FinetuningConfig` | RoBERTa hyper‑parameters (lr, epochs, batch size, …) |
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
Frozen dataclasses that every other module imports. Changing a value here propagates everywhere.

### `src/data_handler.py` — Data Management
`NLIDataHandler` loads MultiNLI JSONL files via Hugging Face `datasets`, standardises column names (`premise`, `hypothesis`, `label`), filters invalid labels (e.g. the MultiNLI sentinel `-`), and exposes paradigm‑specific preprocessing.

For bias evaluation (Section 3), the module also provides:

- **`CrowSPair`** — a frozen dataclass representing a single minimal sentence pair with character‑level diff annotations.
- **`CrowSPairsProcessor`** — loads the CrowS‑Pairs dataset, filters by *socioeconomic status/occupation*, deterministically samples 80 pairs, and runs a character‑level diff to annotate the modified and unmodified spans between the stereotype and anti‑stereotype sentences.
- **`load_crows_pairs()`** — a convenience function returning a list of `CrowSPair` objects ready for PLL computation.

### `src/evaluator_prompting.py` — Paradigm A (Zero‑Shot Prompting)
`FlanT5PromptEvaluator` loads `google/flan-t5-base`, constructs verbalizer‑aware prompts, and performs **first‑transition scoring**: after feeding `<pad>` as the sole decoder input, it slices the logit vector to keep only the three verbalizer token IDs (`Yes` = entailment, `Maybe` = neutral, `No` = contradiction) and takes `argmax`.

### `src/evaluator_finetuning.py` — Paradigm B (Fine‑Tuning)
`RobertaFinetuneEvaluator` fine‑tunes `roberta-base` for 3‑way sequence classification using the Hugging Face `Trainer` API. It reports macro‑averaged precision, recall, and F1 alongside accuracy and always loads the best checkpoint (`checkpoint-462`).

### `src/evaluator_masked.py` — Section 3 (Masked LM Bias Evaluator)
`BiasEvaluator` wraps the loading of a Masked Language Model (e.g., `bert-base-uncased`, `roberta-base`, `microsoft/deberta-base`), performs PLL evaluation over sentence pairs, and provides an explicit `cleanup()` method that deletes the model, clears the MPS cache, and invokes garbage collection. This lifecycle is essential for evaluating multiple models sequentially on Apple M2 hardware.

### `src/metrics.py` — Pseudo‑Log‑Likelihood Computation
Implements the PLL metric from Nangia et al. (2020):

- **`compute_sentence_pll()`** — computes the PLL for a single sentence by iteratively masking each *unmodified* token, forwarding the masked sequence through the Masked LM, and accumulating log‑probabilities.
- **`identify_unmodified_tokens()`** — aligns two minimally‑contrasting sentences via **Longest Common Subsequence (LCS)** of token IDs to classify tokens as *modified (M)* or *unmodified (U)*.
- **`evaluate_bias_with_pll()`** — high‑level function returning the bias score (percentage of pairs where the stereotyping sentence obtains a higher PLL) and per‑pair details.

### `src/extract_case_studies.py` — Section 3 (Qualitative Case Studies)
A standalone script that:

1. Selects three deterministic sentence pairs from the socio‑economic CrowS‑Pairs subset.
2. Computes PLL scores for each pair under BERT, RoBERTa, and DeBERTa.
3. Writes a formatted Markdown table to `results/bias/case_studies.md` indicating per‑model scores and the majority model consensus for each pair.

### `src/hallucination_evaluator.py` — Section 2.2 (Hallucination Detection)
`HallucinationEvaluator` runs both the discriminative and generative NLI models on the WikiBio‑GPT3 dataset and maps the three‑way NLI output to binary hallucination labels:

| NLI Prediction | Hallucination Label |
|---|---|
| Entailment | **0** (Factual) |
| Neutral | **1** (Non‑Factual) |
| Contradiction | **1** (Non‑Factual) |

Metrics are computed with `pos_label=1` (hallucination = positive). The class also exports `hallucination_case_studies.csv` with discrepancy cases prioritised for qualitative analysis.

### `src/error_analysis.py` — Qualitative Analysis
`NLIErrorAnalyzer` identifies misclassified examples and writes them to CSV. Used by `main.py` after each NLI evaluation run.

### `src/utils.py` — Utilities
- `get_device()` — hardware probe (CUDA > MPS > CPU).
- `setup_logger()` — standardised logging with per‑run timestamped files.
- `set_seed()` — deterministic mode across `random`, `numpy`, and `torch`.

---

## Usage

The entry point is [`main.py`](main.py), which exposes a unified CLI with task routing via the `--task` flag.

### Phase 1 — NLI Evaluation

```bash
# Zero-shot prompting on all splits
python main.py --task nli --mode prompting --split both

# Fine-tune RoBERTa and evaluate on mismatched split
python main.py --task nli --mode finetuning --split mismatched

# Run both paradigms (default)
python main.py --task nli --mode all --split both
```

### Phase 2 — Hallucination Detection

```bash
python main.py --task hallucination
```

This executes the discriminative (RoBERTa) and generative (FLAN‑T5) pipelines sequentially, prints metrics, and produces `results/errors/hallucination_case_studies.csv`.

### Phase 3 — Bias Evaluation (Section 3)

#### 3a. Quantitative Bias Assessment

```bash
python main.py --task bias
```

This command:
- Loads the CrowS‑Pairs dataset and filters the *socioeconomic status/occupation* domain.
- Deterministically samples **80 sentence pairs** (seed = 42).
- Sequentially loads `bert-base-uncased`, `roberta-base`, and `microsoft/deberta-base` via `BiasEvaluator`.
- Computes the Pseudo‑Log‑Likelihood (PLL) for each pair under each model.
- Writes the aggregated bias scores to `results/bias/bias_results_YYYYMMDD_HHMMSS.json`.
- Logs per‑model statistics: bias score (%), number of pairs preferred.

**Important:** Between models, `BiasEvaluator.cleanup()` is called to delete the model, clear the MPS cache (`torch.mps.empty_cache()`), and trigger garbage collection. This prevents out‑of‑memory (OOM) errors on Apple M2 hardware that can occur when multiple large models occupy unified memory concurrently.

#### 3b. Qualitative Case Studies

```bash
python src/extract_case_studies.py
```

This standalone script selects **three specific sentence pairs** from the sampled data (indices 0, 1, and 2), computes the PLL for each pair under all three models, and generates a comparative Markdown table:

```
results/bias/case_studies.md
```

The table columns include **Pair ID**, **Sentence Variant** (Stereotype / Anti‑Stereotype), **BERT PLL**, **RoBERTa PLL**, **DeBERTa PLL**, and **Model Consensus** (majority preference among the three models).

### Full CLI Reference

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--task` | `nli`, `hallucination`, `bias` | `nli` | High‑level task |
| `--mode` | `prompting`, `finetuning`, `all` | `all` | Paradigm (NLI only) |
| `--split` | `matched`, `mismatched`, `both` | `both` | Data split (NLI only) |
| `--skip_train` | flag | `False` | Use existing checkpoint (finetuning only) |

---

## Results

### Phase 1 — MultiNLI Accuracy

| Model | Matched | Mismatched |
|---|---|---|
| FLAN‑T5 (zero‑shot) | — | — |
| RoBERTa (fine‑tuned) | — | — |

*(Values will be populated after running the pipeline.)*

### Phase 2 — Hallucination Detection (WikiBio‑GPT3)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| RoBERTa (discriminative) | 0.9580 | 0.9900 | 0.9914 | 0.9907 |
| FLAN‑T5 (generative) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*Metrics computed with `sklearn.metrics` using `pos_label=1` (Non‑Factual).*

### Phase 3 — Bias Evaluation (CrowS‑Pairs, Socioeconomic Domain)

| Model | Bias Score (%) | Pairs Stereotype‑Preferred | Pairs Evaluated |
|---|---|---|---|
| `bert-base-uncased` | 63.75 | 51 | 80 |
| `roberta-base` | 58.75 | 47 | 80 |
| `microsoft/deberta-base` | 47.50 | 38 | 80 |

*An unbiased model would achieve a bias score of 50 %. Scores substantially above 50 % indicate stereotype‑congruent bias; scores below 50 % indicate stereotype‑incongruent (counter‑stereotypical) bias.*

The qualitative case‑study table, saved at `results/bias/case_studies.md`, provides a per‑pair comparative view of PLL scores and model consensus.

---

## Metric & Methodology — Pseudo‑Log‑Likelihood (PLL)

The PLL metric, introduced by Nangia et al. (2020), quantifies the degree to which a Masked Language Model prefers a stereotyping sentence over its minimally‑contrasting anti‑stereotyping counterpart.

### Token Alignment via Longest Common Subsequence (LCS)

Given a sentence pair `(S_stereo, S_anti)`, both sentences are tokenized. The resulting token ID sequences are aligned using `difflib.SequenceMatcher` to find the Longest Common Subsequence. Tokens belonging to identical runs (`'equal'` blocks) are classified as **unmodified (U)**; tokens in `'replace'` or `'delete'` blocks in `S_stereo` are classified as **modified (M)**. Only the *unmodified* tokens `U` carry the contextual signal for the model’s preference.

### PLL Computation

For a sentence `S = U ∪ M`, the PLL is defined as the sum of log‑probabilities of each unmodified token `u_i ∈ U` when that token is masked:

PLL(S) = Σ_i log P(u_i | U \ {u_i}, M ; θ)

This is implemented by creating a separate masked sequence for each `u_i` (masking only that token, never special tokens such as `[CLS]`, `[SEP]`, `[PAD]`), performing a forward pass, and accumulating the log‑probability assigned to the original token at the masked position. The process is batched (`max_batch_size=32`) to improve throughput on Apple MPS hardware.

### Bias Score

For each pair, the model is said to *prefer* the stereotyping sentence if `PLL(S_stereo) > PLL(S_anti)`. The bias score is the percentage of the 80 pairs for which the stereotyping variant is preferred. An unbiased model should achieve **50%**.

---

## Hardware Acceleration &amp; Memory Management

The pipeline is designed for **Apple Silicon (M2)** and prioritises the **MPS** backend for all tensor operations:

```python
device = torch.device("mps")
```

Key optimisations for the bias evaluation task include:

- **Batched masked‑token forward passes** in `compute_sentence_pll()`, which stack multiple masked variants of a sentence into a single batched forward pass (`max_batch_size=32`). This substantially reduces kernel‑launch overhead on MPS.
- **Sequential model lifecycle**: `BiasEvaluator` loads one Masked LM at a time, evaluates it, and then invokes `cleanup()` which:
  1. Deletes the model and tokenizer references.
  2. Calls `torch.mps.empty_cache()` to release any cached allocations in unified memory.
  3. Triggers `gc.collect()` to free any circular references.
- **`torch.no_grad()` decorators** around all inference routines disable gradient computation, minimising memory footprint.
- The fine‑tuned RoBERTa checkpoint is loaded in evaluation mode with frozen parameters.

These measures prevent the Apple M2’s unified memory from being exhausted when evaluating `bert-base-uncased`, `roberta-base`, and `microsoft/deberta-base` in succession.

---

## Reproducibility

- Random seed is set globally: `seed = 42`.
- PyTorch `deterministic` mode is activated (`torch.backends.cudnn.deterministic`).
- The fine‑tuned checkpoint is versioned as `checkpoint-462`.
- Generation is performed with `do_sample=False` (greedy decoding).
- CrowS‑Pairs sampling is deterministic (fixed seed, fixed indices).
- All per‑run logs are timestamped and saved to `results/logs/`.

---

## Academic Context

This pipeline was developed as coursework for **CSIT5520 Natural Language Processing** at The Hong Kong Polytechnic University. It adheres to the project specification’s strict constraint of **zero external API calls** — all models are instantiated and inferenced locally.

### Hallucination Detection Rationale

Natural Language Inference provides a principled framework for hallucination detection: when a premise (reference text) *entails* a hypothesis (generated sentence), the sentence is considered factual; otherwise it is flagged as a potential hallucination (Honovich et al., 2022). Our mapping follows this convention by treating *entailment* as *Factual* and *neutral / contradiction* as *Non‑Factual*.

### Bias Evaluation Rationale

Masked Language Models pre‑trained on broad corpora have been shown to acquire societal biases reflected in their training data. The CrowS‑Pairs dataset (Nangia et al., 2020) operationalises bias measurement through **minimal pairs** — sentences that differ only in a small set of words that signal a stereotype. By comparing the PLL of these paired sentences, we can quantify the extent to which a model’s internal representations align with social stereotypes. The socio‑economic domain was selected for this study to examine occupational and status‑related biases that are prevalent in real‑world NLP deployments.

---

## References

- Honovich, O., Arie, R., Shaham, U., & Levy, O. (2022). *TRUE: Re-evaluating Factual Consistency Evaluation*. arXiv:2204.04991.
- Nangia, N., Vania, C., Bhalerao, R., & Bowman, S. R. (2020). *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models*. In *Proceedings of EMNLP 2020*.
- Williams, A., Nangia, N., & Bowman, S. R. (2018). *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference*. In *Proceedings of NAACL-HLT 2018*.

---

## Licence

This project is for academic use only. All pre‑trained models are used under their respective licences (Flan‑T5: Apache 2.0; RoBERTa: MIT; DeBERTa: MIT).