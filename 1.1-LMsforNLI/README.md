# CSIT5520 Natural Language Processing - Project Context
## Module 1, Task 1.1: Natural Language Inference (NLI) Evaluation

### 1. Academic Objective
Develop a local, hardware-accelerated evaluation pipeline to assess the performance of Language Models on the Natural Language Inference (NLI) task. The task requires a 3-way classification: determining whether a 'hypothesis' is entailed, contradicted, or neutral given a 'premise'.
Code structer(planning):
CSIT5520_NLI_Evaluation/
├── data/                            
│   ├── raw/                         
│   │   ├── dev_matched_sampled-1.jsonl        
│   │   └── dev_mismatched_sampled-1.jsonl     
│   └── processed/                  
├── results/                        
│   ├── checkpoints/               
│   └── logs/                       
├── src/                             
│   ├── __init__.py                 
│   ├── config.py                   
│   ├── data_handler.py             
│   ├── evaluator_prompting.py       
│   ├── evaluator_finetuning.py    
│   └── utils.py                 
├── main.py                          
├── requirements.txt                
└── README.md                      

### 2. Dataset Constraints
- **Dataset**: A sampled subset of the MultiNLI corpus (5000 instances).
- **Structure**: 2500 matched instances (`matched_eval.json`) and 2500 mismatched instances (`mismatched_eval.json`).

### 3. Methodological Requirements
Two parallel paradigms must be implemented and evaluated:
- **Paradigm A (Zero-shot Prompting)**: Utilizing a Sequence-to-Sequence model (`flan-t5-base`) or Causal LM. Requires explicit prompt template construction and a deterministic verbalizer extracting target logits to calculate $\hat{y}=\arg\max_{y\in L}p(v(y))$.
- **Paradigm B (Fine-tuning)**: Utilizing a Masked Language Model (`roberta-base`). Requires updating model weights via sequence classification fine-tuning.

### 4. Strict Engineering Constraints
- **Zero API Testing**: Models must be instantiated and inferenced locally. Calling external commercial APIs (e.g., OpenAI, Gemini) for the evaluation phase is strictly prohibited and will result in academic penalty.
- **Hardware Acceleration**: The pipeline must prioritize the Apple Metal Performance Shaders (`mps`) backend via PyTorch for tensor computations.
- **Evaluation Metric**: Output must strictly report Classification Accuracy for both Matched and Mismatched datasets utilizing `sklearn.metrics.accuracy_score`.