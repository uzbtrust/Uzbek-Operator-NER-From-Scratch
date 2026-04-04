# Multilingual NER from Scratch — BiLSTM-CRF

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NER-BiLSTM--CRF-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Languages-EN%20%2B%20RU-success?style=for-the-badge" />
</p>

<p align="center">
  A production-ready <strong>Named Entity Recognition</strong> system built entirely from scratch in PyTorch.<br/>
  No HuggingFace Transformers. No pretrained BERT. Pure BiLSTM-CRF with hand-written CRF layer.
</p>

---

## Overview

This project implements a **multilingual NER pipeline** for English and Russian, designed to power entity extraction in an operator chatbot RAG system. Every component — from the CRF's forward algorithm to the Viterbi decoder — is written from scratch.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Sentence                         │
│              "John works at Google in London"               │
└───────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────▼───────────┐
    │   Tokenize + Encode   │
    └───────────┬───────────┘
                │
  ┌─────────────▼─────────────────────────────────────┐
  │              Embedding Layer (366-dim)             │
  │  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │
  │  │  FastText   │  │  CharCNN   │  │  Language ID │ │
  │  │  Word Emb   │  │  Features  │  │  Embedding   │ │
  │  │  (300-dim)  │  │  (50-dim)  │  │   (16-dim)   │ │
  │  └────────────┘  └────────────┘  └──────────────┘ │
  └───────────────────────┬───────────────────────────┘
                          │
  ┌───────────────────────▼───────────────────────────┐
  │           BiLSTM Encoder (2 layers)               │
  │         hidden=256 × 2 directions = 512           │
  └───────────────────────┬───────────────────────────┘
                          │
  ┌───────────────────────▼───────────────────────────┐
  │              Linear Projection → 9 tags           │
  └───────────────────────┬───────────────────────────┘
                          │
  ┌───────────────────────▼───────────────────────────┐
  │               CRF Layer (from scratch)            │
  │  • Forward algorithm (log-partition via logsumexp)│
  │  • Viterbi decoding (backpointer tracing)         │
  │  • Learnable transition matrix                    │
  └───────────────────────┬───────────────────────────┘
                          │
    ┌─────────────────────▼─────────────────────────┐
    │  B-PER  O    O   B-ORG  O   B-LOC             │
    │  John  works at  Google in  London            │
    └───────────────────────────────────────────────┘
```

### Entity Types

| Tag | Description | Example |
|-----|-------------|---------|
| `PER` | Person names | *John Smith*, *Иван Петров* |
| `ORG` | Organizations | *Google*, *МегаФон* |
| `LOC` | Locations | *London*, *Москва* |
| `MISC` | Miscellaneous (tariffs, services, USSD codes) | *Unlimited*, *\*100#* |

---

## Results

### Training Curves

<p align="center">
  <img src="results/training_curves.png" alt="Training Curves" width="800"/>
</p>

### Performance Summary

| Metric | English (CoNLL-2003) | Russian (WikiANN) |
|--------|:-------------------:|:-----------------:|
| **F1** | **0.785** | **0.819** |
| Precision | 0.806 | 0.827 |
| Recall | 0.765 | 0.812 |

### Per-Entity Breakdown

#### English
| Entity | Precision | Recall | F1 | Support |
|--------|:---------:|:------:|:--:|:-------:|
| PER | 0.909 | 0.814 | **0.859** | 1,617 |
| LOC | 0.838 | 0.821 | **0.830** | 1,668 |
| ORG | 0.703 | 0.708 | **0.706** | 1,661 |
| MISC | 0.756 | 0.654 | **0.701** | 702 |

#### Russian
| Entity | Precision | Recall | F1 | Support |
|--------|:---------:|:------:|:--:|:-------:|
| PER | 0.909 | 0.900 | **0.904** | 3,543 |
| LOC | 0.832 | 0.837 | **0.834** | 4,560 |
| ORG | 0.747 | 0.707 | **0.727** | 4,074 |

### 3-Stage Training Strategy

```
Stage 1: CoNLL-2003 (EN)     ──►  F1 = 0.875  (11 epochs, lr=1e-3)
                                      │
Stage 2: WikiANN (RU)        ──►  F1 = 0.829  (26 epochs, lr=1e-4)
                                      │
Stage 3: Merged (EN+RU)      ──►  F1 = 0.799  (20 epochs, lr=1e-4)
```

---

## Project Structure

```
.
├── configs/
│   └── config.yaml              # All hyperparameters
│
├── data/
│   ├── download_datasets.py     # CoNLL-2003 + WikiANN downloader
│   ├── generate_synthetic.py    # Operator domain synthetic data
│   ├── preprocess.py            # Dataset processing + collation
│   └── vocab.py                 # Word/Char vocabularies + TagMap
│
├── embeddings/
│   └── load_fasttext.py         # FastText EN+RU download + merge
│
├── model/
│   ├── char_cnn.py              # Character-level CNN
│   ├── embedding_layer.py       # Word + Char + Lang embeddings
│   ├── bilstm.py                # Bidirectional LSTM encoder
│   ├── crf.py                   # CRF (forward algo + Viterbi)
│   └── ner_model.py             # Full BiLSTM-CRF model
│
├── training/
│   ├── train.py                 # Main training loop
│   ├── finetune_domain.py       # Operator domain fine-tuning
│   ├── evaluate.py              # Evaluation with seqeval
│   └── predict.py               # Inference + entity extraction
│
├── evaluation/
│   ├── run_evaluation.py        # Full evaluation suite
│   ├── baselines.py             # Majority / Random / Frequency
│   └── compare.py               # Model vs baseline comparison
│
├── integration/
│   └── rag_pipeline.py          # RAG query enrichment pipeline
│
├── notebooks/
│   └── train_kaggle.ipynb       # Kaggle T4 GPU training notebook
│
└── results/
    ├── training_results.json    # All metrics + training history
    └── training_curves.png      # Loss + F1 plots
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch>=2.0 datasets numpy seqeval pyyaml matplotlib
```

### 2. Download Data + Build Vocab

```bash
python data/download_datasets.py --output_dir data/raw
python data/vocab.py --raw_dir data/raw --output_dir data/processed
```

### 3. Download FastText Embeddings

```bash
python embeddings/load_fasttext.py --lang both --output_dir embeddings/vectors
```

### 4. Preprocess

```bash
python data/preprocess.py --raw_dir data/raw --vocab_dir data/processed --output_dir data/processed
```

### 5. Train (3-Stage)

```bash
# Stage 1: English
python training/train.py \
  --train_data data/processed/conll2003_train.pt \
  --val_data data/processed/conll2003_validation.pt \
  --ckpt_dir checkpoints

# Stage 2: Russian fine-tune
python training/train.py \
  --train_data data/processed/wikiann_ru_train.pt \
  --val_data data/processed/wikiann_ru_validation.pt \
  --ckpt_dir checkpoints \
  --resume checkpoints/best_model.pt \
  --lr 0.0001

# Stage 3: Merged
python training/train.py \
  --train_data data/processed/merged_train.pt \
  --val_data data/processed/merged_validation.pt \
  --ckpt_dir checkpoints \
  --resume checkpoints/best_model.pt \
  --lr 0.0001
```

### 6. Predict

```bash
python training/predict.py \
  --checkpoint checkpoints/merged_best.pt \
  --input "John Smith works at Google in London"
```

Output:
```
  John                 B-PER
  Smith                I-PER
  works                O
  at                   O
  Google               B-ORG
  in                   O
  London               B-LOC

Extracted entities:
  [PER] John Smith
  [ORG] Google
  [LOC] London
```

---

## Operator Domain Fine-tuning

Generate synthetic telecom operator data and fine-tune the model:

```bash
# Generate synthetic data
python data/generate_synthetic.py --output_dir data/raw/operator_domain --n_per_template 50

# Fine-tune from merged checkpoint
python training/finetune_domain.py \
  --checkpoint checkpoints/merged_best.pt \
  --domain_data data/raw/operator_domain \
  --lr 5e-5 \
  --epochs 20 \
  --freeze_embeddings
```

After fine-tuning, the model recognizes operator-specific entities:

```
Input: "Activate the Unlimited tariff by dialing *100#"

  [MISC] Unlimited     (tariff name)
  [MISC] *100#         (USSD code)
```

---

## RAG Integration

Use the NER engine to enrich search queries for a RAG pipeline:

```python
from integration.rag_pipeline import create_pipeline

pipeline = create_pipeline(
    config_path="configs/config.yaml",
    checkpoint_path="checkpoints/merged_best.pt",
    vocab_dir="data/processed",
)

result = pipeline.build_retrieval_context(
    "How much does the Unlimited plan cost in Moscow?"
)
```

```json
{
  "query": "How much does the Unlimited plan cost in Moscow? [MISC: Unlimited] [LOC: Moscow]",
  "filters": {
    "MISC": ["Unlimited"],
    "LOC": ["Moscow"]
  },
  "boost_terms": ["Unlimited", "Moscow"],
  "metadata": {
    "language": "en",
    "detected_entities": [...]
  }
}
```

---

## Training on Kaggle

This project was trained on **Kaggle T4 × 2 GPUs**. Open `notebooks/train_kaggle.ipynb` on Kaggle:

1. Upload the notebook to Kaggle
2. Enable **GPU T4 × 2** in Settings → Accelerator
3. Enable **Internet** in Settings
4. Click **Run All**
5. Training takes ~2-3 hours for all 3 stages

---

## Model Details

| Component | Details |
|-----------|---------|
| Word Embeddings | FastText 300-dim (EN + RU, 500K each) |
| Char Embeddings | CNN with 50 filters, kernel=3, ReLU + MaxPool |
| Language Embeddings | 16-dim learnable (EN=0, RU=1) |
| Input Dimension | 300 + 50 + 16 = **366** |
| Encoder | 2-layer BiLSTM, hidden=256 per direction |
| CRF | Full from-scratch: forward algo + Viterbi decode |
| Tags | 9 (BIO scheme): O, B/I-PER, B/I-ORG, B/I-LOC, B/I-MISC |
| Optimizer | AdamW (lr=1e-3 → 1e-4 for fine-tune) |
| Scheduler | ReduceLROnPlateau (patience=3) |
| Regularization | Dropout=0.5, weight_decay=1e-4, grad_clip=5.0 |
| Mixed Precision | FP16 with GradScaler on CUDA |
| Total Parameters | ~3.5M |

---

## Datasets

| Dataset | Language | Train | Val | Test | Source |
|---------|----------|------:|----:|-----:|--------|
| CoNLL-2003 | English | 14,041 | 3,250 | 3,453 | [HuggingFace](https://huggingface.co/datasets/conll2003) |
| WikiANN | English | 20,000 | 10,000 | 10,000 | [HuggingFace](https://huggingface.co/datasets/wikiann) |
| WikiANN | Russian | 20,000 | 10,000 | 10,000 | [HuggingFace](https://huggingface.co/datasets/wikiann) |

---

## From-Scratch Highlights

This project deliberately avoids high-level NLP libraries to demonstrate deep understanding:

- **CRF Forward Algorithm**: Log-space partition function computation using `logsumexp` for numerical stability, with proper masking for variable-length sequences
- **Viterbi Decoding**: Backpointer-based optimal path finding, handling batched sequences with different lengths
- **Character CNN**: Conv1D over character embeddings with max-pooling to capture morphological features (prefixes, suffixes, capitalization patterns)
- **Embedding Fusion**: Concatenation of word-level (FastText), character-level (CNN), and language-level (learned) representations
- **Pack/Pad Sequences**: Proper handling of variable-length inputs with `pack_padded_sequence` / `pad_packed_sequence`

---

## License

MIT
