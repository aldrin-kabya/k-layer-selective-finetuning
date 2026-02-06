# Training-Free Layer Selection for Partial Fine-Tuning of Language Models

This repository contains the official implementation for "Training-Free Layer Selection for Partial Fine-Tuning of Language Models". It provides methods to select optimal layers for fine-tuning BERT (Classification) and GPT-2 (Generation) using cosine similarity of representative tokens, without requiring gradient-based training for selection.

## Installation

### 1. Environment Setup
The code is compatible with **Linux** (Ubuntu, CentOS, etc.), **Windows**, and **macOS**. We recommend using Conda.

```bash
# Create environment
conda create -n layer_select python=3.10
conda activate layer_select

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Java Requirement (Optional)
The E2E NLG task uses the **METEOR** metric for evaluation. This requires Java (JRE 1.8+) to be installed and available in your system path.

*   **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get update
    sudo apt-get install default-jre
    ```

*   **Windows:**
    1.  Download and install [OpenJDK](https://openjdk.org/) or Oracle Java.
    2.  Ensure `java` is added to your System PATH environment variable.

*Note: If you only plan to run GLUE (Classification) tasks, you can skip this step.*

## Data Preparation

We provide a cross-platform script to automatically download and extract the GLUE benchmark and E2E NLG Challenge dataset. This replaces manual `wget` or `tar` commands.

```bash
python setup_data.py
```
This script will:
1.  Download the GLUE datasets and extract them to `data/glue_data/`.
2.  Download the E2E dataset and extract it to `data/e2e_data/`.
3.  Download necessary NLTK data (punkt, wordnet) required for metrics.

## Usage

### 1. GLUE Benchmark (BERT)

Use `run_glue.py` to train BERT-base on classification tasks.

**Supported Datasets:**
`sst2`, `mrpc`, `qnli`, `rte`, `cola`, `sst5`, `mr`, `cr`, `mpqa`, `subj`, `trec`, `mnli`, `mnli-mm`, `snli`, `qqp`.

**Supported Methods:**
*   `highest`: Selects the $k$ layers with the highest cosine similarity.
*   `lowest`: Selects the $k$ layers with the lowest cosine similarity.
*   `blockwise_highest`: Divides model into $B$ blocks, selects highest scoring layer per block.
*   `blockwise_lowest`: Divides model into $B$ blocks, selects lowest scoring layer per block.
*   `random`: Randomly selects $k$ layers.
*   `full_ft`: Standard full fine-tuning.

**Examples:**

Run **Block-wise Highest** selection on **SST-2** (fine-tune 3 layers total, distributed across 3 blocks):
```bash
python run_glue.py --dataset sst2 --method blockwise_highest --k 3 --B 3 --batch_size 32
```

Run **Lowest** selection on **MRPC** (fine-tune 3 layers):
```bash
python run_glue.py --dataset mrpc --method lowest --k 3 --epochs 5
```

Run **Full Fine-Tuning** on **QNLI**:
```bash
python run_glue.py --dataset qnli --method full_ft
```

### 2. E2E NLG Challenge (GPT-2)

Use `run_e2e.py` to train GPT-2 Small on data-to-text generation.

**Examples:**

Run **Highest** selection with $k=3$ layers:
```bash
python run_e2e.py --method highest --k 3 --batch_size 8
```

Run **Block-wise Lowest** selection:
```bash
python run_e2e.py --method blockwise_lowest --k 3 --B 3
```

## Results

After training completes:
1.  **Console Output:** A summary table of results (Accuracy/F1 for GLUE; BLEU/NIST/METEOR/ROUGE/CIDEr for E2E) will be printed.
2.  **CSV Output:** Detailed results are saved in the `results/` (or `results_e2e/`) directory.

## Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Task name (e.g., `sst2`, `rte`) [GLUE only] | Required |
| `--method` | Layer selection strategy (`highest`, `lowest`, `blockwise...`, `full_ft`) | `highest` |
| `--k` | Number of layers to unfreeze/fine-tune | 3 |
| `--B` | Number of blocks (for blockwise methods) | 3 |
| `--batch_size` | Batch size for training/eval | 32 (BERT), 8 (GPT2) |
| `--epochs` | Number of training epochs | 5 |
| `--lr` | Learning rate | 2e-5 (BERT), 5e-5 (GPT2) |
| `--seed` | Random seed for reproducibility | 42 |
| `--output_dir` | Directory to save CSV results | `./results` |

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{biswas2025training,
  title={Training-Free Layer Selection for Partial Fine-Tuning of Language Models},
  author={Biswas, Aldrin Kabya and Fahim, Md and Fuad, Md Tahmid Hasan and Mazumder, Akm Moshiur Rahman and Ali, Amin Ahsan and Rahman, AKM Mahbubur},
  journal={arXiv preprint},
  year={2026}
}
```