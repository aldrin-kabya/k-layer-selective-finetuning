```markdown
# Training-Free Layer Selection for Partial Fine-Tuning

This repository contains the official implementation for "Training-Free Layer Selection for Partial Fine-Tuning of Language Models". It provides methods to select optimal layers for fine-tuning BERT (Classification) and GPT-2 (Generation) using cosine similarity of representative tokens, without requiring gradient-based training for selection.

## Installation

### 1. Environment Setup (Windows 11)

It is recommended to use Anaconda or a virtual environment.

```bash
conda create -n layer_select python=3.10
conda activate layer_select
pip install -r requirements.txt
```

### 2. Java Requirement
For the E2E NLG metrics (specifically METEOR), **Java** must be installed and added to your system path.
1. Download OpenJDK or Oracle JDK.
2. Set the `JAVA_HOME` environment variable.

### 3. Data Setup
We provide a python script to download and structure the GLUE and E2E datasets automatically.

```bash
python setup_data.py
```

This will create a `data/` directory containing the unpacked datasets and download necessary NLTK data.

## Running Experiments

### GLUE Tasks (BERT)
Supported datasets: `sst2`, `mrpc`, `qnli`, `rte`, `cola`, `mnli`, etc.
Supported methods: `highest`, `lowest`, `blockwise_highest`, `blockwise_lowest`, `random`, `full_ft`.

**Example: Run Block-wise Highest Selection on SST-2 with k=3 layers**
```bash
python run_glue.py --dataset sst2 --method blockwise_highest --k 3 --B 3 --batch_size 32
```

**Example: Run Full Fine-Tuning on MRPC**
```bash
python run_glue.py --dataset mrpc --method full_ft --epochs 5
```

Results are saved in `./results/`.

### E2E NLG Challenge (GPT-2)
**Example: Run Lowest Layer Selection with k=3**
```bash
python run_e2e.py --method lowest --k 3 --batch_size 8
```

Results are saved in `./results_e2e/`.

## Repository Structure
- `src/`: Contains model definitions (`models_bert.py`, `models_gpt2.py`), dataset loaders, and training engines.
- `run_glue.py`: CLI entry point for classification tasks.
- `run_e2e.py`: CLI entry point for generation tasks.
```