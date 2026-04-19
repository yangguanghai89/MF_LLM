# MF-LLM: Multi-domain Information Fusion based on Large Language Models

**MF-LLM** aims to use BGE model to generate enhanced patent embeddings by combining patent entity information and IPC classification description, and then carry out semantic similarity matching and retrieval evaluation between patents.

## 📄 Abstract

Recent advancements in dense retrieval have shown promise for patent search;  however, the lengthy nature of patent documents often exceeds the input length limitations of standard Pre-trained Language Models (PLMs), leading to significant information loss. Furthermore, conventional methods frequently fail to fully exploit the rich hierarchical semantics embedded in standardized metadata, such as the International Patent Classification (IPC), treating them as mere discrete labels rather than valuable semantic resources.
To address these challenges, this paper proposes MF-LLM, a lightweight and fine-tuning-free framework for patent dense retrieval based on multi-domain information fusion. Unlike prior art, MF-LLM leverages the generative capabilities of Large Language Models (LLMs) to enhance input representations through prompt engineering and external knowledge injection. Specifically, we employ the open-source Qwen-7B-Chat model with a zero-shot prompting strategy to automatically extract high-quality technical entities. Concurrently, we expand discrete IPC codes into full-path hierarchical descriptions based on the WIPO IPC Guide. These multi-source semantic units—comprising the original text, LLM-generated entities,  and IPC descriptions—are fused via an early fusion strategy with explicit [SEP] tokens and encoded by an efficient dual-tower architecture.
Extensive experiments on the CLEF-IP 2011 dataset demonstrate the effectiveness of our approach. The MF-LLM framework achieves a MAP of 13.03% and PRES of 38.76% under the  TOP@1000  window. Results show that MF-LLM not only significantly outperforms traditional sparse baselines like BM25 but also surpasses advanced dense retrievers such as ColBERT (achieving a 1.41 percentage point improvement in MAP) using a lightweight encoder (bge-small-en-v1.5). This validates our design philosophy that "high-quality multi-domain input representation is superior to blindly increasing model scale." Ablation studies confirm that the effective fusion of multi-domain information, particularly the incorporation of official IPC descriptions, is the key driver of performance enhancement.

## 🚀 Quick Start

### 1. Environment Setup
We recommend using Conda to manage the environment.

```bash
Create environment
conda create -n mf_llm python=3.9
conda activate mf_llm

Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Please organize your data files into the data/ directory as follows:

```text
| File | Description |
| `data/train.tsv` | Training dataset (ID, Title, Abstract, IPC, etc.) |
| `data/dev.tsv` | Development/Validation dataset |
| `data/test.tsv` | Testing dataset |
| `data/test_qrels.txt` | Ground truth labels for evaluation |
| `data/ipc.tsv` | Mapping file for IPC codes and descriptions |
```

### 3. Model Preparation
The system relies on the BGE model for embedding generation.
Model: bge-small-en-v1.5
Path: Please download the model and place it in the models/bge-small-en-v1.5/ directory.

## ⚙️ Pipeline & Usage

The project workflow consists of four main steps. Please execute them in order.

Step 1: Entity Generation
This step calls an LLM (e.g., Qwen) to extract technical entities from patent texts.
Configuration:
Modify prompt.py to adjust the prompt template (gen_entity_zh / gen_entity_en).

Run Command:
python n1_Build_Entity.py \
    --generation_type "entity" \
    --model_import_type "api" \
    --api_model_name "qwen2-7b-instruct" \
    --api_key "YOUR_API_KEY" \
    --output_path "./data/train_with_entity.tsv" \
    --language "en"

Step 2: IPC Description Enhancement
Maps IPC codes to their natural language descriptions to enrich semantic information.
Run Command:python n2_Build_IPCdescription.py

Step 3: Vector Generation
Generates embeddings by fusing Title, Abstract, IPC Descriptions, and Entities using the BGE model.
Run Command:python n3_process.py

Step 4: Evaluation
Loads the generated vectors to calculate cosine similarity, performs Top-K re-ranking, and computes metrics (Recall, MAP, etc.).
Run Command:python n4_test.py

## 📂 Project Structure

```text
MF-LLM/
├── n1_Build_Entity.py        # Step 1: Entity extraction via LLM
├── n2_Build_IPCdescription.py# Step 2: IPC code to text description
├── n3_process.py             # Step 3: Embedding generation (BGE)
├── n4_test.py                # Step 4: Evaluation script
├── prompt.py                 # Prompt templates for LLMs
├── arguments.py              # Argument parser
├── utils.py                  # Utility functions
├── call_qwen.py              # Qwen API wrapper
├── model_inference.py        # Local model inference wrapper
├── data/                     # Data directory
│   ├── train.tsv             # Original training data
│   ├── dev.tsv               # Original verification data
│   ├── test.tsv              # Original test data
│   ├── test_qrels.txt        # Correlation label (QRELS)
│   └── ipc.tsv               # IPC classification table
└── models/                   # Pre-trained model directory
│   ├──  bge-small-en-v1.5/   # BGE model
```

## 🛠️ Requirements

```text
Hardware: Linux OS (Tested on Ubuntu), GPU (NVIDIA RTX series recommended for embedding generation).
Python Libraries:
torch
sentence-transformers
pandas
numpy
tqdm
transformers
```
