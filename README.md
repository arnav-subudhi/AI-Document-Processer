# RAG-Pipeline 🔍

![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-Google%20Colab-orange)
![LlamaIndex](https://img.shields.io/badge/framework-LlamaIndex-purple)
![Anthropic](https://img.shields.io/badge/LLM-Claude%20Haiku-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

A **Retrieval-Augmented Generation (RAG)** pipeline for intelligent document search and question answering. Built for general document search and pharmaceutical PDF analysis, combining hybrid retrieval, semantic embeddings, and reranking for accurate, context-aware responses.

---

## Features

- **Hybrid Retrieval** — combines vector (semantic) search and BM25 keyword search using Reciprocal Rank Fusion (RRF)
- **Semantic Chunking** — intelligently splits documents by meaning, not just character count
- **Query Expansion** — generates multiple query variations to improve retrieval coverage
- **Reranking** — uses a cross-encoder model to re-sort results by true relevance
- **Anthropic Claude Integration** — uses Claude Haiku for fast, cost-efficient generation
- **Local Embeddings** — runs HuggingFace embeddings locally, no API needed for indexing
- **PDF Support** — extracts and indexes text from multi-page PDFs using PyMuPDF

---

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LlamaIndex |
| LLM | Anthropic Claude Haiku 4.5 |
| Embeddings | `intfloat/e5-small-v2` (HuggingFace, local) |
| Keyword Search | BM25 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF Parsing | PyMuPDF (fitz) |
| Platform | Google Colab (Python 3.12) |

---

## Installation and Imports

Run the following in a Google Colab cell:

```python
!pip install -q llama-index llama-index-llms-anthropic pymupdf
!pip install -q llama-index-embeddings-huggingface
!pip install -q nest_asyncio
!pip install -q llama-index-retrievers-bm25
!pip install -q sentence-transformers

import os
import fitz
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import nest_asyncio
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
```

---

## Setup

Set your Anthropic API key:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your_api_key_here"
```

Or use Colab Secrets (recommended):
1. Click the 🔑 **Secrets** icon in the left sidebar
2. Add a secret named `ANTHROPIC_API_KEY`
3. Access it with:
```python
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get('ANTHROPIC_API_KEY')
```

---

## Usage

### 1. Index a PDF

```python
index = process_and_index_pdf("your_document.pdf")
```

### 2. Query with the Full RAG Pipeline

```python
pipeline = build_rag_pipeline(index)
response = pipeline.query("What are the storage conditions for this product?")
print(response)
```

### 3. Hybrid Retrieval Only

```python
nodes = create_hybrid_retriever(index, "What sterilization method was used?")
for i, node in enumerate(nodes):
    print(f"Result {i+1} (Score: {node.score:.4f}):")
    print(node.get_text())
```

### 4. Query Expansion

```python
expanded_query = rewrite_query("What are the penalties for late payments?")
print(expanded_query)
```

---

## Pipeline Architecture

```
PDF Input
    ↓
PyMuPDF Text Extraction
    ↓
Semantic Chunking
    ↓
HuggingFace Embeddings (local)
    ↓
Vector Index + BM25 Index
    ↓
Query Expansion (Claude)
    ↓
Hybrid Retrieval (Vector + BM25 + RRF)
    ↓
Reranking (cross-encoder)
    ↓
Claude Haiku Generation
    ↓
Response
```

---

## Author

**Arnav Subudhi** — [@arnav-subudhi](https://github.com/arnav-subudhi)
