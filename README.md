# 📄 AI-Powered Document Insights & Data Extraction Chat Bot

A **Retrieval-Augmented Generation (RAG)** system purpose-built for querying complex pharmaceutical documentation — packaging specifications, batch records, regulatory filings, SOPs, and quality reports. Upload a PDF, ask a question in plain English, and get an answer generated strictly from retrieved excerpts of that document — grounded, traceable, and resistant to hallucination.

This project was designed with pharmaceutical manufacturing workflows (e.g., **Pfizer**-style packaging and quality documentation) in mind, where documents are long, dense, highly technical, and mix multiple document types in a single corpus. Rather than treating every document the same, the pipeline classifies content by document type before retrieval, so a question about a "packaging specification change" is answered from packaging specs — not buried in an unrelated batch record.

Built with **LlamaIndex**, **Phi-2**, and **Gradio**.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![RAG](https://img.shields.io/badge/architecture-RAG-purple)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters for Pfizer](#why-this-matters-for-Pfizer)
- [Features](#features)
- [Architecture (RAG Pipeline)](#architecture-rag-pipeline)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Tech Stack](#tech-stack)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Design Rationale](#design-rationale)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This is a **Retrieval-Augmented Generation (RAG)** application for question-answering over pharmaceutical PDF documents. It ingests a document, extracts and OCRs its text, classifies it by document type, and indexes it into a searchable vector store. When a question is asked, the system expands the query, retrieves the most relevant chunks for that document type, and prompts an LLM to answer **using only that retrieved context** — never from the model's general knowledge. This RAG approach keeps every answer traceable back to a specific excerpt in the source document, which is critical in regulated environments where answers need to be auditable.

## Why This Matters for Pfizer

Pharmaceutical documentation — packaging specs, certificates of analysis, batch records, deviation reports, SOPs — is:

- **Long and dense**, making manual search slow and error-prone
- **Highly technical**, with domain-specific terminology that generic search tools miss
- **Mixed in type**, often bundled into a single PDF or shared drive spanning multiple document categories

This tool addresses those constraints directly:

- **Document-type classification** ensures a query about packaging configuration is answered from packaging specs, not a quality report that happens to share vocabulary.
- **Grounded, excerpt-only answers** reduce the risk of hallucinated or unsupported claims — important when answers may inform quality or compliance decisions.
- **OCR support** handles scanned/legacy documents common in long-running manufacturing and regulatory archives.
- **Metadata traceability** (`doc_type`, `doc_id`, `chunk_index`, `source_file`) keeps every retrieved chunk linked back to its exact source for review.

> ⚠️ This tool assists with information retrieval and summarization. It is not a validated system of record and should not replace formal QA/regulatory review processes.

## Features

- 📥 **PDF ingestion with OCR** — handles both digital and scanned pharmaceutical documents
- 🏷️ **Automatic document typing** — classifies content (e.g., packaging spec, batch record, SOP) for targeted retrieval
- ✂️ **Metadata-aware chunking** — every chunk retains `doc_type`, `doc_id`, `chunk_index`, and `source_file`
- 🔍 **RAG query expansion** — the LLM broadens the search query to improve recall across dense technical language
- 🎯 **Type-filtered retrieval** — searches are scoped to the most relevant document type before ranking
- 🧠 **Grounded generation** — answers are constrained to retrieved excerpts only, reducing hallucination
- 🖥️ **Gradio UI** — simple web interface for upload and Q&A, no setup beyond `pip install`

## Architecture (RAG Pipeline)

**Indexing**

```
PDF Upload
    ↓
OCR / Text Extraction
    ↓
Chunk + Tag Metadata
    ↓
LlamaIndex Vector Index
```

**Retrieval-Augmented Query**

```
User Question
    ↓
Query Expansion
    ↓
Doc-Type Classification & Filter
    ↓
Retrieve Top-4 Chunks
    ↓
Prompt Assembly (RAG Context)
    ↓
Phi-2
    ↓
Answer → Gradio
```

| # | Stage | Description |
|---|---|---|
| 1 | **Ingestion** | PDF is uploaded and processed via OCR; each document is tagged with a `doc_type` (e.g., packaging spec, batch record). |
| 2 | **Chunking** | Text is split into chunks carrying metadata (`doc_type`, `doc_id`, `chunk_index`, `source_file`). |
| 3 | **Indexing** | Chunks are embedded and stored in a LlamaIndex vector index — the retrieval backbone of the RAG pipeline. |
| 4 | **Query Expansion** | The LLM generates related search terms, combined with the original question into a `better_query`. |
| 5 | **Classification & Filtering** | The LLM identifies the most relevant `doc_type`; retrieval is scoped to matching chunks only. |
| 6 | **Retrieval** | LlamaIndex returns the top 4 relevant chunks for the `better_query`. |
| 7 | **Prompt Assembly** | Retrieved chunks are combined into a RAG context block and inserted into a prompt template. |
| 8 | **Generation** | The LLM answers using only the retrieved excerpts — no answers from outside the document. |
| 9 | **Display** | The answer is rendered in the Gradio UI. |

## Tech Stack

| Component | Role |
|---|---|
| **LlamaIndex** | RAG orchestration — document indexing, embedding storage, and retrieval |
| **Phi-2** | Query expansion, document classification, and answer generation |
| **Gradio** | Web-based user interface |
| **OCR / PDF extraction (PyMuPDF, PyPDF2, pypdf)** | Document ingestion and text extraction |

> **Note:** the notebook this was built from also imports Gemini and Anthropic LLM integrations (`llama-index-llms-gemini`, `llama-index-llms-anthropic`) alongside Phi-2. If your deployed pipeline uses one of those instead of (or in addition to) Phi-2 for generation, update this table and the "Generation" step above to reflect the actual model in use.

## Dependencies

Installs and imports below are grouped by **function** rather than by the order they appeared in the notebook, since the same packages (e.g., `llama-index`, metadata filtering, `pandas`) are reused across multiple pipeline stages. Adjust freely once you finalize your production entry points.

### RAG Framework Core

```bash
pip install -q llama-index
pip install -q llama-index-readers-file
```

```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    Document,
    Settings,
)
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.readers.file import PDFReader
```

### LLM Backends

```bash
pip install -q llama-index-llms-huggingface     # Phi-2
pip install -q llama-index-llms-gemini           # optional
pip install -q llama-index-llms-anthropic        # optional
pip install -q -U huggingface-hub==0.34.0
```

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from anthropic import Anthropic
import torch
```

### Embeddings & Retrieval

```bash
pip install -q llama-index-embeddings-huggingface
pip install -q sentence-transformers
pip install -q llama-index-retrievers-bm25       # hybrid (keyword + vector) retrieval
```

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
```

### PDF Parsing & OCR

```bash
pip install -q pypdf
pip install -q PyMuPDF
pip install -q PyPDF2
```

```python
import pymupdf as fitz  # PyMuPDF
from PyPDF2 import PdfReader
```

### Model Utilities

```bash
pip install einops
pip install accelerate
pip install langchain-text-splitters
```

### General Utilities

```bash
pip install -q python-dotenv
pip install -q nest_asyncio
```

```python
import os
import time
import uuid
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import nest_asyncio
```

### UI

```bash
pip install -U "gradio>=5.0,<6.0"
```

> **Note on `import fitz` appearing twice:** the notebook this was built from imports `pymupdf as fitz` in the PDF parsing stage and `import fitz` again later in the RAG pipeline stage — these both refer to the same PyMuPDF library. 
> 

## Usage

1. Launch the app and open the Gradio URL in your browser.
2. Upload a pharmaceutical document (e.g., a packaging specification or batch record PDF).
3. Wait for ingestion, OCR, chunking, and indexing to complete.
4. Ask a question in natural language (e.g., *"Were there any packaging configuration changes?"*).
5. Review the generated answer, grounded in the retrieved excerpts from that document.

## Design Rationale

- **Doc-type filtering** narrows the RAG search space before retrieval, improving relevance when a corpus mixes multiple pharmaceutical document types (specs, batch records, SOPs, reports).
- **Query expansion** surfaces relevant chunks even when a reviewer's phrasing doesn't closely match dense technical source text.
- **Context-restricted prompting** ("answer using ONLY the excerpts") reduces hallucination by keeping the model grounded in retrieved evidence — a key requirement for RAG systems used in regulated, quality-sensitive settings.

## Roadmap

- [ ] Multi-document / multi-file upload support
- [ ] Source citation highlighting in the UI (link answers back to exact page/chunk)
- [ ] Swappable LLM backend (Phi-2 / Gemini / Anthropic / larger open-weight models)
- [ ] Evaluation suite for retrieval and answer quality
- [ ] Audit log of queries and retrieved sources for compliance review

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
