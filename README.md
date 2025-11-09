# Lung Cancer RAG: Hybrid & Pure Vector Retrieval QA

This repository contains two RAG (Retrieval-Augmented Generation) pipelines for answering lung cancer research questions:
- **Hybrid mode:** Combines ChromaDB semantic vector search and Neo4j graph traversal for multi-hop/relation-aware context selection.
- **Vector-only mode:** Provides baseline results using only embedding similarity (graph search completely disabled).

## Files
- `Graph- RAG.py` — Full hybrid pipeline (ChromaDB + Neo4j graph search).
- `RAG (No Graph impact).py` — Vector search pipeline (graph part commented out).
- `lung_cancer_fixed_FOR_RAG.json` — Parsed, normalized JSON sections/entities for ingestion.
- `Lung-Cancer-Detection-using-Supervised-Machine-Learning-Techniques.pdf` — Source paper (for reference).

**NOTE:** `chroma_db/` is *excluded* from the repo; regenerate locally by running the ingest script.

## Setup

1. Install requirements:  

pip install -r requirements.txt


2. Run either pipeline script to test.

## Requirements

- Python 3.8+
- chromadb
- neo4j
- sentence-transformers
- numpy
- requests
- (see `requirements.txt`)

## Usage

- Run with vector search only:

python "RAG (No Graph impact).py"

- Run with hybrid (graph + vector):

python "Graph- RAG.py"

- Outputs will include section sources, context, and validated QA.

## Excluded Files

- ChromaDB vector database files and all binaries are in `.gitignore`.

## Citation

Citation and clinical guidance not provided; see original paper for scientific basis.
