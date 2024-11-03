# Codex Project : RAG Model Development

## Overview
AIM: Develop a scalable RAG-based model to efficiently manage and accurately process the extensive, continuously updated archives and guidelines of ICRC and WHO.

## Background
The International Committee of the Red Cross (ICRC) and World Health Organization (WHO) manage vast archives and guidelines. Processing these documents is challenging due to their size, complexity and constant update. Existing large language models (LLMs) require extensive computational resources, which are often inaccessible to these organizations. AIM. This study aims to develop a scalable Retrieval-Augmented Generation (RAG) model to automate the processing, retrieval, and summarization of large document collections for the ICRC and WHO, minimizing the need for extensive computational resources.

## Findings 
A new chunking strategy was developed and applied to a dataset provided by the ICRC, consisting of 200 multilingual documents of varying quality. Various retrieval and embedding techniques were tested, with the RAG model achieving a final Hit Rate of 0.742, a Mean Reciprocal Rank of 0.688, a faithfulness score of 0.91, and a relevancy score of 0.88. The model also benefited from a vector database and fine-tuning for enhanced performance. The proposed RAG model offers an initial exploration of an efficient and scalable solution for managing large multilingual document archives. It addresses the challenges of document processing while minimizing the need for extensive computational resources This project focuses on various text processing, data chunking, and retrieval techniques, utilizing models and embeddings for advanced querying and data management. The project includes scripts for preprocessing, embedding, chunking, and evaluation, as well as configuration and requirements management.

## Codebase File Structure

The directory structure is organized as follows:

In the data_preprocessing folder, the first file is ICRC_extraction.py, which extracts text from ICRC PDFs. The central file is qa_generation.py, responsible for forming nodes by splitting text using various methods and producing a precise queries dataset.

In the retriever folder, evaluation.ipynb evaluates different retrieval methods, while graphs.py and graph_rag.ipynb explore Graph RAG. summarizer_head.ipynb and text_generation_evaluation.py assess the performance of text generation models and produce broad queries. Finally, embedding_finetuning.ipynb and finetuning.py handle the fine-tuning of embeddings.

```txt
.
├── data
│   └── ... # Directory for data storage and management
├── data_preprocessing
│   ├── agentic_chunking.py # Functions for chunking text using agentic methods
│   ├── chain_chunking.py # Functions for chunking text using "chaining" techniques
│   ├── dynamic_chunking.ipynb # Jupyter Notebook for experimenting with diverse chunking methods
│   ├── ICRC_extraction.py # Code for extracting data from International Committee of the Red Cross (ICRC) PDFs
│   ├── LDA_chunking.py # Functions for chunking text using Latent Dirichlet Allocation (LDA)
│   ├── qa_generation.py # Script for chunking data and generating a dataset of questions and answers, improving modularity, documentation, and maintainability
│   ├── semantic_chunking_pairs.py # Functions for semantic chunking to create text pairs
│   └── semantic_chunking.py # Functions for semantic chunking
├── figures
│   ├── plots.ipynb # Script for generating figures in the report
│   ├── ....png
├── retriever
│   ├── embedding_finetuning.ipynb # Jupyter Notebook for fine-tuning embeddings
│   ├── evaluation.ipynb # Jupyter Notebook for evaluating models and retrieval systems
│   ├── finetuning.py # Useful functions for fine-tuning embedding models
│   ├── graphs.py # Useful functions and classes to build Graph RAG models
│   ├── summarizer_head.ipynb # Jupyter Notebook for adding a generative LLM on top of the RAG model and evaluating the generated responses
│   ├── graph_rag.ipynb # Jupyter Notebook for exploring graph-based retrieval-augmented generation methods
│   └── text_generation_evaluation.py # Functions to evaluate the performance of the summarizer head
├── requirements.txt # File listing project dependencies
├── utils.py # Utility functions used across the project
```
