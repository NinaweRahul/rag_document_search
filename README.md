# RAG Document Search with Gemini + ChromaDB

A private document Q&A app that lets you upload any PDF and ask questions 
about it in natural language, with full conversation memory for follow-up questions.

## Tech Stack
- Python
- LangChain (LCEL pipeline)
- ChromaDB (local vector database)
- Google Gemini API (embeddings + chat model)
- Google Colab

## Features
- Dynamic file upload
- Semantic search via Gemini vector embeddings
- Conversation memory for context-aware follow-up questions
- Built using modern LCEL instead of deprecated LangChain chains
- API key secured via Colab Secrets

## How to Run
1. Open the notebook in Google Colab
2. Add your Gemini API key to Colab Secrets as `GEMINI_API_KEY`
3. Run all cells in order
4. Upload any PDF when prompted
5. Start asking questions

# RAG Document Search with Gemini + ChromaDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NinaweRahul/rag_document_search/blob/main/rag_document_search_gemini.ipynb)
