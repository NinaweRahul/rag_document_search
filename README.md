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

## Hallucination Mitigation

During testing, I observed that the chatbot would generate confident-sounding answers 
for queries that fell outside the scope of the indexed documents. The model was drawing 
on its general training rather than acknowledging the limits of the source material.

To address this, I implemented two changes:

### 1. Similarity Threshold on Retrieval

Before passing retrieved chunks to the LLM, I filter for chunks with cosine similarity 
above a calibrated threshold. If no chunks meet the threshold, the system returns a 
fixed fallback response without invoking the LLM at all.

**Threshold calibration process:**
- Initial threshold set at 0.72 based on general guidance
- Tested with in-scope queries - found they were being incorrectly blocked (similarity 
scores of 0.57-0.69 for valid document questions)
- Tested with out-of-scope queries — similarity scores dropped to ~0.22
- Final threshold set at 0.45, which cleanly separates in-scope from out-of-scope queries

### 2. Grounded System Prompt

Updated the system prompt to explicitly instruct the model to answer only from the 
provided context, with a mandatory fallback phrase for cases where context is insufficient.

### Before/After Results (Deloitte State of AI 2026 Report)

| Query | Before | After |
|---|---|---|
| What % of companies moved AI to production? | Hallucinated answer | Correct: 25% moved 40%+ to production |
| What is sovereign AI? | Hallucinated answer | Correct answer from document |
| What is the capital of France? | Hallucinated answer | "I don't have that information in the source documents." |

# RAG Document Search with Gemini + ChromaDB

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NinaweRahul/rag_document_search/blob/main/rag_document_search_gemini.ipynb)
