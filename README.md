# RAG Chatbot System

A comprehensive RAG (Retrieval-Augmented Generation) chatbot system that can answer questions based on your documents.

## Setting up environment
1. `conda create -n chatbot-env python=3.10`
2. `conda activate chatbot-env`
3. `pip install -r requirements.txt`

## Advanced System Architecture

For developers who want a complete, production-ready system:

### File Structure
```
rag-chat-bot/
├── data/
│   ├── documents/              # Your documents (PDF, TXT, DOCX)
│   └── database/               # Vector database
├── src/                        # Advanced system components
│   ├── ingestion/              # Document processing pipeline
│   ├── vectorstore/            # Vector database abstraction
│   ├── retriever/              # Advanced retrieval
│   ├── chatbot/                # Chat pipeline
│   └── api/                    # FastAPI web service
├── notebooks/                  # Jupyter notebooks
└── requirements*.txt           # Dependencies
```
