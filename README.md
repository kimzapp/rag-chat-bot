# RAG Chatbot System

A comprehensive RAG (Retrieval-Augmented Generation) chatbot system that can answer questions based on your documents.

## Advanced System Architecture

For developers who want a complete, production-ready system:

### File Structure
```
rag-chat-bot/
├── data/
│   ├── documents/              # Your documents (PDF, TXT, DOCX)
│   └── database/               # Vector database
├── src/                        # Advanced system components
│   ├── configs/                # Configuration files
│   ├── ingestion/              # Document processing pipeline
│   ├── vectorstore/            # Vector database abstraction
│   ├── retriever/              # Advanced retrieval
│   ├── chatbot/                # Chat pipeline
│   └── api/                    # FastAPI web service
├── notebooks/                  # notebooks for experimenting with new embedding methods, llm models, etc.
└── requirements*.txt           # Dependencies
```