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
│   ├── llms/                   # self-hosted llms
│   └── api/                    # FastAPI web service
├── notebooks/                  # Jupyter notebooks
└── requirements*.txt           # Dependencies
```

### Using Open-source LLMs On Local Enviromment

At the beginning, we deploy LLM using Ollama framework.

To create and use a model:
1. `ollama create [model_name] -f src/llms/models/[model_name]/Modelfile`
2. `ollama run [model_name]`

## References

[1] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., & Wang, H. (2023). Retrieval-augmented generation for large language models: A survey [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2312.10997