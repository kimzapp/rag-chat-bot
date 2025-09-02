### 1. File structure
```
chatbot-rag/
│── data/                         # Nguồn dữ liệu thô
│   ├── admissions.pdf            # Ví dụ: tài liệu tuyển sinh
│   ├── faq.csv                   # FAQ dạng bảng
│   └── website/                  # Nếu crawl từ web
│
│── configs/                      # Cấu hình dự án
│   ├── db_config.yaml            # Config cho vector DB
│   ├── model_config.yaml         # Config cho LLM/embedding
│   └── retriever_config.yaml     # Config cho retriever
│
│── src/
│   ├── ingestion/                # Pipeline nạp dữ liệu
│   │   ├── pdf_loader.py         # Đọc dữ liệu PDF
│   │   ├── web_crawler.py        # Crawl dữ liệu web
│   │   ├── text_splitter.py      # Chia nhỏ văn bản
│   │   └── embedder.py           # Sinh embedding
│   │
│   ├── vectorstore/              # Tầng lưu trữ embedding
│   │   ├── chromadb_client.py
│   │   └── faiss_client.py
│   │
│   ├── retriever/                # Truy xuất dữ liệu
│   │   └── retriever.py
│   │
│   ├── chatbot/                  # Chat pipeline
│   │   ├── prompt_template.py
│   │   ├── rag_chain.py
│   │   └── chat_service.py
│   │
│   └── api/                      # API layer (FastAPI/Flask)
│       ├── main.py
│       └── routes.py
│
│── notebooks/                    # Jupyter notebook test nhanh
│   ├── eda.ipynb
│   └── rag_demo.ipynb
│
│── tests/                        # Unit test
│   ├── test_loader.py
│   ├── test_retriever.py
│   └── test_chat.py
│
│── requirements.txt              # Python dependencies
│── README.md
└── .env                          # API keys, DB URI
```