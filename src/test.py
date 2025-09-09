def test_vector_store():
    from vectorstore import VectorStoreFactory
    from embedding.embedder import EmbedderFactory

    vector_store = VectorStoreFactory.create_vector_store()
    print(f"Using vector store: {type(vector_store).__name__}")
    print(vector_store.config)
    print(type(vector_store))

    # query
    query = "Can AI understand human language?"
    embedder = EmbedderFactory.create_embedder()
    query_embedding = embedder.embed_query((query)).embedding
    print(vector_store.search(query_embedding, top_k=3))

def test_embedder():
    from embedding.embedder import EmbedderFactory

    embedder = EmbedderFactory.create_embedder()
    print(f"Using embedder: {type(embedder).__name__}")
    print(embedder.config)
    print(type(embedder))

    print(embedder.embed_documents(["Hello world", "How are you?"]))
    print(embedder.embed_query("Hello world"))

def test_vector_store_and_embedder():
    from vectorstore import VectorStoreFactory
    from embedding.embedder import EmbedderFactory

    vector_store = VectorStoreFactory.create_vector_store()
    embedder = EmbedderFactory.create_embedder()

    vector_store.set_embedder(embedder)
    print(f"Using vector store: {type(vector_store).__name__}")
    print(vector_store.config)
    print(type(vector_store))

    print(embedder.embed_documents(["Hello world", "How are you?"]))
    print(embedder.embed_query("Hello world"))

def test_llm_client():
    from chatbot.llm_client import LLMClientFactory

    llm_client = LLMClientFactory.create_llm_client()
    print(f"Using LLM client: {type(llm_client).__name__}")
    print(llm_client.config)
    print(type(llm_client))

    import asyncio

    async def run():
        response = await llm_client.chat_completion([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ])
        print(response)

    asyncio.run(run())

def test_llm_service():
    from chatbot.llm_service import LLMService
    import asyncio

    async def run():
        llm_service = LLMService()
        
        try:
            print("Starting streaming response...")
            
            async for chunk in llm_service.generate_stream_response(
                query="Viết một câu chuyện ngắn về Python",
                context="Bạn là một storyteller tài năng.",
                chat_history=[
                    {"role": "user", "content": "Hi!"},
                    {"role": "assistant", "content": "Hello! How can I help you?"}
                ]
            ):
                print(chunk, end="", flush=True)
            
            print("\n\nStreaming completed!")
            
        finally:
            await llm_service.close()

    asyncio.run(run())

def test_document_loader():
    from ingestion.document_loaders import DocumentLoader

    # Test with a sample text file
    file_path = r"E:\rag-chat-bot\data\documents\example.txt"  # Ensure this file exists with some content
    loader = DocumentLoader(file_path)
    loader.process_document()
    chunks = loader.chunk_content(chunk_size=50)
    
    print(f"Loaded content from {file_path}:")
    print(loader.content)
    print("\nChunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")

def test_ingestion_pipeline():
    from pipeline.pipeline import IngestionPipeline

    ingestion_pipeline = IngestionPipeline()
    ingestion_pipeline.run(None)

def test_retrieval_pipeline():
    from pipeline.pipeline import RetrievalPipeline

    retrieval_pipeline = RetrievalPipeline()
    query = "Can machine understand human language?"
    results = retrieval_pipeline.run(query)
    print(f"Results for query '{query}':")
    for result in results:
        print(result)

def test_generation_pipeline():
    from pipeline.pipeline import GenerationPipeline

    generation_pipeline = GenerationPipeline()
    query = "Viết một câu chuyện ngắn về Python"
    context = "Bạn là một storyteller tài năng."
    import asyncio

    async def run():
        print("Starting streaming response...")
        async for chunk in generation_pipeline.run(
            query=query,
            context=context,
            chat_history=[
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello! How can I help you?"}
            ],
            stream=True
        ):
            print(chunk, end="", flush=True)
        print("\n\nStreaming completed!")

    asyncio.run(run())

if __name__ == "__main__":
    test_ingestion_pipeline()