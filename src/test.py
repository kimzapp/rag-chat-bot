def test_vector_store():
    from vectorstore import VectorStoreFactory

    vector_store = VectorStoreFactory.create_vector_store()
    print(f"Using vector store: {type(vector_store).__name__}")
    print(vector_store.config)
    print(type(vector_store))


def test_embedder():
    from embedding.embedder import EmbedderFactory

    embedder = EmbedderFactory.create_embedder()
    print(f"Using embedder: {type(embedder).__name__}")
    print(embedder.config)
    print(type(embedder))

    print(embedder.embed_documents(["Hello world", "How are you?"]))
    print(embedder.embed_query("Hello world"))

def test_vector_store_and_embbder():
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

if __name__ == "__main__":
    test_llm_client()