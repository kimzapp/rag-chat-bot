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

if __name__ == "__main__":
    test_embedder()