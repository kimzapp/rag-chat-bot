from embedding.embedder import EmbedderFactory
from vectorstore import VectorStoreFactory
from chatbot.llm_service import LLMService
from ingestion.document_loaders import DocumentLoader


class BasePipeline:
    def __init__(self):
        pass

    def preprocess(self, data):
        # Default preprocessing steps
        return data

    def postprocess(self, data):
        # Default postprocessing steps
        return data

    def run(self, data):
        raise NotImplementedError("Subclasses should implement this method.")


class IngestionPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.document_loader = DocumentLoader(file_path=r'E:\rag-chat-bot\data\documents\example.txt')
        self.embedder = EmbedderFactory.create_embedder()
        self.vectorstore = VectorStoreFactory.create_vector_store()

    def run(self, data):
        print("Starting ingestion pipeline...")
        
        # load and process documents
        print("Loading and chunking documents...")
        chunked_documents = self.document_loader.process_document()
        print(f"Loaded {len(chunked_documents)} documents.")

        # embed documents
        print("Embedding documents...")
        embedding_results = self.embedder.embed_documents(chunked_documents)

        # add documents and embeddings to vector store
        import uuid
        print("Adding documents to vector store...")
        ids = [str(uuid.uuid4()) for _ in range(len(embedding_results))]
        self.vectorstore.add_documents(
            documents=[result.text for result in embedding_results],
            embeddings=[result.embedding for result in embedding_results], 
            ids=ids
        )

        print("Ingestion completed.")


class RetrievalPipeline(BasePipeline):
    def __init__(self, vectorstore):
        super().__init__()
        self.vectorstore = vectorstore

    def run(self, query):
        results = self.vectorstore.search(query)
        return results


class RAGPipeline(BasePipeline):
    def __init__(self, ingestion_pipeline, retrieval_pipeline, llm_service):
        super().__init__()
        self.ingestion_pipeline = ingestion_pipeline
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_service = llm_service

    def run(self, query):
        retrieved_docs = self.retrieval_pipeline.run(query)
        response = self.llm_service.generate(retrieved_docs, query)
        return response