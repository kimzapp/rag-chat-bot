from abc import abstractmethod
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

    @abstractmethod
    def run(self, data):
        pass


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
    def __init__(self):
        super().__init__()
        self.embedder = EmbedderFactory.create_embedder()
        self.vectorstore = VectorStoreFactory.create_vector_store()

    def run(self, query):
        embedded_query = self.embedder.embed_query(query).embedding
        results = self.vectorstore.search(embedded_query)
        return results

class GenerationPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.llm_service = LLMService()

    def run(self, query, context, chat_history=None, stream=True):
        if stream:
            response = self.llm_service.generate_stream_response(
                query=query, context=context,
                chat_history=chat_history
            )
        else:
            response = self.llm_service.generate_response(
                query=query, context=context,
                chat_history=chat_history
            )
        return response