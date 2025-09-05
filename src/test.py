from vectorstore import get_vector_store
from configs import get_config

vector_store = get_vector_store()
print(f"Using vector store: {type(vector_store).__name__}")
print(vector_store.config)