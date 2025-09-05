from vectorstore import get_vector_store_from_config
from configs import get_config

vector_store = get_vector_store_from_config()
print(f"Using vector store: {type(vector_store).__name__}")
print(vector_store.config)