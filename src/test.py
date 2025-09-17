from llms.ollama_client import OllamaClient

client = OllamaClient(model_name='gemma3_1b_base')
stream = client.answer(query="Tại sao bầu trời lại có màu xanh?", stream=True)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)